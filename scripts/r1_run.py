"""OUROBOROS — R1 headless runner (any Linux GPU box, e.g. a RunPod pod).

Runs ONE arm trajectory end to end as a plain script — no notebook, no
google.colab, no Google Drive. It mirrors the Colab notebook's
orchestration exactly: the driver still runs one stage per subprocess
(fresh process + re-seeded RNG per stage, as on Colab), Ollama is started
as a subprocess if it is not already serving, and the swarm model is
unloaded before stage 2 training.

  python3 scripts/r1_run.py --arm B4 --seed 3 --out /workspace/r1_b4_rep3

Behaviour:
  * writes r1_params.json into --out (the notebook's params-cell job);
    refuses to reuse an --out whose params name a different arm/seed;
  * resumes by default: if --out already holds completed cycles, the
    driver's resume restores the clone state (see r1_driver.py) and the
    run continues at the next cycle instead of restarting;
  * after every cycle: driver snapshot + a checkpoint zip in --out with
    the notebook layout (run/ + repo/), so the run can be copied off the
    box or resumed from the zip alone;
  * per-stage logs land in --out as stage{N}_cycle_{CCCC}.log.

Notes:
  * --no-resume only skips the explicit resume step. The driver's
    init() guard independently refuses to reset a clone when --out
    holds completed cycles, and will restore from the snapshot in
    that case — there is no supported way to silently restart a
    trajectory over an --out that already has completed cycles.
  * If this runner started the Ollama server, it terminates it on
    exit; a server that was already running is left running.
  * OLLAMA_MODELS and HF_HOME default to directories under
    /workspace (when it exists) so model weights survive a pod
    restart instead of living on the ephemeral container disk.

The driver itself must run under the ML venv interpreter; pass
--venv-python (default: the interpreter running this script).
"""
from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "scripts" / "r1_driver.py"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["B4", "B5"], required=True)
    ap.add_argument("--seed", type=int, required=True,
                    help="rng seed / replicate number (rep N uses seed N)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output directory (the OURO_R1_OUT equivalent)")
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--agents", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--train-steps", type=int, default=48)
    ap.add_argument("--train-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--swarm-model", default="qwen2.5:1.5b")
    ap.add_argument("--venv-python", default=sys.executable,
                    help="interpreter for the driver stages (ML venv)")
    ap.add_argument("--ollama-bin", default="ollama")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction,
                    default=True, help="resume from completed cycles in "
                    "--out if present (default: on)")
    ap.add_argument("--dry-run", action="store_true",
                    help="write params, report what resume would do, exit")
    return ap.parse_args()


_OLLAMA_PROC: subprocess.Popen | None = None
_LOCK_FD = None  # held for the whole run; see main()


def _stop_spawned_ollama() -> None:
    global _OLLAMA_PROC
    if _OLLAMA_PROC is not None and _OLLAMA_PROC.poll() is None:
        _OLLAMA_PROC.terminate()
        try:
            _OLLAMA_PROC.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _OLLAMA_PROC.kill()
    _OLLAMA_PROC = None


atexit.register(_stop_spawned_ollama)


def _probe_host(raw: str) -> str:
    """Normalise OLLAMA_HOST for HTTP probing: strip any scheme and
    map a bind-all address to loopback (OLLAMA_HOST is often set as
    http://0.0.0.0:11434, which is a bind address, not a probe one)."""
    h = raw.split("://", 1)[-1].rstrip("/")
    host_part, sep, port = h.partition(":")
    if host_part in ("0.0.0.0", "::", ""):
        host_part = "127.0.0.1"
    return f"{host_part}:{port}" if sep else host_part


def ollama_up(host: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}/api/tags", timeout=3):
            return True
    except Exception:
        return False


def ensure_ollama(args: argparse.Namespace, out: Path, host: str) -> None:
    global _OLLAMA_PROC
    if ollama_up(host):
        print(f"[r1_run] Ollama already serving on {host}")
    else:
        log_path = out / "ollama_serve.log"
        print(f"[r1_run] starting `ollama serve` (log: {log_path})")
        with open(log_path, "ab") as log:
            _OLLAMA_PROC = subprocess.Popen(
                [args.ollama_bin, "serve"], stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(60):
            if ollama_up(host):
                break
            if _OLLAMA_PROC.poll() is not None:
                raise SystemExit(
                    f"`ollama serve` exited early "
                    f"(rc={_OLLAMA_PROC.returncode}) — see {log_path}")
            time.sleep(2)
        else:
            raise SystemExit(f"Ollama did not come up — see {log_path}")
    listed = subprocess.run([args.ollama_bin, "list"], capture_output=True,
                            text=True, errors="replace")
    # Exact match on the NAME column: a substring check would accept
    # e.g. qwen2.5:1.5b-instruct-q8_0 as satisfying qwen2.5:1.5b.
    names = [line.split()[0] for line in listed.stdout.splitlines()[1:]
             if line.strip()]
    if args.swarm_model not in names:
        print(f"[r1_run] pulling swarm model {args.swarm_model}")
        subprocess.run([args.ollama_bin, "pull", args.swarm_model],
                       check=True)
    else:
        print(f"[r1_run] swarm model {args.swarm_model} present")


def unload_swarm_model(args: argparse.Namespace) -> None:
    """Free the GPU before stage 2 training, and VERIFY it is free.

    If this runner spawned the Ollama server, terminate it outright
    (ensure_ollama restarts it after stage 2). Otherwise `ollama stop`
    the swarm model and poll `ollama ps` until nothing is loaded —
    an unchecked stop that silently failed would leave the model in
    VRAM and stage 2 would OOM or crawl.
    """
    if _OLLAMA_PROC is not None and _OLLAMA_PROC.poll() is None:
        print("[r1_run] terminating runner-spawned Ollama server "
              "before stage 2")
        _stop_spawned_ollama()
        return
    subprocess.run([args.ollama_bin, "stop", args.swarm_model],
                   capture_output=True, text=True, errors="replace")
    for _ in range(30):
        ps = subprocess.run([args.ollama_bin, "ps"], capture_output=True,
                            text=True, errors="replace")
        loaded = [line for line in ps.stdout.splitlines()[1:]
                  if line.strip()]
        if not loaded:
            print("[r1_run] Ollama models unloaded")
            return
        time.sleep(2)
    raise SystemExit("[r1_run] Ollama models still loaded 60s after "
                     "`ollama stop` — refusing to start stage 2")


def run_driver(args: argparse.Namespace, out: Path, cmd: str,
               log_name: str) -> None:
    log_path = out / log_name
    print(f"[r1_run] driver {cmd} -> {log_path}")
    # Append mode: a retried cycle must not overwrite the failed
    # attempt's log — on an unattended pod that log IS the diagnostic.
    # errors="replace": one non-UTF-8 byte in stage output (CUDA /
    # library warnings do this) must not crash the runner and orphan
    # the child on a pipe nobody reads.
    with open(log_path, "a") as log:
        p = subprocess.Popen([args.venv_python, str(DRIVER), cmd],
                             cwd=REPO, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             errors="replace")
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            log.write(line)
        p.wait()
    if p.returncode != 0:
        raise SystemExit(f"driver {cmd} failed (rc={p.returncode}) — "
                         f"see {log_path}")


def completed_cycles(out: Path) -> set[int]:
    p = out / "r1_summary.json"
    if not p.exists():
        return set()
    summary = json.loads(p.read_text())
    return {int(r["cycle"]) for r in summary.get("cycles", [])
            if r.get("complete")}


def clone_cycle_count() -> int:
    p = REPO / "data/ouro_state.json"
    if p.exists():
        try:
            return int(json.loads(p.read_text()).get("cycle_count", 0))
        except (json.JSONDecodeError, ValueError, TypeError):
            return 0
    return 0


def write_checkpoint_zip(out: Path, cycle: int) -> Path:
    """Zip OUT into the notebook checkpoint layout: run/ + repo/.

    Written to a .tmp file and os.replace()d into place: a crash
    mid-write must not leave a truncated zip that a later resume
    would pick up and choke on.
    """
    zpath = out / f"r1_checkpoint_cycle_{cycle:02d}.zip"
    tmp = out / f"{zpath.name}.tmp"
    state = out / "state"
    run_files = [p for p in sorted(out.iterdir())
                 if p.is_file() and p != zpath and p != tmp
                 and not p.name.endswith(".zip")
                 and not p.name.endswith(".tmp")]
    state_files = [p for p in sorted(state.rglob("*")) if p.is_file()] \
        if state.exists() else []
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in run_files:
            z.write(p, f"run/{p.name}")
        for p in state_files:
            z.write(p, f"repo/{p.relative_to(state)}")
    os.replace(tmp, zpath)
    print(f"[r1_run] checkpoint zip: {zpath}")
    return zpath


# Keys that define the trajectory itself. Reusing an --out whose
# params differ in ANY of these would silently mix two experiment
# configurations in one summary — refuse. Only venv_python (a property
# of the box, not the experiment) may differ.
_TRAJECTORY_KEYS = ("arm", "rng_seed", "num_agents", "rounds",
                    "train_steps", "train_model", "swarm_model")


def main() -> None:
    global _LOCK_FD
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["OURO_R1_OUT"] = str(out)
    os.environ.setdefault("OLLAMA_HOST", "127.0.0.1:11434")
    os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
    if Path("/workspace").exists():
        # Keep model weights on the persistent volume, not the pod's
        # ephemeral container disk (see module docstring).
        os.environ.setdefault("OLLAMA_MODELS", "/workspace/.ollama/models")
        os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    host = _probe_host(os.environ["OLLAMA_HOST"])

    params = {
        "arm": args.arm, "rng_seed": args.seed,
        "num_agents": args.agents, "rounds": args.rounds,
        "train_steps": args.train_steps, "train_model": args.train_model,
        "swarm_model": args.swarm_model, "venv_python": args.venv_python,
    }
    params_path = out / "r1_params.json"
    if params_path.exists():
        existing = json.loads(params_path.read_text())
        diffs = {k: (existing.get(k), params[k]) for k in _TRAJECTORY_KEYS
                 if existing.get(k) != params[k]}
        if diffs:
            raise SystemExit(
                f"{params_path} does not match this invocation — "
                + "; ".join(f"{k}: existing={old!r} vs requested={new!r}"
                            for k, (old, new) in diffs.items())
                + " — refusing to mix trajectories in one output "
                  "directory")
        print(f"[r1_run] reusing existing {params_path}")
    elif args.dry_run:
        # A dry run must have no side effects: writing params here
        # would poison a fresh --out (a mistyped arm would then block
        # the corrected real run). With no params there are no
        # completed cycles either, so there is nothing to resume.
        print("[r1_run] dry-run: no r1_params.json in --out — fresh "
              "output directory, nothing to resume")
        raise SystemExit(0)
    else:
        params_path.write_text(json.dumps(params, indent=2))
        print(f"[r1_run] wrote {params_path}: {params}")

    if args.dry_run:
        print("[r1_run] dry-run: resume detection only")
        p = subprocess.run([args.venv_python, str(DRIVER), "resume",
                            "--dry-run"], cwd=REPO)
        raise SystemExit(p.returncode)

    # One runner per clone: two runners sharing a checkout would
    # interleave stages on the same data/ state. The lock is held for
    # the whole run (the fd stays open in _LOCK_FD until exit).
    lock_path = REPO / "data" / ".r1.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _LOCK_FD = open(lock_path, "a+")
    try:
        fcntl.flock(_LOCK_FD, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(
            f"another r1_run holds {lock_path} — refusing to share a "
            "clone between two runners")

    ensure_ollama(args, out, host)
    # Resume BEFORE init: init's guard would resume anyway, but doing
    # it explicitly first keeps init a pure no-op confirmation.
    if args.resume:
        run_driver(args, out, "resume", "resume.log")
    run_driver(args, out, "init", "init.log")

    done = completed_cycles(out)
    for cycle in range(args.cycles):
        if cycle in done:
            print(f"[r1_run] cycle {cycle} already complete — skipping")
            continue
        cc = clone_cycle_count()
        if cc != cycle:
            raise SystemExit(
                f"[r1_run] clone cycle_count={cc} but the next cycle "
                f"to run is {cycle} — the clone and the summary "
                "disagree; aborting before stage 1")
        run_driver(args, out, "stage1", f"stage1_cycle_{cycle:04d}.log")
        # Unload the swarm model so training gets the GPU (notebook
        # cell: `ollama stop <model>` before stage 2) — and verify.
        unload_swarm_model(args)
        run_driver(args, out, "stage2", f"stage2_cycle_{cycle:04d}.log")
        ensure_ollama(args, out, host)
        run_driver(args, out, "stage3", f"stage3_cycle_{cycle:04d}.log")
        done = completed_cycles(out)
        if cycle not in done:
            raise SystemExit(
                f"[r1_run] stage 3 exited 0 but cycle {cycle} is not "
                "marked complete in r1_summary.json — aborting before "
                "snapshot so no mid-cycle state is snapshotted")
        run_driver(args, out, "snapshot", f"snapshot_cycle_{cycle:04d}.log")
        write_checkpoint_zip(out, cycle)
    print(f"[r1_run] DONE — completed cycles: {sorted(done)}; "
          f"summary at {out / 'r1_summary.json'}")


if __name__ == "__main__":
    main()
