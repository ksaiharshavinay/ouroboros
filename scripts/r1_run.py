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

The driver itself must run under the ML venv interpreter; pass
--venv-python (default: the interpreter running this script).
"""
from __future__ import annotations

import argparse
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


def ollama_up(host: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}/api/tags", timeout=3):
            return True
    except Exception:
        return False


def ensure_ollama(args: argparse.Namespace, out: Path, host: str) -> None:
    if ollama_up(host):
        print(f"[r1_run] Ollama already serving on {host}")
    else:
        log_path = out / "ollama_serve.log"
        print(f"[r1_run] starting `ollama serve` (log: {log_path})")
        with open(log_path, "ab") as log:
            subprocess.Popen([args.ollama_bin, "serve"], stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(60):
            if ollama_up(host):
                break
            time.sleep(2)
        else:
            raise SystemExit(f"Ollama did not come up — see {log_path}")
    listed = subprocess.run([args.ollama_bin, "list"], capture_output=True,
                            text=True)
    if args.swarm_model not in listed.stdout:
        print(f"[r1_run] pulling swarm model {args.swarm_model}")
        subprocess.run([args.ollama_bin, "pull", args.swarm_model],
                       check=True)
    else:
        print(f"[r1_run] swarm model {args.swarm_model} present")


def run_driver(args: argparse.Namespace, out: Path, cmd: str,
               log_name: str) -> None:
    log_path = out / log_name
    print(f"[r1_run] driver {cmd} -> {log_path}")
    with open(log_path, "w") as log:
        p = subprocess.Popen([args.venv_python, str(DRIVER), cmd],
                             cwd=REPO, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
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


def write_checkpoint_zip(out: Path, cycle: int) -> Path:
    """Zip OUT into the notebook checkpoint layout: run/ + repo/."""
    zpath = out / f"r1_checkpoint_cycle_{cycle:02d}.zip"
    state = out / "state"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and p != zpath and not p.name.endswith(".zip"):
                z.write(p, f"run/{p.name}")
        if state.exists():
            for p in sorted(state.rglob("*")):
                if p.is_file():
                    z.write(p, f"repo/{p.relative_to(state)}")
    print(f"[r1_run] checkpoint zip: {zpath}")
    return zpath


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["OURO_R1_OUT"] = str(out)
    os.environ.setdefault("OLLAMA_HOST", "127.0.0.1:11434")
    os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
    host = os.environ["OLLAMA_HOST"]

    params = {
        "arm": args.arm, "rng_seed": args.seed,
        "num_agents": args.agents, "rounds": args.rounds,
        "train_steps": args.train_steps, "train_model": args.train_model,
        "swarm_model": args.swarm_model, "venv_python": args.venv_python,
    }
    params_path = out / "r1_params.json"
    if params_path.exists():
        existing = json.loads(params_path.read_text())
        if existing.get("arm") != args.arm or \
                int(existing.get("rng_seed", -1)) != args.seed:
            raise SystemExit(
                f"{params_path} is for arm={existing.get('arm')} "
                f"seed={existing.get('rng_seed')}, not arm={args.arm} "
                f"seed={args.seed} — refusing to mix trajectories in one "
                "output directory")
        print(f"[r1_run] reusing existing {params_path}")
    else:
        params_path.write_text(json.dumps(params, indent=2))
        print(f"[r1_run] wrote {params_path}: {params}")

    if args.dry_run:
        print("[r1_run] dry-run: resume detection only")
        p = subprocess.run([args.venv_python, str(DRIVER), "resume",
                            "--dry-run"], cwd=REPO)
        raise SystemExit(p.returncode)

    ensure_ollama(args, out, host)
    run_driver(args, out, "init", "init.log")
    if args.resume:
        run_driver(args, out, "resume", "resume.log")

    done = completed_cycles(out)
    for cycle in range(args.cycles):
        if cycle in done:
            print(f"[r1_run] cycle {cycle} already complete — skipping")
            continue
        run_driver(args, out, "stage1", f"stage1_cycle_{cycle:04d}.log")
        # Unload the swarm model so training gets the GPU (notebook cell:
        # `ollama stop <model>` before stage 2).
        subprocess.run([args.ollama_bin, "stop", args.swarm_model],
                       capture_output=True)
        run_driver(args, out, "stage2", f"stage2_cycle_{cycle:04d}.log")
        ensure_ollama(args, out, host)
        run_driver(args, out, "stage3", f"stage3_cycle_{cycle:04d}.log")
        run_driver(args, out, "snapshot", f"snapshot_cycle_{cycle:04d}.log")
        write_checkpoint_zip(out, cycle)
        done = completed_cycles(out)
    print(f"[r1_run] DONE — completed cycles: {sorted(done)}; "
          f"summary at {out / 'r1_summary.json'}")


if __name__ == "__main__":
    main()
