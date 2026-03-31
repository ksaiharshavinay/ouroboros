#!/usr/bin/env python3
# OUROBOROS — scripts/1_run_swarm.py
# Run the swarm simulation for one cycle.
#
# Usage:
#   python scripts/1_run_swarm.py [--cycle N]

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ouro.config import OuroConfig
from ouro.pipeline import OuroPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OUROBOROS swarm simulation")
    parser.add_argument("--cycle", type=int, default=0, help="Feedback cycle number")
    parser.add_argument("--agents", type=int, default=None, help="Override num_agents")
    parser.add_argument("--rounds", type=int, default=None, help="Override simulation_rounds")
    args = parser.parse_args()

    config = OuroConfig()
    if args.agents:
        config.num_agents = args.agents
    if args.rounds:
        config.simulation_rounds = args.rounds

    pipeline = OuroPipeline(config)
    result = pipeline.run_swarm()

    print("\n=== Swarm Complete ===")
    print(f"  Agents:     {result['num_agents']}")
    print(f"  Turns:      {result['num_turns']}")
    print(f"  Corpus:     {result['corpus_path']}")
    print(f"  Cycle:      {result['cycle']}")
    if result.get("report_preview"):
        print(f"\n  Report preview:\n  {result['report_preview'][:200]}...")


if __name__ == "__main__":
    main()
