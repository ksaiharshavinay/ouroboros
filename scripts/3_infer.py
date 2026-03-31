#!/usr/bin/env python3
# OUROBOROS — scripts/3_infer.py
# Run inference and feed the prediction back into the swarm.
#
# Usage:
#   python scripts/3_infer.py --prompt "What trends will emerge next cycle?"

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ouro.config import OuroConfig
from ouro.pipeline import OuroPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OUROBOROS inference + feedback")
    parser.add_argument(
        "--prompt",
        default="Summarise the key predictions and trends from this simulation cycle.",
        help="Inference prompt",
    )
    parser.add_argument(
        "--from-disk",
        action="store_true",
        help="Run feedback only, loading sim_result from last persisted cycle",
    )
    args = parser.parse_args()

    config = OuroConfig()
    pipeline = OuroPipeline(config)

    if args.from_disk:
        seed_path = pipeline.run_feedback_from_disk(prompt=args.prompt)
        print(f"\n=== Feedback Seed Written ===\n  {seed_path}")
    else:
        if not pipeline.infer.is_ready():
            print("No trained model found. Run scripts/2_run_research.py first.")
            sys.exit(1)
        prediction = pipeline.infer.predict(args.prompt)
        print("\n=== Prediction ===")
        print(prediction)


if __name__ == "__main__":
    main()
