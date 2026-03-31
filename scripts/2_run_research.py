#!/usr/bin/env python3
# OUROBOROS — scripts/2_run_research.py
# Run the AutoResearch training loop.
#
# Usage:
#   python scripts/2_run_research.py

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ouro.config import OuroConfig
from ouro.pipeline import OuroPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    config = OuroConfig()
    pipeline = OuroPipeline(config)

    print("=== AutoResearch Training Loop ===")
    result = pipeline.run_autoresearch()

    print("\n=== Training Complete ===")
    print(f"  Return code: {result['returncode']}")
    if result.get("stdout"):
        print(f"  Output:\n{result['stdout'][-500:]}")


if __name__ == "__main__":
    main()
