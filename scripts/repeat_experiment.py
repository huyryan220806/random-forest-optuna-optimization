"""Repeat the frozen 30-trial protocol for three prespecified seeds."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)


def run_stage(arguments, log):
    command = ["-m", "src.experiment", *arguments]
    print(f"\n{datetime.now(timezone.utc).isoformat()} | python {' '.join(command)}",
          file=log, flush=True)
    result = subprocess.run([sys.executable, "-u", *command], cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT)
    print(f"{datetime.now(timezone.utc).isoformat()} | exit={result.returncode}",
          file=log, flush=True)
    result.check_returncode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="New directory for this batch")
    output = parser.parse_args().directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for seed in SEEDS:
        folder = output / f"seed-{seed}"
        folder.mkdir()
        # Prefer relative paths in the public console transcript.
        location = str(folder.relative_to(ROOT)) if folder.is_relative_to(ROOT) else str(folder)
        print(f"Running seed {seed} in {location}", flush=True)
        with (folder / "console.log").open("x", encoding="utf-8") as log:
            run_stage(["--trials", "30", "--seed", str(seed), "--output-dir", location], log)
            run_stage(["--evaluate-only", "--input-dir", location], log)


if __name__ == "__main__":
    main()
