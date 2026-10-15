"""Generate the NABDH benchmark dataset into data/.

    python scripts/generate_dataset.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nabdh import dataset  # noqa: E402


def main():
    t = time.time()
    episodes, gt, split = dataset.build()
    dataset.save(episodes, gt, split, ROOT / "data")
    print(f"{len(episodes)} episodes, {len(gt)} ground-truth rows -> data/ "
          f"({time.time() - t:.1f}s)")
    print(gt.groupby("kind").size().to_string())


if __name__ == "__main__":
    main()
