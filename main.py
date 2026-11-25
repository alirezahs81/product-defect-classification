"""Train the CNN and print test metrics."""

from __future__ import annotations

import json

from src.train import train


def main() -> None:
    metrics = train()
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
