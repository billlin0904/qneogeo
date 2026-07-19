from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="qneogeo PPO inference bridge")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        emit({"type": "error", "message": f"Model not found: {args.model}"})
        return 2

    try:
        model = MaskablePPO.load(str(args.model), device=args.device)
    except Exception as error:
        emit({"type": "error", "message": f"Could not load model: {error}"})
        return 3

    observation_size = int(model.observation_space.shape[0])
    action_count = int(model.action_space.n)
    emit(
        {
            "type": "ready",
            "observation_size": observation_size,
            "action_count": action_count,
        }
    )

    for line in sys.stdin:
        try:
            request = json.loads(line)
            request_id = int(request["id"])
            observation = np.asarray(request["observation"], dtype=np.float32)
            action_mask = np.asarray(request["mask"], dtype=bool)
            if observation.shape != (observation_size,):
                raise ValueError(
                    f"Expected observation shape ({observation_size},), "
                    f"got {observation.shape}"
                )
            if action_mask.shape != (action_count,):
                raise ValueError(
                    f"Expected action mask shape ({action_count},), "
                    f"got {action_mask.shape}"
                )
            if not action_mask.any():
                action_mask[0] = True

            action, _ = model.predict(
                observation,
                action_masks=action_mask,
                deterministic=True,
            )
            emit(
                {
                    "type": "action",
                    "id": request_id,
                    "action": int(action),
                }
            )
        except Exception as error:
            emit({"type": "error", "message": str(error)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
