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
    parser.add_argument("--combo-model", required=True, type=Path)
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
    if not args.combo_model.is_file():
        emit(
            {
                "type": "error",
                "message": f"Combo model not found: {args.combo_model}",
            }
        )
        return 2

    try:
        model = MaskablePPO.load(str(args.model), device=args.device)
        combo_model = MaskablePPO.load(
            str(args.combo_model),
            device=args.device,
        )
    except Exception as error:
        emit({"type": "error", "message": f"Could not load model: {error}"})
        return 3

    observation_size = int(model.observation_space.shape[0])
    action_count = int(model.action_space.n)
    if (
        combo_model.observation_space.shape != model.observation_space.shape
        or combo_model.action_space.n != action_count
    ):
        emit(
            {
                "type": "error",
                "message": "Fight and combo model spaces do not match",
            }
        )
        return 4

    combo_mode = False
    close_opportunities = 0
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

            input_ready = observation[23] >= 0.5
            distance = abs(float(observation[19])) * 320.0
            combo_finished = combo_mode and input_ready
            if combo_finished:
                combo_mode = False

            if (
                not combo_mode
                and not combo_finished
                and input_ready
                and distance <= 45.0
            ):
                close_opportunities += 1
                if close_opportunities % 2 == 1:
                    combo_mode = True

            active_model = combo_model if combo_mode else model
            action, _ = active_model.predict(
                observation,
                action_masks=action_mask,
                deterministic=True,
            )
            action = int(action)

            # The fight policy converged almost entirely to crouch B/A against
            # the old oniyaki-only opponent. Break that local optimum during
            # deployment so a distant human cannot make it crouch forever.
            if (
                not combo_mode
                and input_ready
                and distance > 45.0
                and action in (10, 11)
                and action_mask[1]
            ):
                action = 1

            emit(
                {
                    "type": "action",
                    "id": request_id,
                    "action": action,
                    "controller": "combo" if combo_mode else "fight",
                }
            )
        except Exception as error:
            emit({"type": "error", "message": str(error)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
