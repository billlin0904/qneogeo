"""PPO 推論橋接:讓 qneogeo 前端(C++)用訓練好的模型操作 P2。

協定:stdin/stdout 的 JSON Lines。
    C++ → Python:{"id": n, "observation": [26 or 140 floats], "mask": [29 bools]}
    Python → C++:{"type": "action", "id": n, "action": id,
                   "controller": "fight"|"combo"}
    啟動時送 {"type":"ready", ...},錯誤送 {"type":"error", ...}。

預設 pure-policy 模式(`--pure-policy`):
    - 只載入 Fight 模型。
    - 不切換 Combo 模型，也不覆寫 policy 選出的 Action。
    - 用於真人與 held-out 驗收，避免部署輔助污染成績。

可選雙模型切換策略(combo_mode 狀態機):
    - 平時用 fight 模型(實戰走位/防守)。
    - 進入近C 射程(距離 ≤45px)且可出招時,每兩次機會切一次
      combo 模型接管,直到連段動作結束(input_ready 恢復)。
    - 用意:fight 模型對舊版單一對手收斂成蹲B/蹲A 戳戳樂,
      combo 模型負責展示完整連段,兩者互補讓 P2 更像人。

觀測索引依賴(與 kof98_env._make_observation 對齊,改觀測必同步改這裡):
    observation[19] = distance_x / 320(還原 ×320)
    observation[23] = input_ready 旗標
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO


OBSERVATION_SCHEMA_V1_ID = "kof98-observation-v1-26"
OBSERVATION_SCHEMA_V2_ID = "kof98-observation-v2-140"
OBSERVATION_SCHEMA_V3_ID = "kof98-observation-v3-event-140"


def model_observation_schema(model) -> str:
    declared = getattr(model, "kof_observation_schema_id", None)
    observation_size = int(model.observation_space.shape[0])
    if declared:
        return str(declared)
    if observation_size == 26:
        return OBSERVATION_SCHEMA_V1_ID
    if observation_size == 140:
        # StrategyV2 checkpoints predate the explicit schema stamp.
        return OBSERVATION_SCHEMA_V2_ID
    raise ValueError(
        "Unsupported unstamped model observation size: "
        f"{observation_size}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="qneogeo PPO inference bridge")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--combo-model", type=Path)
    parser.add_argument(
        "--pure-policy",
        action="store_true",
        help="Use only --model; disable combo switching and action overrides.",
    )
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
    if not args.pure_policy and (
        args.combo_model is None or not args.combo_model.is_file()
    ):
        emit(
            {
                "type": "error",
                "message": f"Combo model not found: {args.combo_model}",
            }
        )
        return 2

    try:
        model = MaskablePPO.load(str(args.model), device=args.device)
        combo_model = (
            None
            if args.pure_policy
            else MaskablePPO.load(
                str(args.combo_model),
                device=args.device,
            )
        )
    except Exception as error:
        emit({"type": "error", "message": f"Could not load model: {error}"})
        return 3

    observation_size = int(model.observation_space.shape[0])
    observation_schema = model_observation_schema(model)
    combo_observation_size = (
        observation_size
        if combo_model is None
        else int(combo_model.observation_space.shape[0])
    )
    combo_observation_schema = (
        observation_schema
        if combo_model is None
        else model_observation_schema(combo_model)
    )
    action_count = int(model.action_space.n)
    if combo_model is not None and combo_model.action_space.n != action_count:
        emit(
            {
                "type": "error",
                "message": "Fight and combo model action spaces do not match",
            }
        )
        return 4
    if combo_observation_size > observation_size:
        emit(
            {
                "type": "error",
                "message": "Combo model observation cannot be wider than the fight model",
            }
        )
        return 4
    if (
        combo_model is not None
        and combo_observation_size == observation_size
        and combo_observation_schema != observation_schema
    ):
        emit(
            {
                "type": "error",
                "message": (
                    "Fight and combo models use different observation "
                    f"schemas: {observation_schema!r} vs "
                    f"{combo_observation_schema!r}"
                ),
            }
        )
        return 4

    combo_mode = False
    close_opportunities = 0
    emit(
        {
            "type": "ready",
            "observation_size": observation_size,
            "observation_schema_id": observation_schema,
            "action_count": action_count,
            "pure_policy": args.pure_policy,
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

            if args.pure_policy:
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
                        "controller": "fight",
                    }
                )
                continue

            # 索引 23 = input_ready 旗標、19 = distance_x/320(見模組說明)。
            input_ready = observation[23] >= 0.5
            distance = abs(float(observation[19])) * 320.0
            # combo 模型接管期間 input_ready 恢復 = 連段演出結束,交還 fight。
            combo_finished = combo_mode and input_ready
            if combo_finished:
                combo_mode = False

            # 近身機會(≤45px 且可出招)每兩次切一次 combo 模型 ——
            # 全切會太魯莽,全不切又只會戳,交替最像人。
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
            active_observation_size = (
                combo_observation_size if combo_mode else observation_size
            )
            action, _ = active_model.predict(
                observation[:active_observation_size],
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
