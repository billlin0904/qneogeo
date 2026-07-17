from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kof98_env import (
    ActionMaskLevel,
    COMBO_SCENARIOS,
    Kof98Env,
    KofEnvClient,
    TrainingProfile,
)


@dataclass(frozen=True)
class ActionCase:
    actions: tuple[int, ...]
    combo_frames: tuple[int, ...]


CASES = {
    "crouch_b_crouch_a_mushiki": ActionCase(
        (11, 10, 19),
        (8, 25, 72, 87, 96, 105, 114),
    ),
    "corner_dokugami": ActionCase((8, 22, 23, 24, 25), (7, 30, 48, 77, 102, 131)),
    "forward_b_kototsuki": ActionCase((8, 22, 15), (7, 30, 48, 78, 98)),
    "forward_b_orochinagi": ActionCase((8, 22, 18), (7, 30, 48, 106)),
    "forward_b_red_kick": ActionCase((8, 22, 17), (7, 30, 48, 78)),
    "forward_b_aragami": ActionCase((8, 22, 14), (7, 30, 48, 71)),
    "close_c_75_kai_orochinagi": ActionCase((8, 26, 18), (7, 36, 57, 140)),
    "close_c_75_kai_red_kick": ActionCase((8, 26, 17), (7, 36, 57, 112)),
    "close_c_75_kai_kototsuki": ActionCase((8, 26, 15), (7, 36, 57, 113, 133)),
    "close_c_75_kai_aragami": ActionCase((8, 26, 14), (7, 36, 57, 111)),
    "corner_75_kai_aragami_chain": ActionCase(
        (8, 26, 14, 27, 28),
        (7, 36, 57, 111, 152, 190),
    ),
    "75_kai_orochinagi": ActionCase((26, 18), (39, 61, 142)),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Verify KOF98 buffered actions frame by frame.")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--case", action="append", choices=tuple(CASES), default=None)
    parser.add_argument("--frames", type=int, default=260)
    parser.add_argument("--skip-scenarios", action="store_true")
    return parser.parse_args()


def run_case(client: KofEnvClient, state_path: Path, case: ActionCase, frame_limit: int):
    client.load_state(state_path)
    action_index = 0
    accepted_actions: list[int] = []
    combo_frames: list[int] = []
    previous = client.observation()

    for frame in range(frame_limit):
        action_id = case.actions[action_index] if action_index < len(case.actions) else 0
        current = client.step(action_id, 1)
        status = client.action_status()
        if action_index < len(case.actions) and status.action_accepted:
            accepted_actions.append(action_id)
            action_index += 1

        if current.p1_combo_count > previous.p1_combo_count:
            combo_frames.append(frame)
        previous = current

    return tuple(accepted_actions), tuple(combo_frames)


def verify_combo_scenarios(root: Path, state_path: Path) -> bool:
    failed = False
    for name in COMBO_SCENARIOS:
        env = Kof98Env(
            root / "build-vs2026-x64" / "Release" / "fbneo_training.dll",
            root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll",
            root / "roms" / "fbneo" / "kof98.zip",
            root / "roms" / "fbneo",
            root / "saves",
            combo_state_path=state_path,
            training_profile=TrainingProfile.COMBO,
            combo_scenario=name,
            action_mask_level=ActionMaskLevel.STRICT,
        )
        try:
            env.reset()
            info = {}
            for _frame in range(480):
                mask = env.action_masks()
                actions = np.flatnonzero(mask & (np.arange(mask.size) != 0))
                action_id = int(actions[0]) if actions.size else 0
                _observation, _reward, terminated, truncated, info = env.step(action_id)
                if terminated or truncated:
                    break

            passed = bool(info.get("combo_success", 0.0))
            status = "PASS" if passed else "FAIL"
            print(
                f"[{status}] scenario {name}: frame={_frame} "
                f"max_combo={int(info.get('episode_max_combo', 0.0))}"
            )
            failed = failed or not passed
        finally:
            env.close()

    return not failed


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    state_path = args.state or root / "saves" / "states" / "kof98.slot1.state"
    if not state_path.is_absolute():
        state_path = root / state_path

    client = KofEnvClient(root / "build-vs2026-x64" / "Release" / "fbneo_training.dll")
    failed = False
    try:
        client.load_core(root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll")
        client.load_game(
            root / "roms" / "fbneo" / "kof98.zip",
            root / "roms" / "fbneo",
            root / "saves",
        )
        for name in args.case or CASES:
            case = CASES[name]
            accepted, combo_frames = run_case(client, state_path, case, args.frames)
            passed = accepted == case.actions and combo_frames[: len(case.combo_frames)] == case.combo_frames
            status = "PASS" if passed else "FAIL"
            print(
                f"[{status}] {name}: actions={accepted} "
                f"combo_frames={combo_frames} max_combo={len(combo_frames)}"
            )
            failed = failed or not passed
    finally:
        client.close()

    if not args.skip_scenarios:
        failed = not verify_combo_scenarios(root, state_path) or failed

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
