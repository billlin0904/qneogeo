from __future__ import annotations

import argparse
import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kof98_env import (
    ActionMaskLevel,
    COMBO_SCENARIOS,
    Kof98Env,
    KofEnvClient,
    STEP_EVENT_ACTION_STARTED,
    STEP_EVENT_COMBO_HIT,
    STEP_EVENTS_VERSION_1,
    StepEventsV1,
    TrainingProfile,
)


@dataclass(frozen=True)
class ActionCase:
    actions: tuple[int, ...]
    combo_frames: tuple[int, ...]


@dataclass(frozen=True)
class PhysicalRewardCase:
    scenario: str
    state_file: str
    actions: tuple[int, ...]
    expected_designated: bool = False
    expected_alternate: bool = False
    expected_actions_consumed: bool = True


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

PHYSICAL_REWARD_CASES = {
    "alternate_aragami": PhysicalRewardCase(
        "kyo_close_c_seventy_five_shiki_kai_orochinagi",
        "kof98.slot1.state",
        (8, 26, 14),
        expected_alternate=True,
    ),
    "alternate_kototsuki": PhysicalRewardCase(
        "kyo_close_c_seventy_five_shiki_kai_orochinagi",
        "kof98.slot1.state",
        (8, 26, 15),
        expected_alternate=True,
    ),
    "alternate_red_kick": PhysicalRewardCase(
        "kyo_close_c_seventy_five_shiki_kai_orochinagi",
        "kof98.slot1.state",
        (8, 26, 17),
        expected_alternate=True,
    ),
    "designated_orochinagi": PhysicalRewardCase(
        "kyo_close_c_seventy_five_shiki_kai_orochinagi",
        "kof98.slot1.state",
        (8, 26, 18),
        expected_designated=True,
    ),
    "designated_kototsuki_no_meter": PhysicalRewardCase(
        "kyo_close_c_seventy_five_shiki_kai_kototsuki",
        "kof98.slot3.state",
        (8, 26, 15),
        expected_designated=True,
    ),
    "alternate_orochinagi_with_meter": PhysicalRewardCase(
        "kyo_close_c_seventy_five_shiki_kai_red_kick",
        "kof98.slot1.state",
        (8, 26, 18),
        expected_alternate=True,
    ),
    "reject_orochinagi_without_meter": PhysicalRewardCase(
        "kyo_close_c_seventy_five_shiki_kai_kototsuki",
        "kof98.slot3.state",
        (8, 26, 18),
        expected_actions_consumed=False,
    ),
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


def verify_step_events(client: KofEnvClient, state_path: Path) -> bool:
    wrong_size = StepEventsV1()
    wrong_size.struct_size = ctypes.sizeof(StepEventsV1) - 1
    wrong_size.version = STEP_EVENTS_VERSION_1
    wrong_size_rejected = not bool(
        client.dll.kof_env_get_step_events_v1(
            client.handle,
            ctypes.byref(wrong_size),
        )
    )

    wrong_version = StepEventsV1()
    wrong_version.struct_size = ctypes.sizeof(StepEventsV1)
    wrong_version.version = STEP_EVENTS_VERSION_1 + 1
    wrong_version_rejected = not bool(
        client.dll.kof_env_get_step_events_v1(
            client.handle,
            ctypes.byref(wrong_version),
        )
    )

    client.load_state(state_path)
    actions = (11, 10, 19)
    action_index = 0
    started: list[tuple[int, int]] = []
    combo_hits: list[tuple[int, int]] = []

    for _frame in range(140):
        action_id = actions[action_index] if action_index < len(actions) else 0
        client.step(action_id, 1)
        status = client.action_status()
        if action_index < len(actions) and status.action_accepted:
            action_index += 1

        for event in client.step_events():
            if event.event_type == STEP_EVENT_ACTION_STARTED:
                started.append((int(event.action_id), int(event.action_serial)))
            elif event.event_type == STEP_EVENT_COMBO_HIT:
                combo_hits.append((int(event.action_id), int(event.action_serial)))

    started_ids = tuple(action_id for action_id, _serial in started)
    started_serials = tuple(serial for _action_id, serial in started)
    combo_hit_ids = tuple(action_id for action_id, _serial in combo_hits)
    serial_by_action = dict(started)
    serials_match = all(
        serial_by_action.get(action_id) == serial
        for action_id, serial in combo_hits
    )

    client.load_state(state_path)
    client.step(11, 1)
    repeated_starts = [
        (int(event.action_id), int(event.action_serial))
        for event in client.step_events()
        if event.event_type == STEP_EVENT_ACTION_STARTED
    ]
    repeated_serial_is_new = (
        len(repeated_starts) == 1
        and repeated_starts[0][0] == 11
        and repeated_starts[0][1] != serial_by_action.get(11)
    )
    passed = (
        started_ids == actions
        and len(set(started_serials)) == len(actions)
        and all(serial > 0 for serial in started_serials)
        and combo_hit_ids == (11, 10, 19, 19, 19, 19, 19)
        and serials_match
        and repeated_serial_is_new
        and wrong_size_rejected
        and wrong_version_rejected
    )
    status = "PASS" if passed else "FAIL"
    print(
        f"[{status}] step events: started={started} "
        f"combo_hits={combo_hits} repeated={repeated_starts} "
        f"abi_rejected={wrong_size_rejected and wrong_version_rejected}"
    )
    return passed


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


def verify_physical_reward_cases(root: Path) -> bool:
    failed = False
    for name, case in PHYSICAL_REWARD_CASES.items():
        env = Kof98Env(
            root / "build-vs2026-x64" / "Release" / "fbneo_training.dll",
            root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll",
            root / "roms" / "fbneo" / "kof98.zip",
            root / "roms" / "fbneo",
            root / "saves",
            combo_state_path=root / "saves" / "states" / case.state_file,
            training_profile=TrainingProfile.COMBO,
            combo_scenario=case.scenario,
            action_mask_level=ActionMaskLevel.PHYSICAL,
        )
        try:
            env.reset()
            pending_actions = list(case.actions)
            info = {}
            for frame in range(480):
                mask = env.action_masks()
                action_id = 0
                if pending_actions and mask[pending_actions[0]]:
                    action_id = pending_actions.pop(0)

                _observation, _reward, terminated, truncated, info = env.step(action_id)
                if terminated or truncated:
                    break

            designated = bool(info.get("combo_success", 0.0))
            alternate = bool(info.get("combo_alternate_success", 0.0))
            passed = (
                (not pending_actions) == case.expected_actions_consumed
                and designated == case.expected_designated
                and alternate == case.expected_alternate
            )
            status = "PASS" if passed else "FAIL"
            print(
                f"[{status}] physical {name}: frame={frame} "
                f"max_combo={int(info.get('episode_max_combo', 0.0))} "
                f"designated={int(designated)} alternate={int(alternate)}"
            )
            failed = failed or not passed
        finally:
            env.close()

    return not failed


def verify_fight_event_attribution(root: Path, state_path: Path) -> bool:
    env = Kof98Env(
        root / "build-vs2026-x64" / "Release" / "fbneo_training.dll",
        root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll",
        root / "roms" / "fbneo" / "kof98.zip",
        root / "roms" / "fbneo",
        root / "saves",
        state_path=state_path,
        action_repeat=1,
        hitbox_reward=False,
        p2_training_ai=False,
        training_profile=TrainingProfile.FIGHT,
        action_mask_level=ActionMaskLevel.PHYSICAL,
    )
    try:
        env.reset()
        pending_actions = [11, 10, 19]
        queued: list[int] = []
        started: list[int] = []
        followup_hits: list[int] = []
        combo_hit_actions: set[int] = set()
        cancel_reward = 0.0
        max_combo = 0

        for _frame in range(180):
            mask = env.action_masks()
            action_id = 0
            if pending_actions and mask[pending_actions[0]]:
                action_id = pending_actions.pop(0)

            _observation, _reward, terminated, truncated, info = env.step(action_id)
            queued.extend(int(value) for value in info["queued_followup_actions"])
            started.extend(int(value) for value in info["started_followup_actions"])
            followup_hits.extend(int(value) for value in info["hit_followup_actions"])
            combo_hit_actions.update(int(value) for value in info["step_combo_hit_action_ids"])
            cancel_reward += float(info["reward_cancel"])
            max_combo = max(max_combo, int(info["p1_combo_count"]))
            if terminated or truncated or (not pending_actions and max_combo >= 7):
                break

        passed = (
            not pending_actions
            and queued == [10, 19]
            and started == [10, 19]
            and followup_hits == [10, 19]
            and combo_hit_actions == {11, 10, 19}
            and cancel_reward == 0.0
            and max_combo >= 7
        )
        status = "PASS" if passed else "FAIL"
        print(
            f"[{status}] fight attribution: queued={queued} started={started} "
            f"followup_hits={followup_hits} hit_actions={sorted(combo_hit_actions)} "
            f"cancel_reward={cancel_reward:.1f} max_combo={max_combo}"
        )
        return passed
    finally:
        env.close()


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
        failed = not verify_step_events(client, state_path) or failed
    finally:
        client.close()

    if not args.skip_scenarios:
        failed = not verify_combo_scenarios(root, state_path) or failed
        failed = not verify_physical_reward_cases(root) or failed
        failed = not verify_fight_event_attribution(root, state_path) or failed

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
