"""KOF98 動作系統的確定性迴歸測試(改 C++ 腳本/觸發幀後必跑)。

三層驗證:
    CASES:            固定動作序列 → 斷言每一段命中的「精確幀號」。
                      幀號變了 = C++ 輸入腳本或時序被改動(可能是有意,
                      也可能是回歸)—— 這是本專案的黃金基準。
    scenario 驗證:    用 Kof98Env combo profile 走完整 scenario,
                      斷言 phase 推進與最終 combo 數。
    PHYSICAL_REWARD_CASES:greedy 時序(第一個合法幀就出招,physical
                      mask 的實際行為)下驗證 designated/alternate
                      收尾的獎勵判定 —— 守護 queue 盲區修正不回歸。

用法:
    python tools/verify_kof98_actions.py               # 全部
    python tools/verify_kof98_actions.py --case NAME   # 指定案例
"""
from __future__ import annotations

import argparse
import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kof98_env import (
    ACTION_COUNT,
    ACTION_SET_VERSION,
    ActionMaskLevel,
    COMBO_PROFILE,
    COMBO_SCENARIOS,
    FightCurriculum,
    FightRewardVersion,
    JoypadState,
    Kof98Env,
    KofEnvClient,
    P2Style,
    P1_HOLD_CHUNK_FRAMES,
    STEP_EVENT_ACTION_STARTED,
    STEP_EVENT_AUTO_GUARD,
    STEP_EVENT_BLOCK_CONTACT,
    STEP_EVENT_BLOCKSTUN_ENDED,
    STEP_EVENT_BLOCKSTUN_STARTED,
    STEP_EVENT_CHIP_DAMAGE,
    STEP_EVENT_CLEAN_HIT,
    STEP_EVENT_COMBO_HIT,
    STEP_EVENT_DAMAGE_ONLY,
    STEP_EVENT_MANUAL_BLOCK_SUCCESS,
    STEP_EVENT_P1_DAMAGE,
    STEP_EVENTS_VERSION_1,
    STEP_EVENTS_VERSION_2,
    STEP_EVENTS_VERSION_3,
    STEP_EVENTS_VERSION_4,
    STEP_EVENTS_VERSION_5,
    StepEventV5,
    StepEventsV1,
    StepEventsV2,
    StepEventsV3,
    StepEventsV4,
    StepEventsV5,
    TrainingProfile,
)
from kof98_observation import (
    OBSERVATION_V2_SIZE,
    OBSERVATION_V3_REPURPOSED_INDICES,
    OBSERVATION_V3_SIZE,
    ObservationVersion,
)
from kof98_curriculum import (
    CurriculumTask,
    LevelRecipe,
    OracleAction,
    RewardMachinePhase,
    TacticalRewardMachine,
    load_level_recipes,
    prepare_level,
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


def verify_runtime_contract_and_hold_chunks(
    client: KofEnvClient,
    state_path: Path,
) -> bool:
    contract_passed = (
        client.action_count == ACTION_COUNT
        and client.action_set_version == ACTION_SET_VERSION
        and client.p1_hold_chunk_frames == P1_HOLD_CHUNK_FRAMES
    )

    client.load_state(state_path)
    safe_after_load = client.snapshot_safe()
    client.step(1, P1_HOLD_CHUNK_FRAMES)
    first_hold = client.last_joypad()
    first_hold_passed = (
        client.input_ready()
        and client.snapshot_safe()
        and bool(first_hold.left) != bool(first_hold.right)
    )

    client.step(1, P1_HOLD_CHUNK_FRAMES)
    second_hold = client.last_joypad()
    repeated_hold_passed = (
        client.input_ready()
        and bool(second_hold.left) != bool(second_hold.right)
    )

    client.step(6, 1)
    attack_status = client.action_status()
    attack_interrupt_passed = (
        bool(attack_status.action_accepted)
        and attack_status.last_started_action_id == 6
    )

    client.load_state(state_path)
    internal_p1_rejected = False
    try:
        client.step(1000, 1)
    except RuntimeError:
        internal_p1_rejected = True

    client.load_state(state_path)
    client.set_p2_action_ai(True)
    internal_p2_rejected = False
    try:
        client.set_p2_action(1000)
    except RuntimeError:
        internal_p2_rejected = True

    client.set_p2_action(1)
    client.run_frames(P1_HOLD_CHUNK_FRAMES)
    p2_still_uses_shared_script = not client.p2_input_ready()
    client.run_frames(2)
    p2_script_completed = client.p2_input_ready()
    client.set_p2_action_ai(False)

    passed = all((
        contract_passed,
        safe_after_load,
        first_hold_passed,
        repeated_hold_passed,
        attack_interrupt_passed,
        internal_p1_rejected,
        internal_p2_rejected,
        p2_still_uses_shared_script,
        p2_script_completed,
    ))
    print(
        f"[{'PASS' if passed else 'FAIL'}] Runtime contract/P1 hold chunks: "
        f"contract={contract_passed} safe={safe_after_load} "
        f"hold={first_hold_passed}/{repeated_hold_passed} "
        f"attack={attack_interrupt_passed} "
        f"internal={internal_p1_rejected}/{internal_p2_rejected} "
        f"p2_legacy={p2_still_uses_shared_script}/{p2_script_completed}"
    )
    return passed


def verify_level_recipe_replay(
    client: KofEnvClient,
    root: Path,
) -> bool:
    safe_state = root / "saves" / "states" / "kof98.slot2.state"
    unsafe_state = root / "saves" / "states" / "kof98.slot4.state"
    recipe = LevelRecipe(
        name="verify_anti_air_replay",
        task=CurriculumTask.ANTI_AIR,
        base_state=safe_state,
        p2_action_id=20,
        p2_prelude_frames=8,
        oracle_actions=(OracleAction(16),),
        trigger_frame=8,
    )
    round_trip = LevelRecipe.from_dict(recipe.to_dict())
    first = prepare_level(client, recipe)
    first_strategy = client.strategy_state()
    second = prepare_level(client, round_trip)
    second_strategy = client.strategy_state()
    deterministic_fields = (
        "round_time",
        "p1_health",
        "p2_health",
        "p1_x",
        "p1_y",
        "p2_x",
        "p2_y",
        "distance_x",
        "distance_y",
    )
    deterministic_replay = all(
        getattr(first, field) == getattr(second, field)
        for field in deterministic_fields
    )
    p2_script_replayed = (
        first_strategy.p2_active_action_id == 20
        and second_strategy.p2_active_action_id == 20
        and first_strategy.p2_action_elapsed_frames
        == second_strategy.p2_action_elapsed_frames
        == 8
    )

    unsafe_rejected = True
    if unsafe_state.is_file():
        client.set_p2_action_ai(False)
        client.load_state(unsafe_state)
        unsafe_rejected = not client.snapshot_safe()
    client.set_p2_action_ai(False)

    passed = deterministic_replay and p2_script_replayed and unsafe_rejected
    print(
        f"[{'PASS' if passed else 'FAIL'}] Level recipe replay: "
        f"deterministic={deterministic_replay} "
        f"p2_script={p2_script_replayed} unsafe_rejected={unsafe_rejected}"
    )
    return passed


def verify_tactical_reward_machine() -> bool:
    machine = TacticalRewardMachine()
    waiting_failure = machine.advance(
        opportunity=False,
        committed=False,
        success=False,
        failure=True,
    )
    opportunity = machine.advance(
        opportunity=True,
        committed=False,
        success=False,
        failure=False,
    )
    committed = machine.advance(
        opportunity=False,
        committed=True,
        success=False,
        failure=False,
    )
    success = machine.advance(
        opportunity=False,
        committed=False,
        success=True,
        failure=False,
    )
    terminal_is_sticky = machine.advance(
        opportunity=False,
        committed=False,
        success=False,
        failure=True,
    )
    machine.reset()
    machine.advance(
        opportunity=True,
        committed=False,
        success=False,
        failure=False,
    )
    failure = machine.advance(
        opportunity=False,
        committed=False,
        success=False,
        failure=True,
    )
    passed = (
        waiting_failure.phase is RewardMachinePhase.WAITING
        and opportunity.phase is RewardMachinePhase.OPPORTUNITY
        and committed.phase is RewardMachinePhase.COMMITTED
        and success.success
        and success.phase is RewardMachinePhase.SUCCEEDED
        and terminal_is_sticky.phase is RewardMachinePhase.SUCCEEDED
        and failure.failure
        and failure.phase is RewardMachinePhase.FAILED
    )
    print(
        f"[{'PASS' if passed else 'FAIL'}] Tactical Reward Machine: "
        f"opportunity={opportunity.phase.name} "
        f"committed={committed.phase.name} success={success.success} "
        f"failure={failure.failure}"
    )
    return passed


def verify_causal_approach_metric(root: Path) -> bool:
    common = dict(
        dll_path=root / "build-vs2026-x64" / "Release" / "fbneo_training.dll",
        core_path=root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll",
        game_path=root / "roms" / "fbneo" / "kof98.zip",
        system_dir=root / "system",
        save_dir=root / "saves",
        fight_state_path=root / "saves" / "states" / "kof98.slot2.state",
        action_repeat=4,
        hitbox_reward=False,
        p2_training_ai=False,
        training_profile=TrainingProfile.FIGHT,
        action_mask_level=ActionMaskLevel.PHYSICAL,
        observation_version=ObservationVersion.V2,
        fight_reward_version=FightRewardVersion.SYMMETRIC_V2,
    )
    p1_env = Kof98Env(**common)
    p2_env = None
    try:
        p1_env.reset()
        p1_success = False
        p1_progress = 0.0
        for _ in range(8):
            _observation, _reward, terminated, truncated, info = p1_env.step(1)
            p1_success = p1_success or bool(info["tactical_approach_success"])
            p1_progress = max(
                p1_progress,
                float(info["tactical_approach_p1_progress"]),
            )
            if p1_success or terminated or truncated:
                break
        p1_env.close()
        p1_env = None

        p2_recipe = LevelRecipe(
            name="verify_p2_cannot_claim_approach",
            task=CurriculumTask.APPROACH,
            base_state=root / "saves" / "states" / "kof98.slot2.state",
            p2_action_id=20,
            p2_start_delay_frames=8,
        )
        p2_env = Kof98Env(**common, level_recipe=p2_recipe)
        p2_env.reset()
        recipe_uses_physical_mask = int(p2_env.action_masks().sum()) > 2
        p2_crossed_threshold = False
        p2_claimed_success = False
        p2_reported_progress = 0.0
        p2_started_sequence: list[bool] = []
        for _ in range(20):
            _observation, _reward, terminated, truncated, info = p2_env.step(0)
            p2_started_sequence.append(bool(info["level_p2_started"]))
            raw = info["raw"]
            p2_crossed_threshold = p2_crossed_threshold or (
                abs(raw.distance_x) <= 90
            )
            p2_claimed_success = p2_claimed_success or bool(
                info["tactical_approach_success"]
            )
            p2_reported_progress = max(
                p2_reported_progress,
                float(info["tactical_approach_p1_progress"]),
            )
            if terminated or truncated:
                break

        passed = (
            p1_success
            and p1_progress >= 12.0
            and p2_crossed_threshold
            and not p2_claimed_success
            and p2_reported_progress == 0.0
            and p2_started_sequence[:3] == [False, False, True]
            and recipe_uses_physical_mask
        )
        print(
            f"[{'PASS' if passed else 'FAIL'}] Causal approach metric: "
            f"p1={p1_success}/{p1_progress:.1f} "
            f"p2_crossed={p2_crossed_threshold} "
            f"p2_claimed={p2_claimed_success}/{p2_reported_progress:.1f} "
            f"delayed={p2_started_sequence[:3]} "
            f"physical_mask={recipe_uses_physical_mask}"
        )
        return passed
    finally:
        if p1_env is not None:
            p1_env.close()
        if p2_env is not None:
            p2_env.close()


def verify_tactical_recipe_oracles(root: Path) -> bool:
    recipe_path = root / "ai_logs" / "oracle" / "kof98_v3c_recipes.json"
    recipes = load_level_recipes(recipe_path)
    failed: list[str] = []
    for recipe in recipes:
        env = Kof98Env(
            root / "build-vs2026-x64" / "Release" / "fbneo_training.dll",
            root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll",
            root / "roms" / "fbneo" / "kof98.zip",
            root / "system",
            root / "saves",
            action_repeat=4,
            hitbox_reward=False,
            p2_training_ai=False,
            training_profile=TrainingProfile.FIGHT,
            action_mask_level=ActionMaskLevel.PHYSICAL,
            observation_version=ObservationVersion.V2,
            fight_reward_version=FightRewardVersion.SYMMETRIC_TACTICAL_V3,
            level_recipe=recipe,
        )
        try:
            env.reset()
            uses_physical_mask = int(env.action_masks().sum()) > 2
            info = {}
            frame_limit = max(240, recipe.settle_frames + 64)
            for _frame in range(0, frame_limit, env.action_repeat):
                expected_action = env._level_oracle_action()
                mask = env.action_masks()
                action_id = (
                    expected_action
                    if expected_action >= 0 and mask[expected_action]
                    else 0
                )
                _observation, _reward, terminated, truncated, info = env.step(
                    action_id
                )
                if terminated or truncated:
                    break

            passed = uses_physical_mask and bool(
                info.get("reward_machine_success", 0.0)
            )
            if not passed:
                failed.append(recipe.name)
        finally:
            env.close()

    passed = not failed
    print(
        f"[{'PASS' if passed else 'FAIL'}] Tactical recipe Oracles: "
        f"{len(recipes) - len(failed)}/{len(recipes)}"
        + (f" failed={failed}" if failed else "")
    )
    return passed


def verify_step_events(client: KofEnvClient, state_path: Path) -> bool:
    def current_epoch_pair() -> tuple[int, int]:
        batch = StepEventsV5()
        batch.struct_size = ctypes.sizeof(StepEventsV5)
        batch.version = STEP_EVENTS_VERSION_5
        client._check(
            client.dll.kof_env_get_step_events_v5(
                client.handle,
                ctypes.byref(batch),
            )
        )
        timing = client.combat_timing_state()
        return int(batch.batch_event_epoch), int(timing.event_epoch)

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
    wrong_v2_size = StepEventsV2()
    wrong_v2_size.struct_size = ctypes.sizeof(StepEventsV2) - 1
    wrong_v2_size.version = STEP_EVENTS_VERSION_2
    wrong_v2_size_rejected = not bool(
        client.dll.kof_env_get_step_events_v2(
            client.handle,
            ctypes.byref(wrong_v2_size),
        )
    )
    wrong_v2_version = StepEventsV2()
    wrong_v2_version.struct_size = ctypes.sizeof(StepEventsV2)
    wrong_v2_version.version = STEP_EVENTS_VERSION_2 + 1
    wrong_v2_version_rejected = not bool(
        client.dll.kof_env_get_step_events_v2(
            client.handle,
            ctypes.byref(wrong_v2_version),
        )
    )
    wrong_v3_size = StepEventsV3()
    wrong_v3_size.struct_size = ctypes.sizeof(StepEventsV3) - 1
    wrong_v3_size.version = STEP_EVENTS_VERSION_3
    wrong_v3_size_rejected = not bool(
        client.dll.kof_env_get_step_events_v3(
            client.handle,
            ctypes.byref(wrong_v3_size),
        )
    )
    wrong_v3_version = StepEventsV3()
    wrong_v3_version.struct_size = ctypes.sizeof(StepEventsV3)
    wrong_v3_version.version = STEP_EVENTS_VERSION_3 + 1
    wrong_v3_version_rejected = not bool(
        client.dll.kof_env_get_step_events_v3(
            client.handle,
            ctypes.byref(wrong_v3_version),
        )
    )
    wrong_v4_size = StepEventsV4()
    wrong_v4_size.struct_size = ctypes.sizeof(StepEventsV4) - 1
    wrong_v4_size.version = STEP_EVENTS_VERSION_4
    wrong_v4_size_rejected = not bool(
        client.dll.kof_env_get_step_events_v4(
            client.handle,
            ctypes.byref(wrong_v4_size),
        )
    )
    wrong_v4_version = StepEventsV4()
    wrong_v4_version.struct_size = ctypes.sizeof(StepEventsV4)
    wrong_v4_version.version = STEP_EVENTS_VERSION_4 + 1
    wrong_v4_version_rejected = not bool(
        client.dll.kof_env_get_step_events_v4(
            client.handle,
            ctypes.byref(wrong_v4_version),
        )
    )
    wrong_v5_size = StepEventsV5()
    wrong_v5_size.struct_size = ctypes.sizeof(StepEventsV5) - 1
    wrong_v5_size.version = STEP_EVENTS_VERSION_5
    wrong_v5_size_rejected = not bool(
        client.dll.kof_env_get_step_events_v5(
            client.handle,
            ctypes.byref(wrong_v5_size),
        )
    )
    wrong_v5_version = StepEventsV5()
    wrong_v5_version.struct_size = ctypes.sizeof(StepEventsV5)
    wrong_v5_version.version = STEP_EVENTS_VERSION_5 + 1
    wrong_v5_version_rejected = not bool(
        client.dll.kof_env_get_step_events_v5(
            client.handle,
            ctypes.byref(wrong_v5_version),
        )
    )

    client.load_state(state_path)
    load_batch_epoch, load_timing_epoch = current_epoch_pair()
    client.reset()
    reset_batch_epoch, reset_timing_epoch = current_epoch_pair()
    epoch_sync_passed = (
        load_batch_epoch == load_timing_epoch
        and reset_batch_epoch == reset_timing_epoch
        and reset_timing_epoch != load_timing_epoch
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
        and wrong_v2_size_rejected
        and wrong_v2_version_rejected
        and wrong_v3_size_rejected
        and wrong_v3_version_rejected
        and wrong_v4_size_rejected
        and wrong_v4_version_rejected
        and wrong_v5_size_rejected
        and wrong_v5_version_rejected
        and epoch_sync_passed
    )
    status = "PASS" if passed else "FAIL"
    print(
        f"[{status}] step events: started={started} "
        f"combo_hits={combo_hits} repeated={repeated_starts} "
        "abi_rejected="
        f"{wrong_size_rejected and wrong_version_rejected and wrong_v2_size_rejected and wrong_v2_version_rejected and wrong_v3_size_rejected and wrong_v3_version_rejected and wrong_v4_size_rejected and wrong_v4_version_rejected and wrong_v5_size_rejected and wrong_v5_version_rejected} "
        f"epoch_sync={epoch_sync_passed}"
    )
    return passed


def verify_step_event_counterexamples(
    client: KofEnvClient,
    state_path: Path,
) -> bool:
    """Verify negative cases that success-only combo tests cannot cover."""

    def direct_trace(
        p1_action: int,
        p2_action: int | None,
        frames: int,
        *,
        repeat_p1: bool = False,
        repeat_p2: bool = False,
    ):
        client.load_state(state_path)
        client.set_p2_training_ai(False)
        client.set_p2_action_ai(p2_action is not None)
        if p2_action is not None:
            client.set_p2_action(p2_action)

        before = client.observation()
        events_by_frame = []
        for frame in range(frames):
            if (
                repeat_p2
                and p2_action is not None
                and client.p2_input_ready()
            ):
                client.set_p2_action(p2_action)
            action_id = p1_action if repeat_p1 or frame == 0 else 0
            client.step(action_id, 1)
            events = client.step_events()
            if events:
                events_by_frame.append((frame, events))
        return before, client.observation(), events_by_frame

    def flat(events_by_frame):
        return [event for _frame, events in events_by_frame for event in events]

    def blockstun_boundary_passed(
        events_by_frame,
        action_id: int,
        expected_frames: int,
    ) -> bool:
        # C++ 只有在候選倒數真的出現 N→N-1 後才公開 START telemetry，
        # 因此 START 比實際倒數載入晚一幀。END 仍是精確的 0→-1 邊緣。
        starts = [
            frame
            for frame, events in events_by_frame
            for event in events
            if event.event_type == STEP_EVENT_BLOCKSTUN_STARTED
            and event.action_id == action_id
            and event.action_serial > 0
        ]
        ends = [
            frame
            for frame, events in events_by_frame
            for event in events
            if event.event_type == STEP_EVENT_BLOCKSTUN_ENDED
            and event.action_id == action_id
            and event.action_serial > 0
        ]
        return (
            len(starts) == 1
            and len(ends) == 1
            and ends[0] - starts[0] == max(0, expected_frames - 1)
        )

    # Grounded and airborne hits prove that anti-air labels use the state at
    # the start of the hit frame instead of a position changed by the hit.
    _before, _after, grounded_frames = direct_trace(16, None, 50)
    grounded_hits = [
        event
        for event in flat(grounded_frames)
        if event.event_type in (STEP_EVENT_COMBO_HIT, STEP_EVENT_DAMAGE_ONLY)
        and event.action_id == 16
    ]
    grounded_airborne_passed = bool(grounded_hits) and all(
        not event.target_airborne_at_event for event in grounded_hits
    )

    _before, _after, close_c_frames = direct_trace(8, None, 25)
    close_c_hits = [
        event
        for event in flat(close_c_frames)
        if event.event_type in (STEP_EVENT_COMBO_HIT, STEP_EVENT_DAMAGE_ONLY)
        and event.action_id == 8
    ]
    hit_stop_target_passed = bool(close_c_hits) and any(
        event.p1_hit_guard_stop_after == event.p1_hit_guard_stop_before
        and event.p2_hit_guard_stop_after > event.p2_hit_guard_stop_before
        for event in close_c_hits
    )

    _before, _after, airborne_frames = direct_trace(16, 20, 50)
    airborne_hits = [
        event
        for event in flat(airborne_frames)
        if event.event_type in (STEP_EVENT_COMBO_HIT, STEP_EVENT_DAMAGE_ONLY)
        and event.action_id == 16
    ]
    airborne_before_passed = bool(airborne_hits) and any(
        event.target_airborne_at_event for event in airborne_hits
    )

    # A normal attack must block without damage. A blocked special that deals
    # chip damage must report both block contact and chip, never a clean hit.
    block_before, block_after, block_frames = direct_trace(
        2, 8, 40, repeat_p1=True
    )
    block_events = flat(block_frames)
    normal_block_passed = (
        block_after.p1_health == block_before.p1_health
        and any(
            event.event_type == STEP_EVENT_BLOCK_CONTACT
            and event.action_id == 8
            and event.action_serial > 0
            for event in block_events
        )
        and any(
            event.event_type == STEP_EVENT_MANUAL_BLOCK_SUCCESS
            and event.action_id == 8
            for event in block_events
        )
        and not any(
            event.event_type == STEP_EVENT_AUTO_GUARD
            for event in block_events
        )
        and not any(
            event.event_type in (STEP_EVENT_CHIP_DAMAGE, STEP_EVENT_CLEAN_HIT)
            for event in block_events
        )
    )
    block_stop_defender_passed = any(
        event.event_type == STEP_EVENT_BLOCK_CONTACT
        and event.p1_hit_guard_stop_after > event.p1_hit_guard_stop_before
        and event.p2_hit_guard_stop_after == event.p2_hit_guard_stop_before
        and event.expected_blockstun_frames == 17
        and event.expected_blockstun_source == 1
        for event in block_events
    )
    heavy_blockstun_boundary_passed = blockstun_boundary_passed(
        block_frames,
        8,
        17,
    )

    chip_before, chip_after, chip_frames = direct_trace(
        2, 14, 45, repeat_p1=True
    )
    chip_events = flat(chip_frames)
    chip_damage = sum(
        int(event.p1_hp_delta)
        for event in chip_events
        if event.event_type == STEP_EVENT_CHIP_DAMAGE
    )
    chip_block_passed = (
        chip_damage == chip_before.p1_health - chip_after.p1_health
        and chip_damage > 0
        and any(
            event.event_type == STEP_EVENT_BLOCK_CONTACT
            and event.action_id == 14
            for event in chip_events
        )
        and any(
            event.event_type == STEP_EVENT_MANUAL_BLOCK_SUCCESS
            and event.action_id == 14
            for event in chip_events
        )
        and not any(
            event.event_type == STEP_EVENT_CLEAN_HIT
            for event in chip_events
        )
    )

    # V5 must expose a mutually exclusive result when P1's special is blocked.
    # The legacy P1 damage path writes DAMAGE_ONLY before the guard result is
    # known; V5 removes that provisional hit once BLOCK/CHIP is confirmed.
    client.load_state(state_path)
    client.set_p2_training_ai(False)
    client.set_p2_action_ai(False)
    p2_chip_events = []
    for frame in range(70):
        p2_guard = JoypadState()
        p2_guard.right = 1
        client.set_joypad_for_port(1, p2_guard)
        client.step(14 if frame == 0 else 0, 1)
        p2_chip_events.extend(client.step_events())
    p2_chip_contact_frames = {
        int(event.absolute_engine_frame)
        for event in p2_chip_events
        if event.source_player == 1
        and event.target_player == 2
        and event.event_type
        in (STEP_EVENT_BLOCK_CONTACT, STEP_EVENT_CHIP_DAMAGE)
    }
    v5_chip_exclusive_passed = (
        bool(p2_chip_contact_frames)
        and any(
            event.event_type == STEP_EVENT_CHIP_DAMAGE
            and event.source_player == 1
            and event.target_player == 2
            for event in p2_chip_events
        )
        and not any(
            event.absolute_engine_frame in p2_chip_contact_frames
            and event.source_player == 1
            and event.target_player == 2
            and event.event_type
            in (
                STEP_EVENT_COMBO_HIT,
                STEP_EVENT_DAMAGE_ONLY,
                STEP_EVENT_CLEAN_HIT,
            )
            for event in p2_chip_events
        )
    )

    # During hitstop the Guard reaction is known, but its remaining time is
    # still unknown. N->N-1 must be observed before remaining becomes valid.
    client.load_state(state_path)
    client.set_p2_training_ai(False)
    client.set_p2_action_ai(True)
    client.set_p2_action(8)
    reaction_unconfirmed_seen = False
    reaction_confirmed_seen = False
    for _frame in range(45):
        client.step(2, 1)
        timing = client.combat_timing_state().p1
        if timing.reaction_valid and not timing.reaction_remaining_valid:
            reaction_unconfirmed_seen = (
                reaction_unconfirmed_seen
                or timing.reaction_remaining == 0
            )
        if (
            timing.reaction_valid
            and timing.reaction_remaining_valid
            and timing.reaction_remaining > 0
        ):
            reaction_confirmed_seen = True
    reaction_validity_passed = (
        reaction_unconfirmed_seen and reaction_confirmed_seen
    )

    # Kototsuki contains a back direction. It is still an attack script and
    # must not be mislabeled as guard merely because the direction overlaps.
    false_guard_before, false_guard_after, false_guard_frames = direct_trace(
        15, 8, 35
    )
    false_guard_events = flat(false_guard_frames)
    false_guard_passed = (
        false_guard_after.p1_health < false_guard_before.p1_health
        and any(
            event.event_type == STEP_EVENT_CLEAN_HIT
            for event in false_guard_events
        )
        and not any(
            event.event_type in (
                STEP_EVENT_BLOCK_CONTACT,
                STEP_EVENT_MANUAL_BLOCK_SUCCESS,
                STEP_EVENT_AUTO_GUARD,
                STEP_EVENT_BLOCKSTUN_STARTED,
                STEP_EVENT_BLOCKSTUN_ENDED,
            )
            for event in false_guard_events
        )
    )

    # Crouch guard must classify a low crouching attack as a manual block too.
    crouch_before, crouch_after, crouch_frames = direct_trace(
        3, 11, 35, repeat_p1=True
    )
    crouch_events = flat(crouch_frames)
    crouch_block_passed = (
        crouch_after.p1_health == crouch_before.p1_health
        and any(
            event.event_type == STEP_EVENT_MANUAL_BLOCK_SUCCESS
            and event.action_id == 11
            for event in crouch_events
        )
        and not any(
            event.event_type == STEP_EVENT_CLEAN_HIT
            for event in crouch_events
        )
    )
    light_blockstun_passed = any(
        event.event_type == STEP_EVENT_BLOCK_CONTACT
        and event.expected_blockstun_frames == 9
        and event.expected_blockstun_source == 1
        for event in crouch_events
    )
    light_blockstun_boundary_passed = blockstun_boundary_passed(
        crouch_frames,
        11,
        9,
    )

    # Action 4 是另一個站立防禦 Action；它必須和 Action 2 一樣產生
    # MANUAL_BLOCK_SUCCESS，不能因為 C++ 漏列而被標成 AUTO_GUARD。
    action4_before, action4_after, action4_frames = direct_trace(
        4, 8, 40, repeat_p1=True
    )
    action4_events = flat(action4_frames)
    action4_block_passed = (
        action4_after.p1_health == action4_before.p1_health
        and any(
            event.event_type == STEP_EVENT_MANUAL_BLOCK_SUCCESS
            and event.action_id == 8
            for event in action4_events
        )
        and not any(
            event.event_type == STEP_EVENT_AUTO_GUARD
            for event in action4_events
        )
        and blockstun_boundary_passed(action4_frames, 8, 17)
    )

    # 用 Close C → 七十五式改形成同一條 block string。最後的
    # BLOCKSTUN_ENDED 必須歸因到最後實際被擋的七十五式改，而非 Close C。
    client.load_state(state_path)
    client.set_p2_training_ai(False)
    client.set_p2_action_ai(True)
    client.set_p2_action(8)
    p2_followup_queued = False
    block_string_events = []
    for _frame in range(180):
        if (
            not p2_followup_queued
            and client.can_queue_p2_action(26)
        ):
            client.set_p2_action(26)
            p2_followup_queued = True
        client.step(2, 1)
        block_string_events.extend(client.step_events())
    block_string_end_events = [
        event
        for event in block_string_events
        if event.event_type == STEP_EVENT_BLOCKSTUN_ENDED
    ]
    block_string_attribution_passed = (
        p2_followup_queued
        and any(
            event.event_type == STEP_EVENT_MANUAL_BLOCK_SUCCESS
            and event.action_id == 26
            for event in block_string_events
        )
        and bool(block_string_end_events)
        and block_string_end_events[-1].action_id == 26
        and block_string_end_events[-1].action_serial > 0
    )

    # Same-frame trades emit one event per damaged side. No single event may
    # carry both deltas, otherwise Python reward aggregation double-counts it.
    trade_before, trade_after, trade_frames = direct_trace(6, 6, 25)
    trade_events = flat(trade_frames)
    trade_frame_found = any(
        sum(int(event.p1_hp_delta) for event in events) > 0
        and sum(int(event.p2_hp_delta) for event in events) > 0
        for _frame, events in trade_frames
    )
    trade_p1_damage = sum(
        int(event.p1_hp_delta)
        for event in trade_events
        if event.event_type in (
            STEP_EVENT_P1_DAMAGE,
            STEP_EVENT_CHIP_DAMAGE,
            STEP_EVENT_CLEAN_HIT,
        )
    )
    trade_p2_damage = sum(
        int(event.p2_hp_delta)
        for event in trade_events
        if event.event_type in (STEP_EVENT_COMBO_HIT, STEP_EVENT_DAMAGE_ONLY)
    )
    trade_passed = (
        trade_frame_found
        and trade_p1_damage == trade_before.p1_health - trade_after.p1_health
        and trade_p2_damage == trade_before.p2_health - trade_after.p2_health
        and all(
            not (event.p1_hp_delta > 0 and event.p2_hp_delta > 0)
            for event in trade_events
        )
    )

    # Physical scripted P2 uses p2_training_action_id_, not the direct-action
    # field. Damage events still need the actual action id and a non-zero serial.
    client.load_state(state_path)
    client.set_p2_action_ai(False)
    client.set_p2_style(P2Style.ONIYAKI)
    client.set_p2_training_ai(True)
    physical_events = []
    for _frame in range(90):
        client.step(0, 1)
        physical_events.extend(client.step_events())
    physical_p2_passed = any(
        event.event_type == STEP_EVENT_CLEAN_HIT
        and event.action_id == 16
        and event.action_serial > 0
        and event.p1_hp_delta > 0
        for event in physical_events
    )

    # Raw P2 joypad has no high-level Action ID. Physical block detection must
    # still emit one reaction transaction and may legitimately attribute the
    # attacker as -1/0.
    client.load_state(state_path)
    client.set_p2_training_ai(False)
    client.set_p2_action_ai(False)
    raw_p2_events = []
    for frame in range(55):
        p2_input = JoypadState()
        p2_input.c = 1 if frame < 4 else 0
        client.set_joypad_for_port(1, p2_input)
        client.step(2, 1)
        raw_p2_events.extend(client.step_events())
    raw_block = next(
        (
            event
            for event in raw_p2_events
            if event.event_type == STEP_EVENT_BLOCK_CONTACT
            and event.action_id == -1
            and event.action_serial == 0
            and event.guard_reaction_serial > 0
        ),
        None,
    )
    raw_manual = next(
        (
            event
            for event in raw_p2_events
            if event.event_type == STEP_EVENT_MANUAL_BLOCK_SUCCESS
        ),
        None,
    )
    raw_end = next(
        (
            event
            for event in raw_p2_events
            if event.event_type == STEP_EVENT_BLOCKSTUN_ENDED
        ),
        None,
    )
    raw_p2_block_passed = (
        raw_block is not None
        and raw_manual is not None
        and raw_end is not None
        and int(raw_manual.event_epoch) == int(raw_block.event_epoch)
        == int(raw_end.event_epoch)
        and int(raw_manual.guard_reaction_serial)
        == int(raw_block.guard_reaction_serial)
        == int(raw_end.guard_reaction_serial)
    )

    checks = {
        "grounded-before": grounded_airborne_passed,
        "airborne-before": airborne_before_passed,
        "hit-stop-target": hit_stop_target_passed,
        "normal-block": normal_block_passed,
        "block-stop-defender": block_stop_defender_passed,
        "heavy-blockstun-boundary": heavy_blockstun_boundary_passed,
        "chip-block": chip_block_passed,
        "v5-chip-exclusive": v5_chip_exclusive_passed,
        "reaction-validity": reaction_validity_passed,
        "crouch-block": crouch_block_passed,
        "light-blockstun": light_blockstun_passed,
        "light-blockstun-boundary": light_blockstun_boundary_passed,
        "action4-block": action4_block_passed,
        "block-string-attribution": block_string_attribution_passed,
        "false-guard": false_guard_passed,
        "same-frame-trade": trade_passed,
        "physical-p2-action": physical_p2_passed,
        "raw-p2-block": raw_p2_block_passed,
    }
    passed = all(checks.values())
    print(
        f"[{'PASS' if passed else 'FAIL'}] step event counterexamples: "
        + " ".join(f"{name}={value}" for name, value in checks.items())
    )
    return passed


def verify_defense_counter_deadline() -> bool:
    """Counter reward must use absolute frames and the started action serial."""

    def make_env() -> Kof98Env:
        env = object.__new__(Kof98Env)
        env.action_repeat = 4
        env.defense_counter_window_frames = 0
        env.defense_blockstun_pending = False
        env.defense_blockstun_pending_deadline_frame = -1
        env.defense_blockstun_pending_epoch = -1
        env.defense_blockstun_pending_reaction_serial = -1
        env.defense_counter_window_start_frame = -1
        env.defense_counter_window_deadline_frame = -1
        env.defense_counter_action_id = -1
        env.defense_counter_action_serial = -1
        return env

    def event(
        event_type: int,
        frame_offset: int,
        action_id: int,
        action_serial: int,
        combo_before: int = 0,
        combo_after: int = 0,
        event_epoch: int = 1,
        guard_reaction_serial: int = 0,
        block_contact: bool = False,
        absolute_frame: int | None = None,
        source_player: int = 1,
        target_player: int = 2,
    ) -> StepEventV5:
        result = StepEventV5()
        result.event_type = event_type
        result.frame_offset = frame_offset
        result.absolute_engine_frame = (
            frame_offset if absolute_frame is None else absolute_frame
        )
        result.action_id = action_id
        result.action_serial = action_serial
        result.combo_before = combo_before
        result.combo_after = combo_after
        result.event_epoch = event_epoch
        result.guard_reaction_serial = guard_reaction_serial
        result.block_contact = 1 if block_contact else 0
        result.source_player = source_player
        result.target_player = target_player
        return result

    def start_boundary_case(start_age: int) -> bool:
        env = make_env()
        env._open_defense_counter_window(100)
        _hits, _opened, _existed, expired = (
            env._process_defense_counter_events(
            [
                event(
                    STEP_EVENT_ACTION_STARTED,
                    0,
                    8,
                    101,
                    absolute_frame=101 + start_age,
                )
            ],
            101 + start_age,
            )
        )
        accepted = env.defense_counter_action_serial == 101
        return accepted if start_age == 35 else not accepted and expired

    def delayed_hit_case(hit_age_after_start: int) -> bool:
        env = make_env()
        env._open_defense_counter_window(100)
        start_frame = 136
        env._process_defense_counter_events(
            [
                event(
                    STEP_EVENT_ACTION_STARTED,
                    0,
                    8,
                    101,
                    absolute_frame=start_frame,
                )
            ],
            start_frame,
        )
        hit_frame = start_frame + hit_age_after_start
        hits, _opened, _existed, expired = env._process_defense_counter_events(
            [
                event(
                    STEP_EVENT_COMBO_HIT,
                    0,
                    8,
                    101,
                    0,
                    1,
                    absolute_frame=hit_frame,
                )
            ],
            hit_frame,
        )
        return bool(hits) if hit_age_after_start == 5 else not hits and expired

    frame35_passed = start_boundary_case(35)
    frame36_passed = start_boundary_case(36)
    frame37_passed = start_boundary_case(37)
    delayed_hit_passed = delayed_hit_case(5)
    hit_timeout_passed = delayed_hit_case(COMBO_PROFILE.action_hit_timeout)

    busy_env = make_env()
    busy_env._open_defense_counter_window(100)
    _hits, _opened, _existed, busy_expired = (
        busy_env._process_defense_counter_events([], 137)
    )
    busy_time_passed = (
        busy_expired and busy_env.defense_counter_window_frames == 0
    )

    serial_env = make_env()
    serial_env._open_defense_counter_window(100)
    serial_env._process_defense_counter_events(
        [
            event(
                STEP_EVENT_ACTION_STARTED,
                0,
                8,
                500,
                absolute_frame=101,
            )
        ],
        101,
    )
    old_hits, _opened, _existed, _expired = (
        serial_env._process_defense_counter_events(
            [
                event(
                    STEP_EVENT_COMBO_HIT,
                    0,
                    8,
                    499,
                    0,
                    1,
                    absolute_frame=108,
                )
            ],
            108,
        )
    )
    matching_hits, _opened, _existed, _expired = (
        serial_env._process_defense_counter_events(
            [
                event(
                    STEP_EVENT_COMBO_HIT,
                    0,
                    8,
                    500,
                    0,
                    1,
                    absolute_frame=112,
                )
            ],
            112,
        )
    )
    serial_passed = not old_hits and bool(matching_hits)

    end_serial_env = make_env()
    end_serial_env._process_defense_counter_events(
        [
            event(
                STEP_EVENT_MANUAL_BLOCK_SUCCESS,
                0,
                -1,
                0,
                event_epoch=7,
                guard_reaction_serial=41,
                block_contact=True,
                absolute_frame=90,
                source_player=2,
                target_player=1,
            )
        ],
        90,
    )
    _hits, invalid_opened, _existed, _expired = (
        end_serial_env._process_defense_counter_events(
            [
                event(
                    STEP_EVENT_BLOCKSTUN_ENDED,
                    0,
                    8,
                    700,
                    event_epoch=7,
                    guard_reaction_serial=42,
                    absolute_frame=100,
                    source_player=2,
                    target_player=1,
                )
            ],
            100,
        )
    )
    invalid_end_rejected = (
        not invalid_opened
        and end_serial_env.defense_blockstun_pending
        and end_serial_env.defense_counter_window_start_frame < 0
    )
    _hits, valid_opened, _existed, _expired = (
        end_serial_env._process_defense_counter_events(
            [
                event(
                    STEP_EVENT_BLOCKSTUN_ENDED,
                    0,
                    -1,
                    0,
                    event_epoch=7,
                    guard_reaction_serial=41,
                    absolute_frame=101,
                    source_player=2,
                    target_player=1,
                )
            ],
            101,
        )
    )
    reliable_end_passed = (
        invalid_end_rejected
        and valid_opened
        and not end_serial_env.defense_blockstun_pending
        and end_serial_env.defense_counter_window_start_frame == 102
    )

    epoch_env = make_env()
    epoch_env._process_defense_counter_events(
        [
            event(
                STEP_EVENT_MANUAL_BLOCK_SUCCESS,
                0,
                -1,
                0,
                event_epoch=10,
                guard_reaction_serial=3,
                block_contact=True,
                absolute_frame=50,
                source_player=2,
                target_player=1,
            )
        ],
        50,
    )
    _hits, cross_epoch_opened, _existed, _expired = (
        epoch_env._process_defense_counter_events(
            [
                event(
                    STEP_EVENT_BLOCKSTUN_ENDED,
                    0,
                    -1,
                    0,
                    event_epoch=11,
                    guard_reaction_serial=3,
                    absolute_frame=60,
                    source_player=2,
                    target_player=1,
                )
            ],
            60,
        )
    )
    cross_epoch_rejected = (
        not cross_epoch_opened
        and not epoch_env.defense_blockstun_pending
        and epoch_env.defense_counter_window_start_frame < 0
    )

    wrong_direction_env = make_env()
    wrong_direction_env._process_defense_counter_events(
        [
            event(
                STEP_EVENT_MANUAL_BLOCK_SUCCESS,
                0,
                8,
                900,
                event_epoch=12,
                guard_reaction_serial=77,
                block_contact=True,
                absolute_frame=80,
                source_player=1,
                target_player=2,
            )
        ],
        80,
    )
    _hits, wrong_direction_opened, _existed, _expired = (
        wrong_direction_env._process_defense_counter_events(
            [
                event(
                    STEP_EVENT_BLOCKSTUN_ENDED,
                    0,
                    8,
                    900,
                    event_epoch=12,
                    guard_reaction_serial=77,
                    absolute_frame=90,
                    source_player=1,
                    target_player=2,
                )
            ],
            90,
        )
    )
    wrong_direction_rejected = (
        not wrong_direction_opened
        and not wrong_direction_env.defense_blockstun_pending
        and wrong_direction_env.defense_counter_window_start_frame < 0
    )

    checks = {
        "frame35": frame35_passed,
        "frame36": frame36_passed,
        "frame37": frame37_passed,
        "delayed-hit": delayed_hit_passed,
        "hit-timeout": hit_timeout_passed,
        "busy-expiry": busy_time_passed,
        "serial-binding": serial_passed,
        "reaction-binding": reliable_end_passed,
        "cross-epoch": cross_epoch_rejected,
        "wrong-direction": wrong_direction_rejected,
    }
    passed = all(checks.values())
    print(
        f"[{'PASS' if passed else 'FAIL'}] defense counter deadline: "
        + " ".join(f"{name}={value}" for name, value in checks.items())
    )
    return passed


def verify_arcade_reaction_state_timing(
    client: KofEnvClient,
    state_path: Path,
) -> bool:
    """Validate the arcade D2/E3 reaction states with 68K byte ordering.

    copy_system_ram() exposes FBNeo's raw 68K memory layout, so logical byte
    addresses must use address ^ 1. The old test omitted this conversion and
    therefore read logical D3 while labelling it D2. D2:D3 is now explicitly
    decoded as a big-endian signed 16-bit reaction countdown.
    """

    p1_base = 0x8100
    p2_base = 0x8300
    reaction_d2_offset = 0xD2
    reaction_e3_offset = 0xE3
    steam_hitstun_offset = 0xD4

    def trace(
        p1_action: int,
        p2_action: int | None,
        *,
        repeat_p1: bool = False,
    ):
        client.load_state(state_path)
        client.set_p2_training_ai(False)
        client.set_p2_action_ai(p2_action is not None)
        if p2_action is not None:
            client.set_p2_action(p2_action)

        snapshots = []
        contact_frame = None
        for frame in range(70):
            client.step(p1_action if repeat_p1 or frame == 0 else 0, 1)
            snapshots.append(client.copy_system_ram())
            events = client.step_events()
            if contact_frame is None and any(
                event.event_type in (
                    STEP_EVENT_COMBO_HIT,
                    STEP_EVENT_DAMAGE_ONLY,
                    STEP_EVENT_BLOCK_CONTACT,
                    STEP_EVENT_CHIP_DAMAGE,
                    STEP_EVENT_CLEAN_HIT,
                )
                for event in events
            ):
                contact_frame = frame
        return snapshots, contact_frame

    def read_logical_u8(snapshot: bytes, address: int) -> int:
        return int(snapshot[address ^ 1])

    def read_logical_s16_be(snapshot: bytes, address: int) -> int:
        value = (
            read_logical_u8(snapshot, address) << 8
        ) | read_logical_u8(snapshot, address + 1)
        return value - 0x10000 if value >= 0x8000 else value

    def block_timing(
        snapshots: list[bytes],
        player_base: int,
        contact_frame: int | None,
    ) -> tuple[int, int] | None:
        if contact_frame is None:
            return None

        start_frame = next((
            frame
            for frame in range(contact_frame, len(snapshots))
            if read_logical_u8(
                snapshots[frame],
                player_base + reaction_d2_offset,
            ) == 0
            and (
                read_logical_u8(
                    snapshots[frame],
                    player_base + reaction_e3_offset,
                ) & 0x20
            ) != 0
        ), None)
        if start_frame is None:
            return None

        end_frame = next((
            frame
            for frame in range(start_frame + 1, len(snapshots))
            if read_logical_u8(
                snapshots[frame],
                player_base + reaction_d2_offset,
            ) == 0xFF
            and (
                read_logical_u8(
                    snapshots[frame],
                    player_base + reaction_e3_offset,
                ) & 0x20
            ) == 0
        ), None)
        if end_frame is None:
            return None

        return start_frame - contact_frame, end_frame - start_frame

    def hit_total_reaction_frames(
        snapshots: list[bytes],
        player_base: int,
        contact_frame: int | None,
    ) -> int | None:
        if contact_frame is None:
            return None

        release_frame = next((
            frame
            for frame in range(contact_frame + 1, len(snapshots))
            if read_logical_u8(
                snapshots[frame],
                player_base + reaction_d2_offset,
            ) == 0xFF
        ), None)
        return (
            None
            if release_frame is None
            else release_frame - contact_frame
        )

    def block_d2d3_sequence(
        snapshots: list[bytes],
        player_base: int,
        contact_frame: int | None,
    ) -> list[int]:
        if contact_frame is None:
            return []

        return [
            read_logical_s16_be(
                snapshot,
                player_base + reaction_d2_offset,
            )
            for snapshot in snapshots[contact_frame:]
            if read_logical_u8(
                snapshot,
                player_base + reaction_d2_offset,
            ) == 0
            and (
                read_logical_u8(
                    snapshot,
                    player_base + reaction_e3_offset,
                ) & 0x20
            ) != 0
        ]

    p1_light, p1_light_contact = trace(6, None)
    p1_heavy, p1_heavy_contact = trace(8, None)
    p2_heavy, p2_heavy_contact = trace(0, 8)
    light_block, light_block_contact = trace(3, 11, repeat_p1=True)
    heavy_block, heavy_block_contact = trace(2, 8, repeat_p1=True)

    # Ground normals use 11F hitstop in the verified arcade data. The D2/E3
    # interval after it is 9F for light block and 17F for heavy block.
    light_block_timing = block_timing(
        light_block,
        p1_base,
        light_block_contact,
    )
    heavy_block_timing = block_timing(
        heavy_block,
        p1_base,
        heavy_block_contact,
    )
    light_block_d2d3 = block_d2d3_sequence(
        light_block,
        p1_base,
        light_block_contact,
    )
    heavy_block_d2d3 = block_d2d3_sequence(
        heavy_block,
        p1_base,
        heavy_block_contact,
    )
    p1_light_total = hit_total_reaction_frames(
        p1_light,
        p2_base,
        p1_light_contact,
    )
    p1_heavy_total = hit_total_reaction_frames(
        p1_heavy,
        p2_base,
        p1_heavy_contact,
    )
    p2_heavy_total = hit_total_reaction_frames(
        p2_heavy,
        p1_base,
        p2_heavy_contact,
    )

    steam_d4_rejected = all(
        read_logical_u8(
            snapshot,
            player_base + steam_hitstun_offset,
        ) == 0
        and read_logical_u8(
            snapshot,
            player_base + steam_hitstun_offset + 1,
        ) == 0
        for snapshots, player_base, contact in (
            (p1_light, p2_base, p1_light_contact),
            (p1_heavy, p2_base, p1_heavy_contact),
            (p2_heavy, p1_base, p2_heavy_contact),
            (light_block, p1_base, light_block_contact),
            (heavy_block, p1_base, heavy_block_contact),
        )
        if contact is not None
        for snapshot in snapshots[contact : contact + 20]
    )

    checks = {
        "light-hit-11+11": p1_light_total == 22,
        "p1-heavy-hit-11+19": p1_heavy_total == 30,
        "p2-heavy-hit-11+19": p2_heavy_total == 30,
        "light-block-11+9": light_block_timing == (11, 9),
        "heavy-block-11+17": heavy_block_timing == (11, 17),
        "light-d2d3-8-to-0": light_block_d2d3 == list(range(8, -1, -1)),
        "heavy-d2d3-16-to-0": heavy_block_d2d3 == list(range(16, -1, -1)),
        "steam-d4-rejected": steam_d4_rejected,
    }
    passed = all(checks.values())
    print(
        f"[{'PASS' if passed else 'FAIL'}] arcade reaction timing: "
        + " ".join(f"{name}={value}" for name, value in checks.items())
    )
    print(
        "[INFO] arcade block D2:D3 signed sequences: "
        f"light={light_block_d2d3} heavy={heavy_block_d2d3}"
    )
    return passed


def verify_guard_release_actionability(
    client: KofEnvClient,
    state_path: Path,
) -> bool:
    """確認地面防禦反應結束當幀即可接受新的按鍵邊緣。

    攻擊方用重攻擊，防守方以 raw joypad 持續按後並交替敲 A。P1/P2
    對稱各測一次。這裡不用防守方的 action API，因為 action API 只能
    證明前端腳本被接受，不能證明遊戲本身已解除 blockstun。
    """

    player_bases = (0x8100, 0x8300)
    trial_count = 10

    def read_logical_u8(snapshot: bytes, address: int) -> int:
        return int(snapshot[address ^ 1])

    def read_logical_s16_be(snapshot: bytes, address: int) -> int:
        value = (
            read_logical_u8(snapshot, address) << 8
        ) | read_logical_u8(snapshot, address + 1)
        return value - 0x10000 if value >= 0x8000 else value

    def set_defender_input(
        defender_port: int,
        back_is_right: bool,
        press_a: bool,
    ) -> None:
        state = JoypadState()
        state.right = 1 if back_is_right else 0
        state.left = 0 if back_is_right else 1
        state.a = 1 if press_a else 0
        client.set_joypad_for_port(defender_port, state)

    def run_trial(
        mode: str,
        defender_port: int,
    ) -> tuple[int | None, int | None, int | None, set[int]]:
        client.load_state(state_path)
        client.set_p2_training_ai(False)
        strategy_state = client.strategy_state()
        back_is_right = bool(
            strategy_state.p1_facing_left
            if defender_port == 0
            else strategy_state.p2_facing_left
        )
        set_defender_input(defender_port, back_is_right, False)

        if defender_port == 0:
            client.set_p2_action_ai(True)
            client.set_p2_action(8)
        else:
            client.set_p2_action_ai(False)
            client.step(8, 1)

        contact_frame = None
        reaction_end_frame = None
        action_state_frame = None
        action_state_at_contact = None
        pressed_frames: set[int] = set()
        previous_counter = None
        previous_e3 = None
        player_base = player_bases[defender_port]
        reaction_counter_address = player_base + 0xD2
        reaction_e3_address = player_base + 0xE3
        action_state_address = player_base + 0xE0

        for frame in range(90):
            if contact_frame is None:
                press_a = False
            elif mode == "hold":
                press_a = True
            elif mode == "end_frame":
                press_a = (frame - contact_frame) % 2 == 0
            else:
                press_a = (frame - contact_frame) % 2 == 1

            if press_a:
                pressed_frames.add(frame)
            set_defender_input(defender_port, back_is_right, press_a)
            client.run_frames(1)

            snapshot = client.copy_system_ram()
            counter = read_logical_s16_be(
                snapshot,
                reaction_counter_address,
            )
            e3 = read_logical_u8(snapshot, reaction_e3_address)
            action_state = read_logical_u8(
                snapshot,
                action_state_address,
            )
            if contact_frame is None and (
                (e3 & 0x60) == 0x20
            ):
                contact_frame = frame
                action_state_at_contact = action_state

            if (
                contact_frame is not None
                and reaction_end_frame is None
                and previous_counter == 0
                and counter == -1
                and previous_e3 is not None
                and (previous_e3 & 0x60) == 0x20
                and (e3 & 0x60) != 0x20
            ):
                reaction_end_frame = frame

            if (
                contact_frame is not None
                and action_state_frame is None
                and action_state != action_state_at_contact
            ):
                action_state_frame = frame

            if (
                reaction_end_frame is not None
                and (
                    action_state_frame is not None
                    or frame >= reaction_end_frame + 8
                )
            ):
                break

            previous_counter = counter
            previous_e3 = e3

        return (
            contact_frame,
            reaction_end_frame,
            action_state_frame,
            pressed_frames,
        )

    checks = {}
    for defender_port in (0, 1):
        end_frame_trials = [
            run_trial("end_frame", defender_port)
            for _ in range(trial_count)
        ]
        pre_end_trials = [
            run_trial("pre_end_frame", defender_port)
            for _ in range(trial_count)
        ]
        held_trials = [
            run_trial("hold", defender_port)
            for _ in range(trial_count)
        ]

        prefix = f"p{defender_port + 1}"
        checks[f"{prefix}-end-frame-input-accepted"] = all(
            contact is not None
            and end is not None
            and action == end
            and end in pressed
            for contact, end, action, pressed in end_frame_trials
        )
        checks[f"{prefix}-timer0-input-rejected"] = all(
            contact is not None
            and end is not None
            and action == end + 1
            and end - 1 in pressed
            and end not in pressed
            and end + 1 in pressed
            for contact, end, action, pressed in pre_end_trials
        )
        checks[f"{prefix}-held-button-needs-new-edge"] = all(
            contact is not None
            and end is not None
            and action is None
            for contact, end, action, _pressed in held_trials
        )

    neutral = JoypadState()
    client.set_joypad_for_port(0, neutral)
    client.set_joypad_for_port(1, neutral)
    client.set_p2_action_ai(False)

    passed = all(checks.values())
    print(
        f"[{'PASS' if passed else 'FAIL'}] guard release actionability: "
        + " ".join(f"{name}={value}" for name, value in checks.items())
    )
    return passed


def verify_raw_jump_and_blowaway_reactions(
    client: KofEnvClient,
    state_path: Path,
) -> bool:
    """以 raw joypad 區分低跳、正常跳與地面 CD 的反應資料。

    這些測試不依賴 Action 20/21 的腳本名稱，直接控制按住上方向的幀數。
    因此可以防止把低跳攻擊或落地後的站立重攻誤標成正常跳攻擊。
    """

    p1_base = 0x8100

    def read_logical_u8(snapshot: bytes, address: int) -> int:
        return int(snapshot[address ^ 1])

    def read_logical_s16_be(snapshot: bytes, address: int) -> int:
        value = (
            read_logical_u8(snapshot, address) << 8
        ) | read_logical_u8(snapshot, address + 1)
        return value - 0x10000 if value >= 0x8000 else value

    def set_port(port: int, **pressed: bool) -> None:
        state = JoypadState()
        for name, value in pressed.items():
            setattr(state, name, 1 if value else 0)
        client.set_joypad_for_port(port, state)

    def trace(
        *,
        guard: bool,
        up_frames: int,
        attack_start: int,
        buttons: tuple[str, ...],
    ) -> dict[str, object]:
        client.load_state(state_path)
        client.set_p2_training_ai(False)
        client.set_p2_action_ai(False)
        observation = client.observation()
        p2_toward_left = observation.p2_x > observation.p1_x
        p1_back_is_right = bool(client.strategy_state().p1_facing_left)

        contact_frame = None
        contact_y = None
        contact_p2_e0 = None
        reaction_end_frame = None
        previous_counter = None
        previous_e3 = None
        countdown = []
        seen_countdown_values = set()
        min_p2_y = int(observation.p2_y)
        event_types = set()
        post_contact_counters = []

        for frame in range(100):
            set_port(
                0,
                left=guard and not p1_back_is_right,
                right=guard and p1_back_is_right,
            )
            p2_input = {}
            if frame < up_frames:
                p2_input["up"] = True
                p2_input["left" if p2_toward_left else "right"] = True
            if attack_start <= frame < attack_start + 2:
                for button in buttons:
                    p2_input[button] = True
            set_port(1, **p2_input)

            client.run_frames(1)
            observation = client.observation()
            min_p2_y = min(min_p2_y, int(observation.p2_y))
            snapshot = client.copy_system_ram()
            counter = read_logical_s16_be(snapshot, p1_base + 0xD2)
            e3 = read_logical_u8(snapshot, p1_base + 0xE3)
            event_types.update(
                event.event_type
                for event in client.step_events()
            )

            if contact_frame is None and (e3 & 0x60) in (0x20, 0x60):
                contact_frame = frame
                contact_y = int(observation.p2_y)
                contact_p2_e0 = read_logical_u8(snapshot, 0x83E0)

            if contact_frame is not None:
                post_contact_counters.append(counter)
                if (
                    (e3 & 0x60) == 0x20
                    and 0 <= counter <= 255
                    and counter not in seen_countdown_values
                ):
                    seen_countdown_values.add(counter)
                    countdown.append(counter)

            if (
                contact_frame is not None
                and reaction_end_frame is None
                and previous_counter == 0
                and counter == -1
                and previous_e3 is not None
                and (previous_e3 & 0x60) == 0x20
                and (e3 & 0x60) != 0x20
            ):
                reaction_end_frame = frame

            previous_counter = counter
            previous_e3 = e3

        neutral = JoypadState()
        client.set_joypad_for_port(0, neutral)
        client.set_joypad_for_port(1, neutral)
        return {
            "contact_frame": contact_frame,
            "contact_y": contact_y,
            "contact_p2_e0": contact_p2_e0,
            "min_p2_y": min_p2_y,
            "reaction_end_frame": reaction_end_frame,
            "countdown": countdown,
            "event_types": event_types,
            "post_contact_counters": post_contact_counters,
        }

    hop_c = trace(
        guard=True,
        up_frames=2,
        attack_start=8,
        buttons=("c",),
    )
    normal_jump_c = trace(
        guard=True,
        up_frames=6,
        attack_start=20,
        buttons=("c",),
    )
    normal_jump_d = trace(
        guard=True,
        up_frames=6,
        attack_start=18,
        buttons=("d",),
    )
    ground_cd_block = trace(
        guard=True,
        up_frames=0,
        attack_start=8,
        buttons=("c", "d"),
    )
    ground_cd_hit = trace(
        guard=False,
        up_frames=0,
        attack_start=8,
        buttons=("c", "d"),
    )

    checks = {
        "hop-c-airborne": (
            hop_c["contact_y"] is not None
            and int(hop_c["contact_y"]) < 185
            and hop_c["contact_p2_e0"] == 0x0B
        ),
        "hop-c-d2d3-8-to-0": hop_c["countdown"] == list(range(8, -1, -1)),
        "normal-jump-c-airborne": (
            normal_jump_c["contact_y"] is not None
            and int(normal_jump_c["contact_y"]) < 185
            and normal_jump_c["min_p2_y"] == 121
            and normal_jump_c["contact_p2_e0"] == 0x03
        ),
        "normal-jump-c-d2d3-16-to-0": (
            normal_jump_c["countdown"] == list(range(16, -1, -1))
        ),
        "normal-jump-d-d2d3-16-to-0": (
            normal_jump_d["countdown"] == list(range(16, -1, -1))
        ),
        "ground-cd-block-d2d3-20-to-0": (
            ground_cd_block["countdown"] == list(range(20, -1, -1))
        ),
        "ground-cd-hit-is-not-block": (
            STEP_EVENT_CLEAN_HIT in ground_cd_hit["event_types"]
            and STEP_EVENT_BLOCKSTUN_STARTED not in ground_cd_hit["event_types"]
            and STEP_EVENT_BLOCKSTUN_ENDED not in ground_cd_hit["event_types"]
            and -2 in ground_cd_hit["post_contact_counters"]
        ),
    }
    passed = all(checks.values())
    print(
        f"[{'PASS' if passed else 'FAIL'}] raw jump/CD reactions: "
        + " ".join(f"{name}={value}" for name, value in checks.items())
    )
    print(
        "[INFO] raw jump/CD D2:D3: "
        f"hopC={hop_c['countdown']} "
        f"jumpC={normal_jump_c['countdown']} "
        f"jumpD={normal_jump_d['countdown']} "
        f"groundCD={ground_cd_block['countdown']}"
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
        combo_4plus_milestone_reward = 0.0
        combo_4plus_milestone_events = 0
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
            milestone_reward = float(info["reward_fight_combo_4plus_milestone"])
            combo_4plus_milestone_reward += milestone_reward
            combo_4plus_milestone_events += int(milestone_reward > 0.0)
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
            and combo_4plus_milestone_reward == 8.0
            and combo_4plus_milestone_events == 1
            and max_combo >= 7
        )
        status = "PASS" if passed else "FAIL"
        print(
            f"[{status}] fight attribution: queued={queued} started={started} "
            f"followup_hits={followup_hits} hit_actions={sorted(combo_hit_actions)} "
            f"cancel_reward={cancel_reward:.1f} "
            f"milestone_reward={combo_4plus_milestone_reward:.1f} "
            f"max_combo={max_combo}"
        )
        return passed
    finally:
        env.close()


def verify_guided_fight_teacher(root: Path, state_path: Path) -> bool:
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
        fight_guided=True,
        training_profile=TrainingProfile.FIGHT,
        combo_scenario="kyo_close_c_seventy_five_shiki_kai_red_kick",
        action_mask_level=ActionMaskLevel.PHYSICAL,
    )
    try:
        env.reset()
        pending_actions = [8, 26, 17]
        accepted_actions: list[int] = []
        teacher_completions = 0
        max_combo = 0

        for _frame in range(180):
            mask = env.action_masks()
            action_id = 0
            if pending_actions and mask[pending_actions[0]]:
                action_id = pending_actions[0]

            _observation, _reward, terminated, truncated, info = env.step(action_id)
            if (
                pending_actions
                and action_id == pending_actions[0]
                and bool(info["input_accepted"])
            ):
                accepted_actions.append(pending_actions.pop(0))
            teacher_completions += int(info["fight_teacher_complete"])
            max_combo = max(max_combo, int(info["p1_combo_count"]))
            if terminated or truncated or teacher_completions:
                break

        passed = (
            not pending_actions
            and accepted_actions == [8, 26, 17]
            and teacher_completions == 1
            and max_combo >= 4
        )
        status = "PASS" if passed else "FAIL"
        print(
            f"[{status}] guided fight teacher: actions={accepted_actions} "
            f"completions={teacher_completions} max_combo={max_combo}"
        )
        return passed
    finally:
        env.close()


def verify_p2_styles(root: Path, state_path: Path) -> bool:
    fight_state_path = root / "saves" / "states" / "kof98.slot2.state"
    if not fight_state_path.exists():
        fight_state_path = state_path

    env = Kof98Env(
        root / "build-vs2026-x64" / "Release" / "fbneo_training.dll",
        root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll",
        root / "roms" / "fbneo" / "kof98.zip",
        root / "roms" / "fbneo",
        root / "saves",
        state_path=fight_state_path,
        action_repeat=1,
        hitbox_reward=False,
        p2_training_ai=True,
        training_profile=TrainingProfile.FIGHT,
        action_mask_level=ActionMaskLevel.PHYSICAL,
    )
    try:
        all_passed = True
        for style in P2Style:
            env.p2_style = style
            env.reset()
            states = []
            reported_styles = set()
            for _frame in range(360):
                _observation, _reward, terminated, truncated, info = env.step(0)
                states.append(env.client.last_joypad(1))
                reported_styles.add(str(info["p2_style"]))
                if terminated or truncated:
                    env.reset()

            has_horizontal = any(state.left or state.right for state in states)
            has_crouch_guard = any(
                state.down
                and (state.left or state.right)
                and not (state.a or state.b or state.c or state.d)
                for state in states
            )
            has_oniyaki = any(
                state.down
                and (state.left or state.right)
                and state.a
                for state in states
            )
            has_jump = any(state.up and (state.left or state.right) for state in states)
            has_jump_attack = any(state.c or state.d for state in states)
            has_poke = any(state.a or state.b for state in states)

            if style is P2Style.ONIYAKI:
                passed = has_oniyaki
            elif style is P2Style.GUARD:
                passed = has_horizontal and has_crouch_guard and not has_poke
            elif style is P2Style.JUMP_IN:
                passed = has_jump and has_jump_attack
            else:
                passed = has_horizontal and has_poke

            passed = passed and reported_styles == {style.value}
            status = "PASS" if passed else "FAIL"
            print(
                f"[{status}] P2 style {style.value}: "
                f"horizontal={has_horizontal} crouch_guard={has_crouch_guard} "
                f"oniyaki={has_oniyaki} jump={has_jump} "
                f"jump_attack={has_jump_attack} poke={has_poke}"
            )
            all_passed = all_passed and passed

        return all_passed
    finally:
        env.close()


def verify_strategy_v2(root: Path, state_path: Path) -> bool:
    env = Kof98Env(
        root / "build-vs2026-x64" / "Release" / "fbneo_training.dll",
        root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll",
        root / "roms" / "fbneo" / "kof98.zip",
        root / "roms" / "fbneo",
        root / "saves",
        fight_state_path=state_path,
        action_repeat=4,
        hitbox_reward=False,
        p2_training_ai=True,
        training_profile=TrainingProfile.FIGHT,
        action_mask_level=ActionMaskLevel.PHYSICAL,
        observation_version=ObservationVersion.V2,
        fight_reward_version=FightRewardVersion.SYMMETRIC_V2,
    )
    try:
        observation, reset_info = env.reset()
        next_observation, _reward, _terminated, _truncated, info = env.step(0)
        strategy_state = env.client.strategy_state()
        scripted_neutral = np.concatenate((
            next_observation[40:42],
            next_observation[104:133],
        ))
        env.p2_training_ai = False
        human_observation, _human_reset_info = env.reset()
        human_neutral = np.concatenate((
            human_observation[40:42],
            human_observation[104:133],
        ))
        passed = (
            observation.shape == (OBSERVATION_V2_SIZE,)
            and next_observation.shape == (OBSERVATION_V2_SIZE,)
            and np.isfinite(observation).all()
            and np.isfinite(next_observation).all()
            and strategy_state.struct_size == ctypes.sizeof(type(strategy_state))
            and strategy_state.version == 1
            and reset_info["observation_version"] == ObservationVersion.V2.value
            and info["fight_reward_version"] == FightRewardVersion.SYMMETRIC_V2.value
            and info["reward_distance"] == 0.0
            and info["reward_hitbox"] == 0.0
            and info["reward_fight_combo_4plus_milestone"] == 0.0
            and next_observation[40] == 0.0
            and next_observation[41] == 0.0
            and next_observation[43] == 1.0
            and np.count_nonzero(next_observation[104:133]) == 0
            and next_observation[133] == 1.0
            and np.array_equal(scripted_neutral, human_neutral)
            and np.count_nonzero(human_neutral) == 0
        )
        print(
            f"[{'PASS' if passed else 'FAIL'}] StrategyV2: "
            f"shape={observation.shape} abi={strategy_state.struct_size} "
            f"phase={info['combat_phase']} reward={info['fight_reward_version']} "
            f"script_neutral_equal={np.array_equal(scripted_neutral, human_neutral)}"
        )
        return passed
    finally:
        env.close()


def verify_strategy_v4_observation(root: Path, state_path: Path) -> bool:
    def make_env(event_features: bool) -> Kof98Env:
        return Kof98Env(
            root / "build-vs2026-x64" / "Release" / "fbneo_training.dll",
            root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll",
            root / "roms" / "fbneo" / "kof98.zip",
            root / "roms" / "fbneo",
            root / "saves",
            fight_state_path=state_path,
            action_repeat=4,
            hitbox_reward=False,
            p2_training_ai=True,
            training_profile=TrainingProfile.FIGHT,
            action_mask_level=ActionMaskLevel.PHYSICAL,
            observation_version=ObservationVersion.V3,
            observation_event_features=event_features,
            fight_reward_version=FightRewardVersion.SYMMETRIC_V2,
        )

    neutral_env = make_env(False)
    event_env = None
    try:
        neutral, neutral_info = neutral_env.reset()
        # FBNeo/libretro exposes process-global state and cannot safely host
        # two loaded runtimes in one process. Close the neutral-field A/B arm
        # before creating the event-enabled arm.
        neutral_env.close()
        neutral_env = None
        event_env = make_env(True)
        event_observation, event_info = event_env.reset()
        event_observation, _reward, _terminated, _truncated, _info = (
            event_env.step(0)
        )
        timing = event_env.client.combat_timing_state()
        indices = list(OBSERVATION_V3_REPURPOSED_INDICES)
        passed = (
            neutral.shape == (OBSERVATION_V3_SIZE,)
            and event_observation.shape == (OBSERVATION_V3_SIZE,)
            and np.isfinite(neutral).all()
            and np.isfinite(event_observation).all()
            and np.count_nonzero(neutral[indices]) == 0
            and np.count_nonzero(event_observation[indices]) > 0
            and timing.version == 1
            and timing.struct_size == ctypes.sizeof(type(timing))
            and timing.p1.actionable_valid == 0
            and timing.p2.actionable_valid == 0
            and timing.p1.recovery_valid == 0
            and timing.p2.recovery_valid == 0
            and timing.frame_advantage_valid == 0
            and neutral_info["observation_version"]
            == ObservationVersion.V3.value
            and event_info["observation_version"]
            == ObservationVersion.V3.value
        )
        print(
            f"[{'PASS' if passed else 'FAIL'}] StrategyV4 observation: "
            f"shape={event_observation.shape} timing_abi={timing.struct_size} "
            f"neutral_nonzero={np.count_nonzero(neutral[indices])} "
            f"event_nonzero={np.count_nonzero(event_observation[indices])} "
            f"actionable_valid={timing.p1.actionable_valid}/"
            f"{timing.p2.actionable_valid}"
        )
        return passed
    finally:
        if neutral_env is not None:
            neutral_env.close()
        if event_env is not None:
            event_env.close()


def verify_strategy_v4_transition_equivalence(
    root: Path,
    state_path: Path,
) -> bool:
    """固定輸入下，B/C 除了 V3 的 32 個事件欄位外必須完全相同。"""

    def make_env(event_features: bool) -> Kof98Env:
        return Kof98Env(
            root / "build-vs2026-x64" / "Release" / "fbneo_training.dll",
            root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll",
            root / "roms" / "fbneo" / "kof98.zip",
            root / "roms" / "fbneo",
            root / "saves",
            fight_state_path=state_path,
            action_repeat=4,
            hitbox_reward=False,
            p2_training_ai=True,
            p2_style=P2Style.ONIYAKI,
            training_profile=TrainingProfile.FIGHT,
            action_mask_level=ActionMaskLevel.PHYSICAL,
            observation_version=ObservationVersion.V3,
            observation_event_features=event_features,
            fight_reward_version=FightRewardVersion.SYMMETRIC_TACTICAL_V3,
        )

    def event_signature(event: StepEventV5) -> tuple[int, ...]:
        return tuple(
            int(getattr(event, field_name))
            for field_name, _field_type in StepEventV5._fields_
        )

    def run_trace(event_features: bool) -> tuple[np.ndarray, list[tuple]]:
        env = make_env(event_features)
        try:
            observation, _reset_info = env.reset(seed=98)
            trace: list[tuple] = []
            action_pattern = (0, 2, 0, 8, 0, 16, 0, 1)
            for step_index in range(64):
                action_id = action_pattern[step_index % len(action_pattern)]
                action_mask = env.action_masks().copy()
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action_id)
                raw = info["raw"]
                trace.append((
                    tuple(bool(value) for value in action_mask),
                    float(reward),
                    int(raw.round_time),
                    int(raw.p1_health),
                    int(raw.p2_health),
                    int(raw.p1_combo_count),
                    int(raw.p2_combo_count),
                    bool(terminated),
                    bool(truncated),
                    tuple(event_signature(event) for event in env.last_step_events),
                    next_observation.copy(),
                ))
                observation = next_observation
                if terminated or truncated:
                    break
            return observation, trace
        finally:
            env.close()

    neutral_observation, neutral_trace = run_trace(False)
    event_observation, event_trace = run_trace(True)
    indices = list(OBSERVATION_V3_REPURPOSED_INDICES)
    preserved_indices = [
        index
        for index in range(OBSERVATION_V3_SIZE)
        if index not in OBSERVATION_V3_REPURPOSED_INDICES
    ]
    transitions_match = len(neutral_trace) == len(event_trace)
    event_nonzero_seen = False
    if transitions_match:
        for neutral_step, event_step in zip(neutral_trace, event_trace):
            if neutral_step[:-1] != event_step[:-1]:
                transitions_match = False
                break
            neutral_vector = neutral_step[-1]
            event_vector = event_step[-1]
            if not np.array_equal(
                neutral_vector[preserved_indices],
                event_vector[preserved_indices],
            ):
                transitions_match = False
                break
            if np.count_nonzero(neutral_vector[indices]) != 0:
                transitions_match = False
                break
            event_nonzero_seen = (
                event_nonzero_seen
                or np.count_nonzero(event_vector[indices]) > 0
            )

    passed = (
        transitions_match
        and event_nonzero_seen
        and np.count_nonzero(neutral_observation[indices]) == 0
        and np.isfinite(neutral_observation).all()
        and np.isfinite(event_observation).all()
    )
    print(
        f"[{'PASS' if passed else 'FAIL'}] StrategyV4 B/C transition: "
        f"steps={len(neutral_trace)}/{len(event_trace)} "
        f"transitions_match={transitions_match} "
        f"event_nonzero_seen={event_nonzero_seen}"
    )
    return passed


def verify_combo_scenario_rotation(root: Path, state_path: Path) -> bool:
    rotation = [(scenario_name, state_path) for scenario_name in COMBO_SCENARIOS]
    env = Kof98Env(
        root / "build-vs2026-x64" / "Release" / "fbneo_training.dll",
        root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll",
        root / "roms" / "fbneo" / "kof98.zip",
        root / "roms" / "fbneo",
        root / "saves",
        combo_state_path=state_path,
        training_profile=TrainingProfile.COMBO,
        combo_scenario_rotation=rotation,
        combo_rotation_offset=0,
        combo_rotation_stride=1,
        observation_version=ObservationVersion.V2,
    )
    try:
        visited = []
        for _ in rotation:
            _observation, info = env.reset()
            visited.append(info["combo_scenario"])
        passed = set(visited) == set(COMBO_SCENARIOS) and len(visited) == len(rotation)
        print(
            f"[{'PASS' if passed else 'FAIL'}] Combo rotation: "
            f"visited={len(set(visited))}/{len(rotation)}"
        )
        return passed
    finally:
        env.close()


def verify_fight_rotation(root: Path, state_path: Path) -> bool:
    rotation = [
        (FightCurriculum.DEFENSE, P2Style.POKE),
        (FightCurriculum.ANTI_AIR, P2Style.JUMP_IN),
        (FightCurriculum.APPROACH, P2Style.GUARD),
        (FightCurriculum.HIT_CONFIRM, P2Style.POKE),
    ]
    env = Kof98Env(
        root / "build-vs2026-x64" / "Release" / "fbneo_training.dll",
        root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll",
        root / "roms" / "fbneo" / "kof98.zip",
        root / "roms" / "fbneo",
        root / "saves",
        fight_state_path=state_path,
        action_repeat=4,
        p2_training_ai=True,
        training_profile=TrainingProfile.FIGHT,
        fight_rotation=rotation,
        fight_rotation_offset=1,
    )
    try:
        visited = []
        for _ in rotation:
            _observation, info = env.reset()
            visited.append((info["fight_curriculum"], info["p2_style"]))
        expected = [
            (curriculum.value, style.value)
            for curriculum, style in rotation[1:] + rotation[:1]
        ]
        passed = visited == expected
        print(
            f"[{'PASS' if passed else 'FAIL'}] Fight rotation: "
            f"visited={visited}"
        )
        return passed
    finally:
        env.close()


def verify_targeted_fight_masks(root: Path, state_path: Path) -> bool:
    env = Kof98Env(
        root / "build-vs2026-x64" / "Release" / "fbneo_training.dll",
        root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll",
        root / "roms" / "fbneo" / "kof98.zip",
        root / "roms" / "fbneo",
        root / "saves",
        fight_state_path=state_path,
        action_repeat=4,
        p2_training_ai=False,
        training_profile=TrainingProfile.FIGHT,
        action_mask_level=ActionMaskLevel.PHYSICAL,
        observation_version=ObservationVersion.V2,
        fight_reward_version=FightRewardVersion.SYMMETRIC_V2,
    )
    try:
        env.reset()
        observation = env.previous_observation
        strategy_state = env.previous_strategy_state
        if observation is None or strategy_state is None:
            print("[FAIL] Targeted Fight masks: no initial strategy snapshot")
            return False

        observation.p1_has_position = 1
        observation.p2_has_position = 1
        observation.p1_y = 200
        observation.p2_y = 200
        strategy_state.p2_active_action_id = 7

        env.fight_curriculum = FightCurriculum.DEFENSE
        observation.distance_x = 60
        defense = env.action_masks()
        defense_passed = (
            defense[0]
            and defense[2]
            and defense[3]
            and defense[4]
            and not defense[8]
        )

        strategy_state.p2_active_action_id = -1
        env.defense_counter_window_frames = 24
        counter = env.action_masks()
        counter_passed = (
            counter[0]
            and counter[8]
            and counter[11]
            and counter[16]
            and counter.sum() == 4
        )
        env.defense_counter_window_frames = 0

        env.fight_curriculum = FightCurriculum.ANTI_AIR
        strategy_state.p2_active_action_id = 5
        observation.p2_y = 150
        observation.distance_x = 80
        anti_air = env.action_masks()
        anti_air_passed = anti_air[0] and anti_air[16] and anti_air.sum() == 2

        env.fight_curriculum = FightCurriculum.APPROACH
        observation.p2_y = 200
        observation.distance_x = 140
        approach = env.action_masks()
        approach_passed = (
            not approach[0]
            and approach[1]
            and approach[5]
            and approach.sum() == 2
        )

        env.fight_curriculum = FightCurriculum.HIT_CONFIRM
        env.fight_guided = True
        env.fight_teacher_phase = 0
        env.fight_teacher_route_completed = False
        observation.distance_x = 30
        hit_confirm = env.action_masks()
        expected_action = env.combo_scenario.phases[0].action_id
        hit_confirm_passed = (
            hit_confirm[expected_action]
            and hit_confirm.sum() == 1
        )

        passed = all((
            defense_passed,
            counter_passed,
            anti_air_passed,
            approach_passed,
            hit_confirm_passed,
        ))
        print(
            f"[{'PASS' if passed else 'FAIL'}] Targeted Fight masks: "
            f"defense={defense_passed} counter={counter_passed} "
            f"anti_air={anti_air_passed} "
            f"approach={approach_passed} hit_confirm={hit_confirm_passed}"
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
    failed = not verify_tactical_reward_machine()
    try:
        client.load_core(root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll")
        client.load_game(
            root / "roms" / "fbneo" / "kof98.zip",
            root / "roms" / "fbneo",
            root / "saves",
        )
        failed = not verify_runtime_contract_and_hold_chunks(client, state_path) or failed
        failed = not verify_level_recipe_replay(client, root) or failed
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
        failed = not verify_step_event_counterexamples(client, state_path) or failed
        failed = not verify_defense_counter_deadline() or failed
        failed = not verify_arcade_reaction_state_timing(client, state_path) or failed
        failed = not verify_guard_release_actionability(client, state_path) or failed
        failed = not verify_raw_jump_and_blowaway_reactions(client, state_path) or failed
    finally:
        client.close()

    if not args.skip_scenarios:
        failed = not verify_combo_scenarios(root, state_path) or failed
        failed = not verify_physical_reward_cases(root) or failed
        failed = not verify_fight_event_attribution(root, state_path) or failed
        failed = not verify_guided_fight_teacher(root, state_path) or failed
        failed = not verify_p2_styles(root, state_path) or failed
        failed = not verify_strategy_v2(root, state_path) or failed
        failed = not verify_strategy_v4_observation(root, state_path) or failed
        failed = (
            not verify_strategy_v4_transition_equivalence(root, state_path)
            or failed
        )
        failed = not verify_combo_scenario_rotation(root, state_path) or failed
        failed = not verify_fight_rotation(root, state_path) or failed
        failed = not verify_targeted_fight_masks(root, state_path) or failed
        failed = not verify_causal_approach_metric(root) or failed
        failed = not verify_tactical_recipe_oracles(root) or failed

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
