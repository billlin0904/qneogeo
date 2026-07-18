from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - kept for quick local smoke tests.
    gym = None
    spaces = None


class JoypadState(ctypes.Structure):
    _fields_ = [
        ("up", ctypes.c_uint8),
        ("down", ctypes.c_uint8),
        ("left", ctypes.c_uint8),
        ("right", ctypes.c_uint8),
        ("a", ctypes.c_uint8),
        ("b", ctypes.c_uint8),
        ("c", ctypes.c_uint8),
        ("d", ctypes.c_uint8),
        ("start", ctypes.c_uint8),
        ("coin", ctypes.c_uint8),
    ]


class Kof98Observation(ctypes.Structure):
    _fields_ = [
        ("round_time", ctypes.c_int32),
        ("p1_health", ctypes.c_int32),
        ("p2_health", ctypes.c_int32),
        ("p1_power", ctypes.c_int32),
        ("p2_power", ctypes.c_int32),
        ("p1_power_state", ctypes.c_int32),
        ("p2_power_state", ctypes.c_int32),
        ("p1_advanced_power_value", ctypes.c_int32),
        ("p1_advanced_power_stocks", ctypes.c_int32),
        ("p2_advanced_power_value", ctypes.c_int32),
        ("p2_advanced_power_stocks", ctypes.c_int32),
        ("p1_stun", ctypes.c_int32),
        ("p2_stun", ctypes.c_int32),
        ("p1_combo_count", ctypes.c_int32),
        ("p2_combo_count", ctypes.c_int32),
        ("p1_x", ctypes.c_int32),
        ("p1_y", ctypes.c_int32),
        ("p2_x", ctypes.c_int32),
        ("p2_y", ctypes.c_int32),
        ("distance_x", ctypes.c_int32),
        ("distance_y", ctypes.c_int32),
        ("p1_has_position", ctypes.c_uint8),
        ("p2_has_position", ctypes.c_uint8),
    ]


class HitboxRectResult(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int32),
        ("owner", ctypes.c_int32),
        ("left", ctypes.c_int32),
        ("top", ctypes.c_int32),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
    ]


class HitboxAxisResult(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
    ]


class ActionStatus(ctypes.Structure):
    _fields_ = [
        ("active_action_id", ctypes.c_int32),
        ("queued_action_id", ctypes.c_int32),
        ("last_started_action_id", ctypes.c_int32),
        ("action_accepted", ctypes.c_uint8),
        ("step_last_hit_action_id", ctypes.c_int32),
    ]


STEP_EVENTS_VERSION_1 = 1
STEP_EVENT_CAPACITY_V1 = 16
STEP_EVENT_ACTION_STARTED = 1
STEP_EVENT_COMBO_HIT = 2
STEP_EVENT_DAMAGE_ONLY = 3


class StepEventV1(ctypes.Structure):
    _fields_ = [
        ("frame_offset", ctypes.c_int32),
        ("event_type", ctypes.c_int32),
        ("action_id", ctypes.c_int32),
        ("action_serial", ctypes.c_uint32),
        ("combo_before", ctypes.c_int32),
        ("combo_after", ctypes.c_int32),
        ("p2_hp_delta", ctypes.c_int32),
    ]


class StepEventsV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("event_count", ctypes.c_uint32),
        ("dropped_event_count", ctypes.c_uint32),
        ("events", StepEventV1 * STEP_EVENT_CAPACITY_V1),
    ]


VideoRefreshCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_size_t,
    ctypes.c_void_p,
)

HITBOX_ATTACK = 1
HITBOX_VULNERABILITY = 2
HITBOX_PROJECTILE_VULNERABILITY = 3
HITBOX_PROJECTILE_ATTACK = 4

HITBOX_OWNER_P1 = 1
HITBOX_OWNER_P2 = 2
HITBOX_REWARD_SOURCE_WIDTH = 320
HITBOX_REWARD_SOURCE_HEIGHT = 224
# Edge-triggered (False -> True transition only): per-step overlap payouts
# double-count with the hp reward once the attack actually connects.
P1_ATTACK_OVERLAP_REWARD = 0.5
P2_ATTACK_OVERLAP_PENALTY = 0.5
EFFECTIVE_DISTANCE_MIN = 35
EFFECTIVE_DISTANCE_MAX = 85
DISTANCE_TOO_FAR_START = 110
DISTANCE_MAX_PENALTY_AT = 180
DISTANCE_IN_RANGE_REWARD = 0.02
DISTANCE_FAR_PENALTY = 0.04
ACTION_COUNT = 29
GUARD_ACTION_IDS = {2, 3, 4}
DEFENSE_PRESSURE_MARGIN = 28
DEFENSE_GUARD_REWARD = 0.08
DEFENSE_UNGUARDED_PRESSURE_PENALTY = 0.04
DEFENSE_BAD_GUARD_PENALTY = 0.08
FIGHT_REWARD_VERSION = "combo4_milestone_v3"
# Escalating per-hit combo rewards: hit 1 is paid through hp damage, later
# hits pay increasingly more so continuing a chain beats resetting to neutral
# even under KOF98 damage scaling. Hits past 5 pay the cap.
FIGHT_COMBO_HIT_REWARDS = {2: 1.0, 3: 2.0, 4: 3.5, 5: 5.0}
FIGHT_COMBO_HIT_REWARD_CAP = 6.0
FIGHT_COMBO_4PLUS_MILESTONE_HITS = 4
FIGHT_COMBO_4PLUS_MILESTONE_REWARD = 8.0
SUPER_COMBO_BONUS = 3.0
SUPER_ACTION_IDS = {18, 19}
SUPER_POWER_STOCKS_REQUIRED = 1
SUPER_NO_STOCK_PENALTY = 0.5
ONIYAKI_ACTION_ID = 16
P2_AIRBORNE_Y_THRESHOLD = 185
ONIYAKI_ANTI_AIR_BONUS = 1.0
ATTACK_RISK_ACTION_IDS = set(range(14, ACTION_COUNT))
ATTACK_RISK_WINDOW_FRAMES = 24
ATTACK_RISK_CLOSE_DISTANCE = 55
ATTACK_RISK_PUNISH_PENALTY = 4.0
ATTACK_RISK_UNSAFE_CLOSE_PENALTY = 0.12
ATTACK_RISK_SAFE_REWARD = 0.05
FAST_WIN_BONUS_MAX = 15.0
# Per-step shaping terms were tuned at action_repeat=6; scale them by
# action_repeat / 6 so reward accrued per emulated second stays constant.
FIGHT_SHAPING_BASELINE_FRAMES = 6.0
FIGHT_FOLLOWUP_TRACK_WINDOW_FRAMES = 180

IDLE_ACTION_ID = 0
FORWARD_ACTION_ID = 1
CLOSE_C_ACTION_ID = 8
CROUCH_A_ACTION_ID = 10
CROUCH_B_ACTION_ID = 11
ARAGAMI_ACTION_ID = 14
KOTOTSUKI_YOU_ACTION_ID = 15
RED_KICK_ACTION_ID = 17
OROCHINAGI_ACTION_ID = 18
MUSHIKI_ACTION_ID = 19
FORWARD_B_ACTION_ID = 22
POISON_BITE_ACTION_ID = 23
TSUMI_YOMI_ACTION_ID = 24
BATSU_YOMI_ACTION_ID = 25
SEVENTY_FIVE_SHIKI_KAI_ACTION_ID = 26
YANO_SABI_ACTION_ID = 27
MIGIRI_UGACHI_ACTION_ID = 28

COMBO_CLOSE_DISTANCE = 45
COMBO_PHASE_AGE_SCALE_FRAMES = 30.0
COMBO_WRONG_ACTION_PENALTY = 0.1
GUIDED_DISTRACTOR_COUNT = 2

SEVENTY_FIVE_KAI_FINISHER_ACTION_IDS = (
    ARAGAMI_ACTION_ID,
    KOTOTSUKI_YOU_ACTION_ID,
    RED_KICK_ACTION_ID,
    OROCHINAGI_ACTION_ID,
)
SEVENTY_FIVE_KAI_FINISHER_HIT_TIMEOUTS = {
    ARAGAMI_ACTION_ID: 130,
    KOTOTSUKI_YOU_ACTION_ID: 150,
    RED_KICK_ACTION_ID: 120,
    OROCHINAGI_ACTION_ID: 160,
}
SEVENTY_FIVE_KAI_ALTERNATE_REWARD = 8.0


def seventy_five_kai_alternates(designated_action_id: int) -> tuple[tuple[int, float], ...]:
    return tuple(
        (action_id, SEVENTY_FIVE_KAI_ALTERNATE_REWARD)
        for action_id in SEVENTY_FIVE_KAI_FINISHER_ACTION_IDS
        if action_id != designated_action_id
    )


@dataclass(frozen=True)
class ComboPhase:
    action_id: int
    required_combo: int
    reward: float
    wait_after_hit_frames: int = 0
    queue_during_previous: bool = False
    require_combo_increment: bool = False
    require_power_stock_spent: bool = False
    require_damage: bool = True
    hit_timeout_frames: Optional[int] = None
    allow_started_followup_on_hit: bool = False
    # Sibling finishers that legitimately combo from the same opener. Landing
    # one ends the episode with its (smaller) reward instead of a wrong-action
    # penalty, so scenarios sharing an opener stop emitting contradictory
    # gradients against real combos.
    alternate_actions: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True)
class ComboScenario:
    name: str
    phases: tuple[ComboPhase, ...]
    complete_reward: float = 25.0


KYO_CORNER_DOKUGAMI_SCENARIO = ComboScenario(
    name="kyo_corner_dokugami",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(FORWARD_B_ACTION_ID, 2, 3.0),
        ComboPhase(
            POISON_BITE_ACTION_ID,
            4,
            10.0,
            queue_during_previous=True,
            require_combo_increment=True,
        ),
        ComboPhase(TSUMI_YOMI_ACTION_ID, 5, 20.0),
        ComboPhase(
            BATSU_YOMI_ACTION_ID,
            6,
            30.0,
            queue_during_previous=True,
            require_combo_increment=True,
        ),
    ),
)

KYO_FORWARD_B_KOTOTSUKI_SCENARIO = ComboScenario(
    name="kyo_forward_b_kototsuki",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(FORWARD_B_ACTION_ID, 2, 3.0),
        ComboPhase(
            KOTOTSUKI_YOU_ACTION_ID,
            5,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
            # The second hit increments the combo before its delayed explosion updates HP.
            require_damage=False,
            hit_timeout_frames=120,
        ),
    ),
)

KYO_FORWARD_B_OROCHINAGI_SCENARIO = ComboScenario(
    name="kyo_forward_b_orochinagi",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(FORWARD_B_ACTION_ID, 2, 3.0),
        ComboPhase(
            OROCHINAGI_ACTION_ID,
            4,
            30.0,
            queue_during_previous=True,
            require_combo_increment=True,
            require_power_stock_spent=True,
            hit_timeout_frames=140,
        ),
    ),
)

KYO_FORWARD_B_RED_KICK_SCENARIO = ComboScenario(
    name="kyo_forward_b_red_kick",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(FORWARD_B_ACTION_ID, 2, 3.0),
        ComboPhase(
            RED_KICK_ACTION_ID,
            4,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
        ),
    ),
)

KYO_FORWARD_B_ARAGAMI_SCENARIO = ComboScenario(
    name="kyo_forward_b_aragami",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(FORWARD_B_ACTION_ID, 2, 3.0),
        ComboPhase(
            ARAGAMI_ACTION_ID,
            4,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
        ),
    ),
)

KYO_CLOSE_C_SEVENTY_FIVE_SHIKI_KAI_OROCHINAGI_SCENARIO = ComboScenario(
    name="kyo_close_c_seventy_five_shiki_kai_orochinagi",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(
            SEVENTY_FIVE_SHIKI_KAI_ACTION_ID,
            3,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
        ),
        ComboPhase(
            OROCHINAGI_ACTION_ID,
            4,
            30.0,
            queue_during_previous=True,
            require_combo_increment=True,
            require_power_stock_spent=True,
            hit_timeout_frames=160,
            alternate_actions=seventy_five_kai_alternates(OROCHINAGI_ACTION_ID),
        ),
    ),
)

KYO_CLOSE_C_SEVENTY_FIVE_SHIKI_KAI_RED_KICK_SCENARIO = ComboScenario(
    name="kyo_close_c_seventy_five_shiki_kai_red_kick",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(
            SEVENTY_FIVE_SHIKI_KAI_ACTION_ID,
            3,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
        ),
        ComboPhase(
            RED_KICK_ACTION_ID,
            4,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
            hit_timeout_frames=120,
            alternate_actions=seventy_five_kai_alternates(RED_KICK_ACTION_ID),
        ),
    ),
)

KYO_CLOSE_C_SEVENTY_FIVE_SHIKI_KAI_KOTOTSUKI_SCENARIO = ComboScenario(
    name="kyo_close_c_seventy_five_shiki_kai_kototsuki",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(
            SEVENTY_FIVE_SHIKI_KAI_ACTION_ID,
            3,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
        ),
        ComboPhase(
            KOTOTSUKI_YOU_ACTION_ID,
            5,
            20.0,
            queue_during_previous=True,
            require_combo_increment=True,
            require_damage=False,
            hit_timeout_frames=150,
            alternate_actions=seventy_five_kai_alternates(KOTOTSUKI_YOU_ACTION_ID),
        ),
    ),
)

KYO_CLOSE_C_SEVENTY_FIVE_SHIKI_KAI_ARAGAMI_SCENARIO = ComboScenario(
    name="kyo_close_c_seventy_five_shiki_kai_aragami",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(
            SEVENTY_FIVE_SHIKI_KAI_ACTION_ID,
            3,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
        ),
        ComboPhase(
            ARAGAMI_ACTION_ID,
            4,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
            hit_timeout_frames=130,
            alternate_actions=seventy_five_kai_alternates(ARAGAMI_ACTION_ID),
        ),
    ),
)

KYO_CORNER_SEVENTY_FIVE_SHIKI_KAI_ARAGAMI_CHAIN_SCENARIO = ComboScenario(
    name="kyo_corner_seventy_five_shiki_kai_aragami_chain",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(
            SEVENTY_FIVE_SHIKI_KAI_ACTION_ID,
            3,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
        ),
        ComboPhase(
            ARAGAMI_ACTION_ID,
            4,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
            hit_timeout_frames=130,
            allow_started_followup_on_hit=True,
        ),
        ComboPhase(
            YANO_SABI_ACTION_ID,
            5,
            20.0,
            queue_during_previous=True,
            require_combo_increment=True,
            hit_timeout_frames=180,
            allow_started_followup_on_hit=True,
        ),
        ComboPhase(
            MIGIRI_UGACHI_ACTION_ID,
            6,
            30.0,
            queue_during_previous=True,
            require_combo_increment=True,
            hit_timeout_frames=180,
        ),
    ),
)

KYO_CROUCH_B_CROUCH_A_MUSHIKI_SCENARIO = ComboScenario(
    name="kyo_crouch_b_crouch_a_mushiki",
    phases=(
        ComboPhase(CROUCH_B_ACTION_ID, 1, 1.0),
        ComboPhase(
            CROUCH_A_ACTION_ID,
            2,
            8.0,
            queue_during_previous=True,
            require_combo_increment=True,
            hit_timeout_frames=80,
            allow_started_followup_on_hit=True,
        ),
        ComboPhase(
            MUSHIKI_ACTION_ID,
            7,
            40.0,
            queue_during_previous=True,
            require_combo_increment=True,
            require_power_stock_spent=True,
            hit_timeout_frames=180,
        ),
    ),
    complete_reward=35.0,
)

COMBO_SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        KYO_CORNER_DOKUGAMI_SCENARIO,
        KYO_FORWARD_B_KOTOTSUKI_SCENARIO,
        KYO_FORWARD_B_OROCHINAGI_SCENARIO,
        KYO_FORWARD_B_RED_KICK_SCENARIO,
        KYO_FORWARD_B_ARAGAMI_SCENARIO,
        KYO_CLOSE_C_SEVENTY_FIVE_SHIKI_KAI_OROCHINAGI_SCENARIO,
        KYO_CLOSE_C_SEVENTY_FIVE_SHIKI_KAI_RED_KICK_SCENARIO,
        KYO_CLOSE_C_SEVENTY_FIVE_SHIKI_KAI_KOTOTSUKI_SCENARIO,
        KYO_CLOSE_C_SEVENTY_FIVE_SHIKI_KAI_ARAGAMI_SCENARIO,
        KYO_CORNER_SEVENTY_FIVE_SHIKI_KAI_ARAGAMI_CHAIN_SCENARIO,
        KYO_CROUCH_B_CROUCH_A_MUSHIKI_SCENARIO,
    )
}
DEFAULT_COMBO_SCENARIO_NAME = KYO_CORNER_DOKUGAMI_SCENARIO.name


class TrainingProfile(str, Enum):
    COMBO = "combo"
    FIGHT = "fight"


class ActionMaskLevel(str, Enum):
    STRICT = "strict"
    GUIDED = "guided"
    PHYSICAL = "physical"


@dataclass(frozen=True)
class ComboProfileConfig:
    action_repeat: int = 1
    max_episode_steps: int = 480
    damage_reward: float = 0.01
    damage_reward_cap: float = 0.25
    action_hit_timeout: int = 75
    chain_timeout: int = 120
    phase_reset_penalty: float = 1.0
    episode_timeout_penalty: float = 1.0
    ko_without_combo_penalty: float = 3.0
    time_penalty: float = 0.001


COMBO_PROFILE = ComboProfileConfig()


def can_use_super(observation: Optional["Kof98Observation"]) -> bool:
    if observation is None:
        return False

    return observation.p1_advanced_power_stocks >= SUPER_POWER_STOCKS_REQUIRED


def is_p2_airborne(observation: Optional["Kof98Observation"]) -> bool:
    if observation is None or not observation.p2_has_position:
        return False

    return 0 <= observation.p2_y < P2_AIRBORNE_Y_THRESHOLD


class KofEnvClient:
    def __init__(self, dll_path: str | Path):
        self.dll_path = Path(dll_path)
        self.dll = ctypes.CDLL(str(self.dll_path))
        self._configure_api()
        self.handle = self.dll.kof_env_create()
        self._video_callback_ref: Optional[VideoRefreshCallback] = None
        if not self.handle:
            raise RuntimeError("kof_env_create failed")

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.dll.kof_env_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    def load_core(self, core_path: str | Path) -> None:
        self._check(self.dll.kof_env_load_core(self.handle, str(core_path)))

    def load_game(self, game_path: str | Path, system_dir: str | Path, save_dir: str | Path) -> None:
        self._check(
            self.dll.kof_env_load_game(
                self.handle,
                str(game_path),
                str(system_dir),
                str(save_dir),
            )
        )

    def reset(self) -> None:
        self._check(self.dll.kof_env_reset(self.handle))

    def load_state(self, state_path: str | Path) -> None:
        self._check(self.dll.kof_env_load_state(self.handle, str(state_path)))

    def save_state(self, state_path: str | Path) -> None:
        self._check(self.dll.kof_env_save_state(self.handle, str(state_path)))

    def set_video_refresh_callback(
        self,
        callback: Optional[Callable[[int, int, int, int], None]],
    ) -> None:
        if callback is None:
            self._video_callback_ref = None
            self.dll.kof_env_set_video_refresh(self.handle, None, None)
            return

        def _callback(data, width, height, pitch, _user_data):
            callback(data, int(width), int(height), int(pitch))

        self._video_callback_ref = VideoRefreshCallback(_callback)
        self.dll.kof_env_set_video_refresh(
            self.handle,
            ctypes.cast(self._video_callback_ref, ctypes.c_void_p),
            None,
        )

    def set_p2_training_ai(self, enabled: bool) -> None:
        self.dll.kof_env_set_p2_random_ai(self.handle, 1 if enabled else 0)

    def step(self, action_id: int, frames: int = 6) -> Kof98Observation:
        observation = Kof98Observation()
        self._check(self.dll.kof_env_step(self.handle, action_id, frames, ctypes.byref(observation)))
        return observation

    def observation(self) -> Kof98Observation:
        observation = Kof98Observation()
        self._check(self.dll.kof_env_get_observation(self.handle, ctypes.byref(observation)))
        return observation

    def p1_ready_for_action(self) -> bool:
        return bool(self.dll.kof_env_p1_ready_for_action(self.handle))

    def input_ready(self) -> bool:
        return bool(self.dll.kof_env_input_ready(self.handle))

    def can_queue_action(self, action_id: int) -> bool:
        return bool(self.dll.kof_env_can_queue_action(self.handle, int(action_id)))

    def action_status(self) -> ActionStatus:
        status = ActionStatus()
        self._check(self.dll.kof_env_get_action_status(self.handle, ctypes.byref(status)))
        return status

    def step_events(self) -> list[StepEventV1]:
        result = StepEventsV1()
        result.struct_size = ctypes.sizeof(StepEventsV1)
        result.version = STEP_EVENTS_VERSION_1
        self._check(self.dll.kof_env_get_step_events_v1(self.handle, ctypes.byref(result)))
        if result.dropped_event_count:
            raise RuntimeError(
                "Step event buffer overflowed: "
                f"{result.dropped_event_count} event(s) were dropped"
            )

        events: list[StepEventV1] = []
        for index in range(result.event_count):
            source = result.events[index]
            events.append(
                StepEventV1(
                    source.frame_offset,
                    source.event_type,
                    source.action_id,
                    source.action_serial,
                    source.combo_before,
                    source.combo_after,
                    source.p2_hp_delta,
                )
            )
        return events

    def last_joypad(self, port: int = 0) -> JoypadState:
        state = JoypadState()
        self._check(
            self.dll.kof_env_get_last_joypad_for_port(
                self.handle,
                port,
                ctypes.byref(state),
            )
        )
        return state

    def copy_system_ram(self) -> bytes:
        size = self.dll.kof_env_system_ram_size(self.handle)
        if size <= 0:
            return b""

        buffer = ctypes.create_string_buffer(size)
        self._check(self.dll.kof_env_copy_system_ram(self.handle, buffer, size))
        return buffer.raw

    def get_hitbox_overlay(
        self,
        source_width: int,
        source_height: int,
        rect_capacity: int = 128,
        axis_capacity: int = 16,
    ) -> tuple[list[HitboxRectResult], list[HitboxAxisResult]]:
        rect_array_type = HitboxRectResult * rect_capacity
        axis_array_type = HitboxAxisResult * axis_capacity
        rects = rect_array_type()
        axes = axis_array_type()
        rect_count = ctypes.c_uint32()
        axis_count = ctypes.c_uint32()

        self._check(
            self.dll.kof_env_get_hitbox_overlay(
                self.handle,
                int(source_width),
                int(source_height),
                rects,
                rect_capacity,
                ctypes.byref(rect_count),
                axes,
                axis_capacity,
                ctypes.byref(axis_count),
            )
        )

        return (
            [rects[index] for index in range(min(rect_count.value, rect_capacity))],
            [axes[index] for index in range(min(axis_count.value, axis_capacity))],
        )

    def _configure_api(self) -> None:
        self.dll.kof_env_create.restype = ctypes.c_void_p
        self.dll.kof_env_destroy.argtypes = [ctypes.c_void_p]

        self.dll.kof_env_load_core.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        self.dll.kof_env_load_core.restype = ctypes.c_int

        self.dll.kof_env_load_game.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
        ]
        self.dll.kof_env_load_game.restype = ctypes.c_int

        self.dll.kof_env_reset.argtypes = [ctypes.c_void_p]
        self.dll.kof_env_reset.restype = ctypes.c_int

        self.dll.kof_env_load_state.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        self.dll.kof_env_load_state.restype = ctypes.c_int

        self.dll.kof_env_save_state.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        self.dll.kof_env_save_state.restype = ctypes.c_int

        self.dll.kof_env_set_video_refresh.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]

        self.dll.kof_env_set_p2_random_ai.argtypes = [ctypes.c_void_p, ctypes.c_int]

        self.dll.kof_env_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(Kof98Observation),
        ]
        self.dll.kof_env_step.restype = ctypes.c_int

        self.dll.kof_env_get_observation.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(Kof98Observation),
        ]
        self.dll.kof_env_get_observation.restype = ctypes.c_int

        self.dll.kof_env_input_ready.argtypes = [ctypes.c_void_p]
        self.dll.kof_env_input_ready.restype = ctypes.c_int

        self.dll.kof_env_can_queue_action.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        self.dll.kof_env_can_queue_action.restype = ctypes.c_int

        self.dll.kof_env_get_action_status.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ActionStatus),
        ]
        self.dll.kof_env_get_action_status.restype = ctypes.c_int

        self.dll.kof_env_get_step_events_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(StepEventsV1),
        ]
        self.dll.kof_env_get_step_events_v1.restype = ctypes.c_int

        self.dll.kof_env_p1_ready_for_action.argtypes = [ctypes.c_void_p]
        self.dll.kof_env_p1_ready_for_action.restype = ctypes.c_int

        self.dll.kof_env_get_last_joypad_for_port.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.POINTER(JoypadState),
        ]
        self.dll.kof_env_get_last_joypad_for_port.restype = ctypes.c_int

        self.dll.kof_env_system_ram_size.argtypes = [ctypes.c_void_p]
        self.dll.kof_env_system_ram_size.restype = ctypes.c_uint32

        self.dll.kof_env_copy_system_ram.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self.dll.kof_env_copy_system_ram.restype = ctypes.c_int

        self.dll.kof_env_get_hitbox_overlay.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(HitboxRectResult),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(HitboxAxisResult),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.dll.kof_env_get_hitbox_overlay.restype = ctypes.c_int

        self.dll.kof_env_last_error.argtypes = [ctypes.c_void_p]
        self.dll.kof_env_last_error.restype = ctypes.c_char_p

    def _check(self, result: int) -> None:
        if result:
            return

        message = self.dll.kof_env_last_error(self.handle)
        raise RuntimeError(message.decode("utf-8", errors="replace") if message else "fbneo_training failed")


def observation_to_vector(observation: Kof98Observation) -> np.ndarray:
    p1_x = observation.p1_x if observation.p1_has_position else 0
    p1_y = observation.p1_y if observation.p1_has_position else 0
    p2_x = observation.p2_x if observation.p2_has_position else 0
    p2_y = observation.p2_y if observation.p2_has_position else 0
    p1_combo_count = observation.p1_combo_count if observation.p1_combo_count >= 0 else 0
    p2_combo_count = observation.p2_combo_count if observation.p2_combo_count >= 0 else 0
    p1_advanced_power_value = observation.p1_advanced_power_value if observation.p1_advanced_power_value >= 0 else 0
    p1_advanced_power_stocks = observation.p1_advanced_power_stocks if observation.p1_advanced_power_stocks >= 0 else 0
    p2_advanced_power_value = observation.p2_advanced_power_value if observation.p2_advanced_power_value >= 0 else 0
    p2_advanced_power_stocks = observation.p2_advanced_power_stocks if observation.p2_advanced_power_stocks >= 0 else 0

    return np.array(
        [
            observation.round_time / 99.0,
            observation.p1_health / 103.0,
            observation.p2_health / 103.0,
            observation.p1_power / 128.0,
            observation.p2_power / 128.0,
            observation.p1_power_state / 128.0,
            observation.p2_power_state / 128.0,
            p1_advanced_power_value / 128.0,
            p1_advanced_power_stocks / 5.0,
            p2_advanced_power_value / 128.0,
            p2_advanced_power_stocks / 5.0,
            observation.p1_stun / 255.0,
            observation.p2_stun / 255.0,
            p1_combo_count / 99.0,
            p2_combo_count / 99.0,
            p1_x / 320.0,
            p1_y / 224.0,
            p2_x / 320.0,
            p2_y / 224.0,
            observation.distance_x / 320.0,
            observation.distance_y / 224.0,
            float(observation.p1_has_position),
            float(observation.p2_has_position),
        ],
        dtype=np.float32,
    )


def rects_overlap(a: HitboxRectResult, b: HitboxRectResult, margin: int = 0) -> bool:
    return (
        a.left - margin < b.left + b.width
        and a.left + a.width + margin > b.left
        and a.top - margin < b.top + b.height
        and a.top + a.height + margin > b.top
    )


def has_attack_hurtbox_overlap(
    rects: list[HitboxRectResult],
    attacker_owner: int,
    defender_owner: int,
    margin: int = 0,
) -> bool:
    attack_types = {HITBOX_ATTACK, HITBOX_PROJECTILE_ATTACK}
    hurtbox_types = {HITBOX_VULNERABILITY, HITBOX_PROJECTILE_VULNERABILITY}
    attacks = [
        rect for rect in rects
        if rect.owner == attacker_owner and rect.type in attack_types
    ]
    hurtboxes = [
        rect for rect in rects
        if rect.owner == defender_owner and rect.type in hurtbox_types
    ]

    return any(rects_overlap(attack, hurtbox, margin) for attack in attacks for hurtbox in hurtboxes)


class Kof98Env(gym.Env if gym else object):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dll_path: str | Path,
        core_path: str | Path,
        game_path: str | Path,
        system_dir: str | Path,
        save_dir: str | Path,
        state_path: Optional[str | Path] = None,
        action_repeat: int = 6,
        hitbox_reward: bool = True,
        p2_training_ai: bool = False,
        training_profile: TrainingProfile | str = TrainingProfile.COMBO,
        combo_state_path: Optional[str | Path] = None,
        fight_state_path: Optional[str | Path] = None,
        combo_scenario: ComboScenario | str = DEFAULT_COMBO_SCENARIO_NAME,
        action_mask_level: ActionMaskLevel | str = ActionMaskLevel.STRICT,
    ):
        if gym is None or spaces is None:
            raise RuntimeError("Install gymnasium before using Kof98Env")

        super().__init__()
        self.client = KofEnvClient(dll_path)
        self.client.load_core(core_path)
        self.client.load_game(game_path, system_dir, save_dir)
        self.state_path = Path(state_path) if state_path else None
        self.combo_state_path = Path(combo_state_path) if combo_state_path else self.state_path
        self.fight_state_path = Path(fight_state_path) if fight_state_path else self.state_path
        self.training_profile = TrainingProfile(training_profile)
        self.action_mask_level = ActionMaskLevel(action_mask_level)
        if isinstance(combo_scenario, ComboScenario):
            self.combo_scenario = combo_scenario
        else:
            try:
                self.combo_scenario = COMBO_SCENARIOS[combo_scenario]
            except KeyError as error:
                raise ValueError(f"Unknown combo scenario: {combo_scenario}") from error
        self.action_repeat = (
            COMBO_PROFILE.action_repeat
            if self.training_profile is TrainingProfile.COMBO
            else action_repeat
        )
        self.fight_frame_scale = float(self.action_repeat) / FIGHT_SHAPING_BASELINE_FRAMES
        self.hitbox_reward = hitbox_reward and self.training_profile is TrainingProfile.FIGHT
        self.p2_training_ai = p2_training_ai
        self.previous_observation: Optional[Kof98Observation] = None
        self.pending_attack_risk: Optional[dict[str, float]] = None
        self.fight_pending_followups: list[dict] = []
        self.fight_prev_p1_attack_overlap = False
        self.fight_prev_p2_attack_overlap = False
        self.fight_combo_4plus_rewarded = False
        self.episode_steps = 0
        self.episode_max_combo = 0
        self.combo_phase = 0
        self.pending_chain_action: Optional[int] = None
        self.pending_action_age = 0
        self.pending_action_power_stocks: Optional[int] = None
        self.pending_followups: dict[int, int] = {}
        self.pending_alternate_action: Optional[int] = None
        self.pending_alternate_reward = 0.0
        self.pending_alternate_age = 0
        self.pending_alternate_min_combo = 0
        self.pending_alternate_power_stocks: Optional[int] = None
        self.pending_alternate_hit_timeout = COMBO_PROFILE.action_hit_timeout
        self.frames_since_chain_hit = 0
        self.phase_wait_remaining = 0

        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(26,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        state_path = self.combo_state_path if self.training_profile is TrainingProfile.COMBO else self.fight_state_path
        if state_path:
            self.client.load_state(state_path)
        else:
            self.client.reset()

        p2_training_ai = self.p2_training_ai if self.training_profile is TrainingProfile.FIGHT else False
        self.client.set_p2_training_ai(p2_training_ai)
        observation = self.client.observation()
        self.previous_observation = observation
        self.pending_attack_risk = None
        self.fight_pending_followups = []
        self.fight_prev_p1_attack_overlap = False
        self.fight_prev_p2_attack_overlap = False
        self.fight_combo_4plus_rewarded = False
        self.episode_steps = 0
        self.episode_max_combo = 0
        self.combo_phase = 0
        self.pending_chain_action = None
        self.pending_action_age = 0
        self.pending_action_power_stocks = None
        self.pending_followups = {}
        self.pending_alternate_action = None
        self.pending_alternate_reward = 0.0
        self.pending_alternate_age = 0
        self.pending_alternate_min_combo = 0
        self.pending_alternate_power_stocks = None
        self.pending_alternate_hit_timeout = COMBO_PROFILE.action_hit_timeout
        self.frames_since_chain_hit = 0
        self.phase_wait_remaining = 0
        return self._make_observation(observation), {
            "raw": observation,
            "training_profile": self.training_profile.value,
            "combo_scenario": self.combo_scenario.name,
            "action_mask_level": self.action_mask_level.value,
        }

    def step(self, action):
        action_id = int(action)
        if self.training_profile is TrainingProfile.COMBO:
            return self._step_combo(action_id)

        return self._step_fight(action_id)

    def _combo_phase_at(self, phase_index: int) -> Optional[ComboPhase]:
        if 0 <= phase_index < len(self.combo_scenario.phases):
            return self.combo_scenario.phases[phase_index]
        return None

    def _current_combo_phase(self) -> Optional[ComboPhase]:
        return self._combo_phase_at(self.combo_phase)

    def _queueable_followup_phase(self) -> Optional[ComboPhase]:
        # Scan every remaining phase: at repeat=1 a follow-up window can open
        # before the previous phase's hit registers (e.g. 75 Kai fires on
        # frame 5 while Close C's hit lands on frame 7), so the queueable
        # phase may sit more than one step past combo_phase.
        for phase_index in range(self.combo_phase, len(self.combo_scenario.phases)):
            phase = self._combo_phase_at(phase_index)
            if (
                phase is not None
                and phase.queue_during_previous
                and self.client.can_queue_action(phase.action_id)
            ):
                return phase
        return None

    def _step_combo(self, action_id: int):
        previous = self.previous_observation
        self.phase_wait_remaining = max(
            0,
            self.phase_wait_remaining - COMBO_PROFILE.action_repeat,
        )
        input_ready_before_step = self._combo_input_ready()
        queueable_followup_phase = self._queueable_followup_phase()
        expected_phase_before_step = queueable_followup_phase or self._current_combo_phase()
        observation = self.client.step(action_id, COMBO_PROFILE.action_repeat)
        action_status = self.client.action_status()
        action_accepted = bool(action_status.action_accepted)
        self._track_pending_chain_action(
            action_id,
            input_ready_before_step,
            action_accepted,
        )
        # Alternates from every remaining phase: an early queue can target a
        # phase more than one step ahead while combo_phase lags the DLL state.
        expected_alternates: dict[int, float] = {}
        alternate_min_combo: dict[int, int] = {}
        alternate_hit_timeouts: dict[int, int] = {}
        for phase_index in range(self.combo_phase, len(self.combo_scenario.phases)):
            future_phase = self.combo_scenario.phases[phase_index]
            previous_required = (
                self.combo_scenario.phases[phase_index - 1].required_combo
                if phase_index > 0
                else 0
            )
            for alternate_id, alternate_reward in future_phase.alternate_actions:
                if alternate_id not in expected_alternates:
                    expected_alternates[alternate_id] = alternate_reward
                    alternate_min_combo[alternate_id] = previous_required + 1
                    alternate_hit_timeouts[alternate_id] = (
                        SEVENTY_FIVE_KAI_FINISHER_HIT_TIMEOUTS.get(
                            alternate_id,
                            future_phase.hit_timeout_frames
                            or COMBO_PROFILE.action_hit_timeout,
                        )
                    )
        if (
            not input_ready_before_step
            and queueable_followup_phase is not None
            and action_id == queueable_followup_phase.action_id
            and action_accepted
            and (
                action_status.queued_action_id == action_id
                or action_status.active_action_id == action_id
                or action_status.last_started_action_id == action_id
            )
        ):
            self.pending_followups[action_id] = 0
            if previous is not None:
                self.pending_action_power_stocks = max(
                    0,
                    previous.p1_advanced_power_stocks,
                )
        elif (
            not input_ready_before_step
            and action_id in expected_alternates
            and action_accepted
            and (
                action_status.queued_action_id == action_id
                or action_status.active_action_id == action_id
                or action_status.last_started_action_id == action_id
            )
        ):
            self.pending_alternate_action = action_id
            self.pending_alternate_reward = expected_alternates[action_id]
            self.pending_alternate_age = 0
            self.pending_alternate_min_combo = alternate_min_combo[action_id]
            self.pending_alternate_power_stocks = (
                max(0, previous.p1_advanced_power_stocks)
                if previous is not None
                else None
            )
            self.pending_alternate_hit_timeout = alternate_hit_timeouts[action_id]
        self.episode_steps += 1

        if self.pending_chain_action is not None:
            self.pending_action_age += COMBO_PROFILE.action_repeat
        for pending_followup_action in self.pending_followups:
            self.pending_followups[pending_followup_action] += COMBO_PROFILE.action_repeat
        if self.pending_alternate_action is not None:
            self.pending_alternate_age += COMBO_PROFILE.action_repeat

        p1_damage = 0.0
        p2_damage = 0.0
        if previous is not None:
            if previous.p1_health >= 0 and observation.p1_health >= 0:
                p1_damage = float(max(0, previous.p1_health - observation.p1_health))
            if previous.p2_health >= 0 and observation.p2_health >= 0:
                p2_damage = float(max(0, previous.p2_health - observation.p2_health))

        phase_before_reward = self.combo_phase
        reward, reward_parts, combo_success, alternate_success = self._combo_reward(
            previous,
            observation,
            p1_damage,
            p2_damage,
            action_status,
        )
        wrong_action = (
            action_accepted
            and action_id != IDLE_ACTION_ID
            and action_id not in expected_alternates
            and (
                expected_phase_before_step is None
                or action_id != expected_phase_before_step.action_id
            )
        )
        reward_parts["wrong_action"] = 0.0
        if wrong_action:
            reward_parts["wrong_action"] -= COMBO_WRONG_ACTION_PENALTY
            reward -= COMBO_WRONG_ACTION_PENALTY
        phase_advanced = self.combo_phase > phase_before_reward
        advanced_phase = self._combo_phase_at(phase_before_reward)
        advanced_action = advanced_phase.action_id if advanced_phase is not None else None
        p1_ko = 0 <= observation.p1_health <= 0
        p2_ko = 0 <= observation.p2_health <= 0
        round_finished = observation.round_time == 0
        terminated = combo_success or alternate_success or p1_ko or p2_ko or round_finished
        truncated = not terminated and self.episode_steps >= COMBO_PROFILE.max_episode_steps

        self.previous_observation = observation
        info = {
            "raw": observation,
            "training_profile": self.training_profile.value,
            "combo_scenario": self.combo_scenario.name,
            "action_mask_level": self.action_mask_level.value,
            "action": action_id,
            "frame_count": float(COMBO_PROFILE.action_repeat),
            "p1_health": observation.p1_health,
            "p2_health": observation.p2_health,
            "p1_damage": p1_damage,
            "p2_damage": p2_damage,
            "p1_combo_count": float(max(0, observation.p1_combo_count)),
            "p2_combo_count": float(max(0, observation.p2_combo_count)),
            "episode_max_combo": float(self.episode_max_combo),
            "combo_phase": float(self.combo_phase),
            "combo_success": float(combo_success),
            "combo_alternate_success": float(alternate_success),
            "combo_alternate_action": float(
                self.pending_alternate_action
                if self.pending_alternate_action is not None
                else -1
            ),
            "input_ready": float(self._combo_input_ready()),
            "input_accepted": float(action_accepted),
            "pending_chain_action": float(self.pending_chain_action if self.pending_chain_action is not None else -1),
            "pending_followup_count": float(len(self.pending_followups)),
            "phase_wait_remaining": float(self.phase_wait_remaining),
            "active_action_id": float(action_status.active_action_id),
            "queued_action_id": float(action_status.queued_action_id),
            "last_started_action_id": float(action_status.last_started_action_id),
            "action_14": float(action_id == ARAGAMI_ACTION_ID and action_accepted),
            "action_14_hit": float(phase_advanced and advanced_action == ARAGAMI_ACTION_ID),
            "action_15": float(action_id == KOTOTSUKI_YOU_ACTION_ID and action_accepted),
            "action_15_hit": float(phase_advanced and advanced_action == KOTOTSUKI_YOU_ACTION_ID),
            "action_17": float(action_id == RED_KICK_ACTION_ID and action_accepted),
            "action_17_hit": float(phase_advanced and advanced_action == RED_KICK_ACTION_ID),
            "action_18": float(action_id == OROCHINAGI_ACTION_ID and action_accepted),
            "action_18_hit": float(phase_advanced and advanced_action == OROCHINAGI_ACTION_ID),
            "action_19": float(action_id == MUSHIKI_ACTION_ID and action_accepted),
            "action_19_hit": float(phase_advanced and advanced_action == MUSHIKI_ACTION_ID),
            "action_22": float(action_id == FORWARD_B_ACTION_ID and action_accepted),
            "action_22_hit": float(phase_advanced and advanced_action == FORWARD_B_ACTION_ID),
            "action_23": float(action_id == POISON_BITE_ACTION_ID and action_accepted),
            "action_23_hit": float(phase_advanced and advanced_action == POISON_BITE_ACTION_ID),
            "action_24": float(action_id == TSUMI_YOMI_ACTION_ID and action_accepted),
            "action_24_hit": float(phase_advanced and advanced_action == TSUMI_YOMI_ACTION_ID),
            "action_25": float(action_id == BATSU_YOMI_ACTION_ID and action_accepted),
            "action_25_hit": float(phase_advanced and advanced_action == BATSU_YOMI_ACTION_ID),
            "action_26": float(action_id == SEVENTY_FIVE_SHIKI_KAI_ACTION_ID and action_accepted),
            "action_26_hit": float(phase_advanced and advanced_action == SEVENTY_FIVE_SHIKI_KAI_ACTION_ID),
            "action_27": float(action_id == YANO_SABI_ACTION_ID and action_accepted),
            "action_27_hit": float(phase_advanced and advanced_action == YANO_SABI_ACTION_ID),
            "action_28": float(action_id == MIGIRI_UGACHI_ACTION_ID and action_accepted),
            "action_28_hit": float(phase_advanced and advanced_action == MIGIRI_UGACHI_ACTION_ID),
            "reward_damage": reward_parts["damage"],
            "reward_phase": reward_parts["phase"],
            "reward_complete": reward_parts["complete"],
            "reward_alternate": reward_parts["alternate"],
            "reward_phase_reset": reward_parts["phase_reset"],
            "reward_timeout": reward_parts["timeout"],
            "reward_ko_without_combo": reward_parts["ko_without_combo"],
            "reward_time": reward_parts["time"],
            "reward_wrong_action": reward_parts["wrong_action"],
        }
        return self._make_observation(observation), reward, terminated, truncated, info

    def _step_fight(self, action_id: int):
        previous = self.previous_observation
        dll_input_ready = bool(self.client.input_ready())
        free_decision = dll_input_ready and bool(self.client.p1_ready_for_action())
        action_availability = self._physical_action_mask()
        legal_action_count = int(action_availability.sum())
        queue_decision = not free_decision and legal_action_count > 1
        forced_idle = legal_action_count <= 1
        observation = self.client.step(action_id, self.action_repeat)
        action_status = self.client.action_status()
        step_events = self.client.step_events()
        action_accepted = bool(action_status.action_accepted)
        hitbox_rects: list[HitboxRectResult] = []
        if self.hitbox_reward:
            try:
                hitbox_rects, _axes = self.client.get_hitbox_overlay(
                    HITBOX_REWARD_SOURCE_WIDTH,
                    HITBOX_REWARD_SOURCE_HEIGHT,
                )
            except RuntimeError:
                hitbox_rects = []

        p1_attack_overlap = has_attack_hurtbox_overlap(hitbox_rects, HITBOX_OWNER_P1, HITBOX_OWNER_P2)
        p2_attack_overlap = has_attack_hurtbox_overlap(hitbox_rects, HITBOX_OWNER_P2, HITBOX_OWNER_P1)
        p2_attack_pressure = has_attack_hurtbox_overlap(
            hitbox_rects,
            HITBOX_OWNER_P2,
            HITBOX_OWNER_P1,
            DEFENSE_PRESSURE_MARGIN,
        )
        p1_attack_overlap_edge = p1_attack_overlap and not self.fight_prev_p1_attack_overlap
        p2_attack_overlap_edge = p2_attack_overlap and not self.fight_prev_p2_attack_overlap
        self.fight_prev_p1_attack_overlap = p1_attack_overlap
        self.fight_prev_p2_attack_overlap = p2_attack_overlap

        p2_damage = 0.0
        p1_damage = 0.0
        if previous is not None:
            if previous.p2_health >= 0 and observation.p2_health >= 0:
                p2_damage = float(max(0, previous.p2_health - observation.p2_health))
            if previous.p1_health >= 0 and observation.p1_health >= 0:
                p1_damage = float(max(0, previous.p1_health - observation.p1_health))

        previous_combo = max(0, previous.p1_combo_count) if previous is not None else 0
        current_combo = max(0, observation.p1_combo_count)
        combo_delta = max(0, current_combo - previous_combo)

        # A follow-up is an action issued while the DLL runtime is busy that it
        # accepted into its queue (or that already fired within this step's
        # frames). Gate on DLL-level input_ready: setAction only accepts a
        # non-idle action while busy through canQueueAction, so acceptance
        # here proves a genuine queue rather than a raw start.
        queued_followup = (
            not dll_input_ready
            and action_accepted
            and action_id != IDLE_ACTION_ID
            and (
                action_status.queued_action_id == action_id
                or action_status.active_action_id == action_id
                or action_status.last_started_action_id == action_id
            )
        )
        queued_followup_actions: list[int] = []
        started_followup_actions: list[int] = []
        hit_followup_actions: list[int] = []
        if queued_followup:
            self.fight_pending_followups.append(
                {
                    "action": action_id,
                    "age": 0,
                    "started": False,
                    "serial": None,
                },
            )
            queued_followup_actions.append(action_id)

        for pending in self.fight_pending_followups:
            pending["age"] += self.action_repeat

        for event in step_events:
            if event.event_type == STEP_EVENT_ACTION_STARTED:
                for pending in self.fight_pending_followups:
                    if pending["started"] or pending["action"] != event.action_id:
                        continue

                    pending["started"] = True
                    pending["serial"] = int(event.action_serial)
                    started_followup_actions.append(pending["action"])
                    break
            elif event.event_type == STEP_EVENT_COMBO_HIT:
                for pending_index, pending in enumerate(self.fight_pending_followups):
                    if (
                        not pending["started"]
                        or pending["action"] != event.action_id
                        or pending["serial"] != int(event.action_serial)
                        or event.combo_after <= event.combo_before
                        or event.combo_after < 2
                    ):
                        continue

                    hit_followup_actions.append(pending["action"])
                    del self.fight_pending_followups[pending_index]
                    break

        self.fight_pending_followups = [
            pending
            for pending in self.fight_pending_followups
            if pending["age"] <= FIGHT_FOLLOWUP_TRACK_WINDOW_FRAMES
        ]

        hit_action_ids = {
            int(event.action_id)
            for event in step_events
            if event.event_type in (STEP_EVENT_COMBO_HIT, STEP_EVENT_DAMAGE_ONLY)
            and event.action_id >= 0
        }
        combo_hit_action_ids = {
            int(event.action_id)
            for event in step_events
            if event.event_type == STEP_EVENT_COMBO_HIT
            and event.action_id >= 0
        }
        damage_action_id = -1
        for event in step_events:
            if (
                event.event_type in (STEP_EVENT_COMBO_HIT, STEP_EVENT_DAMAGE_ONLY)
                and event.p2_hp_delta > 0
            ):
                damage_action_id = int(event.action_id)

        followup_action = -1
        for actions in (
            hit_followup_actions,
            started_followup_actions,
            queued_followup_actions,
        ):
            if actions:
                followup_action = actions[0]
                break
        if followup_action < 0 and self.fight_pending_followups:
            followup_action = self.fight_pending_followups[0]["action"]

        started_followup = bool(started_followup_actions)
        followup_hit = bool(hit_followup_actions)

        guard_action = action_id in GUARD_ACTION_IDS
        guard_success = guard_action and p2_attack_pressure and p1_damage <= 0.0
        super_action = action_id in SUPER_ACTION_IDS
        super_available_before_action = can_use_super(previous)
        super_without_stock = super_action and not super_available_before_action
        p2_airborne_before_action = is_p2_airborne(previous)
        oniyaki_anti_air_hit = (
            ONIYAKI_ACTION_ID in hit_action_ids
            and p2_airborne_before_action
            and p2_damage > 0.0
        )
        reward, reward_parts = self._reward(
            previous,
            observation,
            action_id,
            damage_action_id,
            combo_hit_action_ids,
            p1_damage,
            p1_attack_overlap_edge,
            p2_attack_overlap_edge,
            p2_attack_pressure,
            super_available_before_action,
            p2_airborne_before_action,
            self.fight_frame_scale,
        )
        safety_reward, safety_info = self._update_attack_safety(
            action_id,
            observation,
            p1_damage,
            p2_damage,
        )
        reward += safety_reward
        reward_parts["safety"] = safety_reward
        reward_parts["cancel"] = 0.0
        combo_4plus_milestone = (
            not self.fight_combo_4plus_rewarded
            and previous_combo < FIGHT_COMBO_4PLUS_MILESTONE_HITS
            and current_combo >= FIGHT_COMBO_4PLUS_MILESTONE_HITS
        )
        reward_parts["combo_4plus_milestone"] = (
            FIGHT_COMBO_4PLUS_MILESTONE_REWARD
            if combo_4plus_milestone
            else 0.0
        )
        if combo_4plus_milestone:
            self.fight_combo_4plus_rewarded = True
            reward += reward_parts["combo_4plus_milestone"]
        self.previous_observation = observation

        terminated = (
            0 <= observation.p1_health <= 0
            or 0 <= observation.p2_health <= 0
            or observation.round_time == 0
        )
        truncated = False
        fight_outcome = ""
        if terminated:
            p1_ko = 0 <= observation.p1_health <= 0
            p2_ko = 0 <= observation.p2_health <= 0
            if p1_ko and p2_ko:
                fight_outcome = "draw_ko"
            elif p2_ko:
                fight_outcome = "win_ko"
            elif p1_ko:
                fight_outcome = "loss_ko"
            elif observation.p1_health > observation.p2_health:
                fight_outcome = "win_timeout"
            elif observation.p1_health < observation.p2_health:
                fight_outcome = "loss_timeout"
            else:
                fight_outcome = "draw_timeout"
        info = {
            "raw": observation,
            "training_profile": self.training_profile.value,
            "action": action_id,
            "frame_count": float(self.action_repeat),
            "free_decision": float(free_decision),
            "queue_decision": float(queue_decision),
            "forced_idle": float(forced_idle),
            "legal_action_count": float(legal_action_count),
            "action_availability": action_availability.copy(),
            "queued_followup": float(queued_followup),
            "started_followup": float(started_followup),
            "followup_hit": float(followup_hit),
            "followup_action": float(followup_action),
            "queued_followup_actions": queued_followup_actions,
            "started_followup_actions": started_followup_actions,
            "hit_followup_actions": hit_followup_actions,
            "step_event_count": float(len(step_events)),
            "step_event_dropped": 0.0,
            "damage_action_id": float(damage_action_id),
            "step_hit_action_ids": sorted(hit_action_ids),
            "step_combo_hit_action_ids": sorted(combo_hit_action_ids),
            "fight_outcome": fight_outcome,
            "p1_health": observation.p1_health,
            "p2_health": observation.p2_health,
            "p1_advanced_power_value": float(max(0, observation.p1_advanced_power_value)),
            "p1_advanced_power_stocks": float(max(0, observation.p1_advanced_power_stocks)),
            "p2_advanced_power_value": float(max(0, observation.p2_advanced_power_value)),
            "p2_advanced_power_stocks": float(max(0, observation.p2_advanced_power_stocks)),
            "p1_damage": p1_damage,
            "p2_damage": p2_damage,
            "p1_combo_count": float(max(0, observation.p1_combo_count)),
            "p2_combo_count": float(max(0, observation.p2_combo_count)),
            "p1_attack_overlap": float(p1_attack_overlap),
            "p2_attack_overlap": float(p2_attack_overlap),
            "p2_attack_pressure": float(p2_attack_pressure),
            "guard_action": float(guard_action),
            "guard_success": float(guard_success),
            "super_available": float(super_available_before_action),
            "super_without_stock": float(super_without_stock),
            "p2_airborne": float(p2_airborne_before_action),
            "oniyaki_anti_air_hit": float(oniyaki_anti_air_hit),
            "attack_safety_pending": safety_info["pending"],
            "attack_safety_punished": safety_info["punished"],
            "attack_safety_unsafe_close": safety_info["unsafe_close"],
            "attack_safety_safe": safety_info["safe"],
            "action_14": float(action_id == 14),
            "action_14_hit": float(14 in hit_action_ids),
            "action_15": float(action_id == 15),
            "action_15_hit": float(15 in hit_action_ids),
            "action_16": float(action_id == 16),
            "action_16_hit": float(16 in hit_action_ids),
            "action_17": float(action_id == 17),
            "action_17_hit": float(17 in hit_action_ids),
            "action_18": float(action_id == 18),
            "action_18_hit": float(18 in hit_action_ids),
            "action_19": float(action_id == 19),
            "action_19_hit": float(19 in hit_action_ids),
            "action_20": float(action_id == 20),
            "action_20_hit": float(20 in hit_action_ids),
            "action_21": float(action_id == 21),
            "action_21_hit": float(21 in hit_action_ids),
            "action_22": float(action_id == 22),
            "action_22_hit": float(22 in hit_action_ids),
            "action_23": float(action_id == 23),
            "action_23_hit": float(23 in hit_action_ids),
            "action_24": float(action_id == 24),
            "action_24_hit": float(24 in hit_action_ids),
            "action_25": float(action_id == 25),
            "action_25_hit": float(25 in hit_action_ids),
            "action_26": float(action_id == SEVENTY_FIVE_SHIKI_KAI_ACTION_ID),
            "action_26_hit": float(SEVENTY_FIVE_SHIKI_KAI_ACTION_ID in hit_action_ids),
            "action_27": float(action_id == YANO_SABI_ACTION_ID),
            "action_27_hit": float(YANO_SABI_ACTION_ID in hit_action_ids),
            "action_28": float(action_id == MIGIRI_UGACHI_ACTION_ID),
            "action_28_hit": float(MIGIRI_UGACHI_ACTION_ID in hit_action_ids),
            "distance_x_abs": float(abs(observation.distance_x)),
            "reward_hp": reward_parts["hp"],
            "reward_hitbox": reward_parts["hitbox"],
            "reward_distance": reward_parts["distance"],
            "reward_defense": reward_parts["defense"],
            "reward_combo": reward_parts["combo"],
            "reward_fight_combo_4plus_milestone": reward_parts["combo_4plus_milestone"],
            "reward_super": reward_parts["super"],
            "reward_anti_air": reward_parts["anti_air"],
            "reward_safety": reward_parts["safety"],
            "reward_cancel": reward_parts["cancel"],
            "reward_fast_win": reward_parts["fast_win"],
            "reward_time": reward_parts["time"],
        }
        return self._make_observation(observation), reward, terminated, truncated, info

    def _combo_reward(
        self,
        previous: Optional[Kof98Observation],
        current: Kof98Observation,
        p1_damage: float,
        p2_damage: float,
        action_status: ActionStatus,
    ) -> tuple[float, dict[str, float], bool, bool]:
        reward_parts = {
            "damage": 0.0,
            "phase": 0.0,
            "complete": 0.0,
            "alternate": 0.0,
            "phase_reset": 0.0,
            "timeout": 0.0,
            "ko_without_combo": 0.0,
            "time": -COMBO_PROFILE.time_penalty,
        }
        if previous is None:
            return 0.0, reward_parts, False, False

        if p2_damage > 0.0:
            reward_parts["damage"] += min(
                p2_damage * COMBO_PROFILE.damage_reward,
                COMBO_PROFILE.damage_reward_cap,
            )

        previous_combo = max(0, previous.p1_combo_count)
        current_combo = max(0, current.p1_combo_count)
        self.episode_max_combo = max(self.episode_max_combo, current_combo)

        phase = self._current_combo_phase()
        expected_action = phase.action_id if phase is not None else None
        phase_hit_timeout = (
            phase.hit_timeout_frames
            if phase is not None and phase.hit_timeout_frames is not None
            else COMBO_PROFILE.action_hit_timeout
        )
        pending_action_hit = (
            phase is not None
            and (p2_damage > 0.0 or not phase.require_damage)
            and self.pending_chain_action == expected_action
            and self.pending_action_age <= phase_hit_timeout
            and (
                not phase.queue_during_previous
                or action_status.active_action_id == expected_action
                or action_status.last_started_action_id == expected_action
                or phase.allow_started_followup_on_hit
            )
        )

        if phase is not None and phase.require_combo_increment:
            continuity_valid = (
                previous_combo >= max(0, phase.required_combo - 1)
                and current_combo >= phase.required_combo
                and current_combo > previous_combo
            )
        else:
            continuity_valid = (
                phase is not None
                and current_combo >= phase.required_combo
            )

        phase_advanced = pending_action_hit and continuity_valid
        if phase is not None and phase.require_power_stock_spent:
            phase_advanced = (
                phase_advanced
                and self.pending_action_power_stocks is not None
                and 0 <= current.p1_advanced_power_stocks
                < self.pending_action_power_stocks
            )
        if phase_advanced:
            reward_parts["phase"] += phase.reward
            self.combo_phase += 1
            self.phase_wait_remaining = phase.wait_after_hit_frames
            next_phase = self._current_combo_phase()
            followup_age = (
                self.pending_followups.get(next_phase.action_id)
                if next_phase is not None
                else None
            )
            followup_is_queued_or_started = (
                next_phase is not None
                and next_phase.queue_during_previous
                and followup_age is not None
                and (
                    action_status.queued_action_id == next_phase.action_id
                    or action_status.active_action_id == next_phase.action_id
                    or action_status.last_started_action_id == next_phase.action_id
                )
            )
            if followup_is_queued_or_started:
                self.pending_chain_action = next_phase.action_id
                self.pending_action_age = followup_age
                del self.pending_followups[next_phase.action_id]
            else:
                self.pending_chain_action = None
                self.pending_action_age = 0
                self.pending_action_power_stocks = None
            remaining_phases = self.combo_scenario.phases[self.combo_phase:]
            remaining_actions = {phase.action_id for phase in remaining_phases}
            self.pending_followups = {
                pending_action: pending_age
                for pending_action, pending_age in self.pending_followups.items()
                if pending_action in remaining_actions
            }
            remaining_alternates = {
                alternate_id
                for phase in remaining_phases
                for alternate_id, _ in phase.alternate_actions
            }
            if self.pending_alternate_action not in remaining_alternates:
                self.pending_alternate_action = None
                self.pending_alternate_reward = 0.0
                self.pending_alternate_age = 0
                self.pending_alternate_min_combo = 0
                self.pending_alternate_power_stocks = None
                self.pending_alternate_hit_timeout = COMBO_PROFILE.action_hit_timeout
            self.frames_since_chain_hit = 0
        elif self.combo_phase > 0:
            self.frames_since_chain_hit += COMBO_PROFILE.action_repeat

        combo_success = self.combo_phase >= len(self.combo_scenario.phases)
        if combo_success:
            reward_parts["complete"] += self.combo_scenario.complete_reward
            return float(sum(reward_parts.values())), reward_parts, True, False

        alternate_power_valid = (
            self.pending_alternate_action not in SUPER_ACTION_IDS
            or (
                self.pending_alternate_power_stocks is not None
                and 0 <= current.p1_advanced_power_stocks
                < self.pending_alternate_power_stocks
            )
        )
        alternate_hit = (
            not phase_advanced
            and self.pending_alternate_action is not None
            and current_combo > previous_combo
            and current_combo >= self.pending_alternate_min_combo
            and self.pending_alternate_age <= self.pending_alternate_hit_timeout
            and alternate_power_valid
            # The queued sibling must actually have fired; otherwise hits from
            # the still-active opener would bank the alternate reward early.
            and (
                action_status.active_action_id == self.pending_alternate_action
                or action_status.last_started_action_id == self.pending_alternate_action
            )
        )
        if alternate_hit:
            reward_parts["alternate"] += self.pending_alternate_reward
            return float(sum(reward_parts.values())), reward_parts, False, True

        p2_ko = previous.p2_health > 0 and 0 <= current.p2_health <= 0
        if p2_ko:
            reward_parts["ko_without_combo"] -= COMBO_PROFILE.ko_without_combo_penalty

        chain_expired = False
        if self.combo_phase > 0 and p1_damage > 0.0:
            chain_expired = True
        if (
            self.combo_phase > 0
            and self.frames_since_chain_hit > COMBO_PROFILE.chain_timeout
        ):
            chain_expired = True
        if (
            self.combo_phase > 0
            and previous_combo > 0
            and current_combo == 0
        ):
            chain_expired = True
        if (
            self.pending_chain_action is not None
            and self.pending_action_age > phase_hit_timeout
        ):
            self.pending_chain_action = None
            self.pending_action_age = 0
            self.pending_action_power_stocks = None
        if (
            self.pending_alternate_action is not None
            and self.pending_alternate_age > self.pending_alternate_hit_timeout
        ):
            self.pending_alternate_action = None
            self.pending_alternate_reward = 0.0
            self.pending_alternate_age = 0
            self.pending_alternate_min_combo = 0
            self.pending_alternate_power_stocks = None
            self.pending_alternate_hit_timeout = COMBO_PROFILE.action_hit_timeout

        if chain_expired:
            reward_parts["phase_reset"] -= COMBO_PROFILE.phase_reset_penalty
            self.combo_phase = 0
            self.pending_chain_action = None
            self.pending_action_age = 0
            self.pending_action_power_stocks = None
            self.pending_followups = {}
            self.pending_alternate_action = None
            self.pending_alternate_reward = 0.0
            self.pending_alternate_age = 0
            self.pending_alternate_min_combo = 0
            self.pending_alternate_power_stocks = None
            self.pending_alternate_hit_timeout = COMBO_PROFILE.action_hit_timeout
            self.frames_since_chain_hit = 0
            self.phase_wait_remaining = 0

        timeout = self.episode_steps >= COMBO_PROFILE.max_episode_steps
        if timeout:
            reward_parts["timeout"] -= COMBO_PROFILE.episode_timeout_penalty

        return float(sum(reward_parts.values())), reward_parts, combo_success, False

    def _track_pending_chain_action(
        self,
        action_id: int,
        input_ready: bool,
        action_accepted: bool,
    ) -> None:
        if not input_ready or not action_accepted:
            return

        phase = self._current_combo_phase()
        expected_action = phase.action_id if phase is not None else None
        alternate_rewards = dict(phase.alternate_actions) if phase is not None else {}
        if action_id == expected_action:
            self.pending_chain_action = action_id
            self.pending_action_age = 0
            if self.previous_observation is not None:
                self.pending_action_power_stocks = max(
                    0,
                    self.previous_observation.p1_advanced_power_stocks,
                )
            self.pending_alternate_action = None
            self.pending_alternate_reward = 0.0
            self.pending_alternate_age = 0
            self.pending_alternate_min_combo = 0
            self.pending_alternate_power_stocks = None
            self.pending_alternate_hit_timeout = COMBO_PROFILE.action_hit_timeout
        elif action_id in alternate_rewards:
            self.pending_alternate_action = action_id
            self.pending_alternate_reward = alternate_rewards[action_id]
            self.pending_alternate_age = 0
            self.pending_alternate_min_combo = (
                self.combo_scenario.phases[self.combo_phase - 1].required_combo + 1
                if self.combo_phase > 0
                else 1
            )
            self.pending_alternate_power_stocks = (
                max(0, self.previous_observation.p1_advanced_power_stocks)
                if self.previous_observation is not None
                else None
            )
            self.pending_alternate_hit_timeout = (
                SEVENTY_FIVE_KAI_FINISHER_HIT_TIMEOUTS.get(
                    action_id,
                    phase.hit_timeout_frames or COMBO_PROFILE.action_hit_timeout,
                )
            )
            self.pending_chain_action = None
            self.pending_action_age = 0
            self.pending_action_power_stocks = None
        elif action_id != IDLE_ACTION_ID:
            self.pending_chain_action = None
            self.pending_action_age = 0
            self.pending_action_power_stocks = None
            self.pending_alternate_action = None
            self.pending_alternate_reward = 0.0
            self.pending_alternate_age = 0
            self.pending_alternate_min_combo = 0
            self.pending_alternate_power_stocks = None
            self.pending_alternate_hit_timeout = COMBO_PROFILE.action_hit_timeout

    def _combo_input_ready(self) -> bool:
        if (
            not self.client.input_ready()
            or not self.client.p1_ready_for_action()
            or self.phase_wait_remaining > 0
        ):
            return False
        return self.pending_chain_action is None

    def _make_observation(self, observation: Kof98Observation) -> np.ndarray:
        input_ready = self.client.input_ready() and self.client.p1_ready_for_action()
        if self.training_profile is TrainingProfile.COMBO:
            input_ready = self._combo_input_ready()
        normalized_phase = float(self.combo_phase) / float(
            max(1, len(self.combo_scenario.phases))
        )
        phase_age = min(
            float(self.frames_since_chain_hit) / COMBO_PHASE_AGE_SCALE_FRAMES,
            1.0,
        )
        return np.concatenate(
            [
                observation_to_vector(observation),
                np.array(
                    [float(input_ready), normalized_phase, phase_age],
                    dtype=np.float32,
                ),
            ],
        )

    def _physical_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[IDLE_ACTION_ID] = True
        super_available = can_use_super(self.previous_observation)
        if self.client.input_ready() and self.client.p1_ready_for_action():
            mask[:] = True
            if not super_available:
                for action_id in SUPER_ACTION_IDS:
                    mask[action_id] = False
            return mask

        for action_id in range(1, self.action_space.n):
            if action_id in SUPER_ACTION_IDS and not super_available:
                continue
            if self.client.can_queue_action(action_id):
                mask[action_id] = True
        return mask

    def _add_guided_distractors(
        self,
        mask: np.ndarray,
        physical_mask: np.ndarray,
    ) -> None:
        candidates = [
            action_id
            for action_id in range(1, self.action_space.n)
            if physical_mask[action_id] and not mask[action_id]
        ]
        if not candidates:
            return

        offset = self.combo_phase % len(candidates)
        ordered = candidates[offset:] + candidates[:offset]
        for action_id in ordered[:GUIDED_DISTRACTOR_COUNT]:
            mask[action_id] = True

    def action_masks(self) -> np.ndarray:
        physical_mask = self._physical_action_mask()
        if self.training_profile is not TrainingProfile.COMBO:
            return physical_mask
        if self.action_mask_level is ActionMaskLevel.PHYSICAL:
            return physical_mask

        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[IDLE_ACTION_ID] = True
        queueable_followup_phase = self._queueable_followup_phase()
        if queueable_followup_phase is not None:
            mask[queueable_followup_phase.action_id] = True
            if self.action_mask_level is ActionMaskLevel.GUIDED:
                self._add_guided_distractors(mask, physical_mask)
            return mask

        if not self._combo_input_ready():
            return mask

        if (
            self.combo_phase == 0
            and self.previous_observation is not None
            and abs(self.previous_observation.distance_x) > COMBO_CLOSE_DISTANCE
        ):
            mask[IDLE_ACTION_ID] = False
            mask[FORWARD_ACTION_ID] = True
            if self.action_mask_level is ActionMaskLevel.GUIDED:
                self._add_guided_distractors(mask, physical_mask)
            return mask

        phase = self._current_combo_phase()
        if phase is not None:
            mask[phase.action_id] = True
        if self.action_mask_level is ActionMaskLevel.GUIDED:
            self._add_guided_distractors(mask, physical_mask)
        return mask

    def close(self):
        self.client.close()

    def _update_attack_safety(
        self,
        action_id: int,
        current: Kof98Observation,
        p1_damage: float,
        p2_damage: float,
    ) -> tuple[float, dict[str, float]]:
        info = {
            "pending": 0.0,
            "punished": 0.0,
            "unsafe_close": 0.0,
            "safe": 0.0,
        }
        reward = 0.0

        if self.pending_attack_risk is not None:
            self.pending_attack_risk["frames_left"] -= float(self.action_repeat)
            self.pending_attack_risk["p2_damage"] += p2_damage

            close_to_p2 = (
                current.p1_has_position
                and current.p2_has_position
                and abs(current.distance_x) < ATTACK_RISK_CLOSE_DISTANCE
            )
            attack_has_hit = self.pending_attack_risk["p2_damage"] > 0.0

            if attack_has_hit:
                self.pending_attack_risk = None
            elif p1_damage > 0.0:
                reward -= ATTACK_RISK_PUNISH_PENALTY
                info["punished"] = 1.0
                self.pending_attack_risk = None
            elif self.pending_attack_risk["frames_left"] <= 0.0:
                if close_to_p2:
                    reward -= ATTACK_RISK_UNSAFE_CLOSE_PENALTY
                    info["unsafe_close"] = 1.0
                else:
                    reward += ATTACK_RISK_SAFE_REWARD
                    info["safe"] = 1.0
                self.pending_attack_risk = None

        if (
            action_id in ATTACK_RISK_ACTION_IDS
            and p1_damage <= 0.0
            and p2_damage <= 0.0
            and self.pending_attack_risk is None
        ):
            self.pending_attack_risk = {
                "action_id": float(action_id),
                "frames_left": float(ATTACK_RISK_WINDOW_FRAMES),
                "p2_damage": 0.0,
            }

        info["pending"] = float(self.pending_attack_risk is not None)
        return reward, info

    @staticmethod
    def _reward(
        previous: Optional[Kof98Observation],
        current: Kof98Observation,
        action_id: int,
        damage_action_id: int,
        combo_hit_action_ids: set[int],
        p1_damage: float,
        p1_attack_overlap_edge: bool,
        p2_attack_overlap_edge: bool,
        p2_attack_pressure: bool,
        super_available: bool,
        p2_airborne: bool,
        frame_scale: float = 1.0,
    ) -> tuple[float, dict[str, float]]:
        reward_parts = {
            "hp": 0.0,
            "hitbox": 0.0,
            "distance": 0.0,
            "defense": 0.0,
            "combo": 0.0,
            "super": 0.0,
            "anti_air": 0.0,
            "safety": 0.0,
            "fast_win": 0.0,
            "time": -0.001 * frame_scale,
        }
        if previous is None:
            return 0.0, reward_parts

        if previous.p2_health >= 0 and current.p2_health >= 0:
            p2_health_delta = float(previous.p2_health - current.p2_health)
            if damage_action_id != ONIYAKI_ACTION_ID or p2_airborne:
                reward_parts["hp"] += p2_health_delta * 2.0
            if (
                damage_action_id == ONIYAKI_ACTION_ID
                and p2_airborne
                and p2_health_delta > 0.0
            ):
                reward_parts["anti_air"] += ONIYAKI_ANTI_AIR_BONUS
        if previous.p1_health >= 0 and current.p1_health >= 0:
            reward_parts["hp"] -= float(previous.p1_health - current.p1_health) * 2.0

        # Edge events, not per-step accrual, so no frame_scale here.
        if p1_attack_overlap_edge:
            reward_parts["hitbox"] += P1_ATTACK_OVERLAP_REWARD
        if p2_attack_overlap_edge:
            reward_parts["hitbox"] -= P2_ATTACK_OVERLAP_PENALTY

        guard_action = action_id in GUARD_ACTION_IDS
        if p2_attack_pressure:
            if guard_action and p1_damage <= 0.0:
                reward_parts["defense"] += DEFENSE_GUARD_REWARD * frame_scale
            elif guard_action:
                reward_parts["defense"] -= DEFENSE_BAD_GUARD_PENALTY * frame_scale
            else:
                reward_parts["defense"] -= DEFENSE_UNGUARDED_PRESSURE_PENALTY * frame_scale

        if previous.p1_combo_count >= 0 and current.p1_combo_count >= 0:
            previous_combo = max(0, previous.p1_combo_count)
            current_combo = max(0, current.p1_combo_count)
            if current_combo > previous_combo:
                # A multi-hit step credits every newly reached hit number.
                for hit_number in range(previous_combo + 1, current_combo + 1):
                    if hit_number >= 2:
                        reward_parts["combo"] += FIGHT_COMBO_HIT_REWARDS.get(
                            hit_number,
                            FIGHT_COMBO_HIT_REWARD_CAP,
                        )
            if (
                combo_hit_action_ids.intersection(SUPER_ACTION_IDS)
                and current.p1_combo_count >= 2
            ):
                reward_parts["combo"] += SUPER_COMBO_BONUS

        if action_id in SUPER_ACTION_IDS and not super_available:
            reward_parts["super"] -= SUPER_NO_STOCK_PENALTY

        if previous.p2_health > 0 and 0 <= current.p2_health <= 0:
            remaining_time = max(0.0, min(99.0, float(current.round_time)))
            reward_parts["fast_win"] += FAST_WIN_BONUS_MAX * (remaining_time / 99.0)

        if current.p1_has_position and current.p2_has_position:
            distance = abs(current.distance_x)
            if EFFECTIVE_DISTANCE_MIN <= distance <= EFFECTIVE_DISTANCE_MAX:
                reward_parts["distance"] += DISTANCE_IN_RANGE_REWARD * frame_scale
            elif distance > DISTANCE_TOO_FAR_START:
                penalty_t = min(
                    1.0,
                    (distance - DISTANCE_TOO_FAR_START)
                    / max(1, DISTANCE_MAX_PENALTY_AT - DISTANCE_TOO_FAR_START),
                )
                reward_parts["distance"] -= DISTANCE_FAR_PENALTY * penalty_t * frame_scale

        reward = (
            reward_parts["hp"]
            + reward_parts["hitbox"]
            + reward_parts["distance"]
            + reward_parts["defense"]
            + reward_parts["combo"]
            + reward_parts["super"]
            + reward_parts["anti_air"]
            + reward_parts["safety"]
            + reward_parts["fast_win"]
            + reward_parts["time"]
        )
        return reward, reward_parts
