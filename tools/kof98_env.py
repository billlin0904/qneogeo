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
P1_ATTACK_OVERLAP_REWARD = 2.0
P2_ATTACK_OVERLAP_PENALTY = 2.0
EFFECTIVE_DISTANCE_MIN = 35
EFFECTIVE_DISTANCE_MAX = 85
DISTANCE_TOO_FAR_START = 110
DISTANCE_MAX_PENALTY_AT = 180
DISTANCE_IN_RANGE_REWARD = 0.02
DISTANCE_FAR_PENALTY = 0.04
ACTION_COUNT = 27
GUARD_ACTION_IDS = {2, 3, 4}
DEFENSE_PRESSURE_MARGIN = 28
DEFENSE_GUARD_REWARD = 0.08
DEFENSE_UNGUARDED_PRESSURE_PENALTY = 0.04
DEFENSE_BAD_GUARD_PENALTY = 0.08
COMBO_DELTA_REWARD = 1.0
COMBO_LENGTH_REWARD = 0.25
SUPER_COMBO_BONUS = 3.0
SUPER_ACTION_IDS = {18, 19}
SUPER_POWER_STOCKS_REQUIRED = 1
SUPER_NO_STOCK_PENALTY = 0.5
ONIYAKI_ACTION_ID = 16
P2_AIRBORNE_Y_THRESHOLD = 185
ONIYAKI_ANTI_AIR_BONUS = 1.0
ATTACK_RISK_ACTION_IDS = set(range(14, ACTION_COUNT))
ATTACK_RISK_WINDOW_STEPS = 4
ATTACK_RISK_CLOSE_DISTANCE = 55
ATTACK_RISK_PUNISH_PENALTY = 4.0
ATTACK_RISK_UNSAFE_CLOSE_PENALTY = 0.12
ATTACK_RISK_SAFE_REWARD = 0.05
FAST_WIN_BONUS_MAX = 15.0

IDLE_ACTION_ID = 0
FORWARD_ACTION_ID = 1
CLOSE_C_ACTION_ID = 8
ARAGAMI_ACTION_ID = 14
KOTOTSUKI_YOU_ACTION_ID = 15
RED_KICK_ACTION_ID = 17
OROCHINAGI_ACTION_ID = 18
FORWARD_B_ACTION_ID = 22
POISON_BITE_ACTION_ID = 23
TSUMI_YOMI_ACTION_ID = 24
BATSU_YOMI_ACTION_ID = 25
SEVENTY_FIVE_SHIKI_KAI_ACTION_ID = 26

COMBO_CLOSE_DISTANCE = 45
COMBO_PHASE_AGE_SCALE_FRAMES = 30.0


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

COMBO_SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        KYO_CORNER_DOKUGAMI_SCENARIO,
        KYO_FORWARD_B_KOTOTSUKI_SCENARIO,
        KYO_FORWARD_B_OROCHINAGI_SCENARIO,
        KYO_FORWARD_B_RED_KICK_SCENARIO,
        KYO_FORWARD_B_ARAGAMI_SCENARIO,
    )
}
DEFAULT_COMBO_SCENARIO_NAME = KYO_CORNER_DOKUGAMI_SCENARIO.name


class TrainingProfile(str, Enum):
    COMBO = "combo"
    FIGHT = "fight"


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
        self.hitbox_reward = hitbox_reward and self.training_profile is TrainingProfile.FIGHT
        self.p2_training_ai = p2_training_ai
        self.previous_observation: Optional[Kof98Observation] = None
        self.pending_attack_risk: Optional[dict[str, float]] = None
        self.episode_steps = 0
        self.episode_max_combo = 0
        self.combo_phase = 0
        self.pending_chain_action: Optional[int] = None
        self.pending_action_age = 0
        self.pending_action_power_stocks: Optional[int] = None
        self.pending_followup_action: Optional[int] = None
        self.pending_followup_age = 0
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
        self.episode_steps = 0
        self.episode_max_combo = 0
        self.combo_phase = 0
        self.pending_chain_action = None
        self.pending_action_age = 0
        self.pending_action_power_stocks = None
        self.pending_followup_action = None
        self.pending_followup_age = 0
        self.frames_since_chain_hit = 0
        self.phase_wait_remaining = 0
        return self._make_observation(observation), {
            "raw": observation,
            "training_profile": self.training_profile.value,
            "combo_scenario": self.combo_scenario.name,
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
        for phase_index in (self.combo_phase, self.combo_phase + 1):
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
        observation = self.client.step(action_id, COMBO_PROFILE.action_repeat)
        action_status = self.client.action_status()
        action_accepted = bool(action_status.action_accepted)
        self._track_pending_chain_action(
            action_id,
            input_ready_before_step,
            action_accepted,
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
            self.pending_followup_action = action_id
            self.pending_followup_age = 0
            if previous is not None:
                self.pending_action_power_stocks = max(
                    0,
                    previous.p1_advanced_power_stocks,
                )
        self.episode_steps += 1

        if self.pending_chain_action is not None:
            self.pending_action_age += COMBO_PROFILE.action_repeat
        if self.pending_followup_action is not None:
            self.pending_followup_age += COMBO_PROFILE.action_repeat

        p1_damage = 0.0
        p2_damage = 0.0
        if previous is not None:
            if previous.p1_health >= 0 and observation.p1_health >= 0:
                p1_damage = float(max(0, previous.p1_health - observation.p1_health))
            if previous.p2_health >= 0 and observation.p2_health >= 0:
                p2_damage = float(max(0, previous.p2_health - observation.p2_health))

        phase_before_reward = self.combo_phase
        reward, reward_parts, combo_success = self._combo_reward(
            previous,
            observation,
            p1_damage,
            p2_damage,
            action_status,
        )
        phase_advanced = self.combo_phase > phase_before_reward
        advanced_phase = self._combo_phase_at(phase_before_reward)
        advanced_action = advanced_phase.action_id if advanced_phase is not None else None
        p1_ko = 0 <= observation.p1_health <= 0
        p2_ko = 0 <= observation.p2_health <= 0
        round_finished = observation.round_time == 0
        terminated = combo_success or p1_ko or p2_ko or round_finished
        truncated = not terminated and self.episode_steps >= COMBO_PROFILE.max_episode_steps

        self.previous_observation = observation
        info = {
            "raw": observation,
            "training_profile": self.training_profile.value,
            "combo_scenario": self.combo_scenario.name,
            "action": action_id,
            "p1_health": observation.p1_health,
            "p2_health": observation.p2_health,
            "p1_damage": p1_damage,
            "p2_damage": p2_damage,
            "p1_combo_count": float(max(0, observation.p1_combo_count)),
            "p2_combo_count": float(max(0, observation.p2_combo_count)),
            "episode_max_combo": float(self.episode_max_combo),
            "combo_phase": float(self.combo_phase),
            "combo_success": float(combo_success),
            "input_ready": float(self._combo_input_ready()),
            "input_accepted": float(action_accepted),
            "pending_chain_action": float(self.pending_chain_action if self.pending_chain_action is not None else -1),
            "pending_followup_action": float(self.pending_followup_action if self.pending_followup_action is not None else -1),
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
            "reward_damage": reward_parts["damage"],
            "reward_phase": reward_parts["phase"],
            "reward_complete": reward_parts["complete"],
            "reward_phase_reset": reward_parts["phase_reset"],
            "reward_timeout": reward_parts["timeout"],
            "reward_ko_without_combo": reward_parts["ko_without_combo"],
            "reward_time": reward_parts["time"],
        }
        return self._make_observation(observation), reward, terminated, truncated, info

    def _step_fight(self, action_id: int):
        previous = self.previous_observation
        observation = self.client.step(action_id, self.action_repeat)
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

        p2_damage = 0.0
        p1_damage = 0.0
        if previous is not None:
            if previous.p2_health >= 0 and observation.p2_health >= 0:
                p2_damage = float(max(0, previous.p2_health - observation.p2_health))
            if previous.p1_health >= 0 and observation.p1_health >= 0:
                p1_damage = float(max(0, previous.p1_health - observation.p1_health))

        guard_action = action_id in GUARD_ACTION_IDS
        guard_success = guard_action and p2_attack_pressure and p1_damage <= 0.0
        super_action = action_id in SUPER_ACTION_IDS
        super_available_before_action = can_use_super(previous)
        super_without_stock = super_action and not super_available_before_action
        p2_airborne_before_action = is_p2_airborne(previous)
        oniyaki_anti_air_hit = (
            action_id == ONIYAKI_ACTION_ID
            and p2_airborne_before_action
            and p2_damage > 0.0
        )
        reward, reward_parts = self._reward(
            previous,
            observation,
            action_id,
            p1_damage,
            p2_damage,
            p1_attack_overlap,
            p2_attack_overlap,
            p2_attack_pressure,
            super_available_before_action,
            p2_airborne_before_action,
        )
        safety_reward, safety_info = self._update_attack_safety(
            action_id,
            observation,
            p1_damage,
            p2_damage,
        )
        reward += safety_reward
        reward_parts["safety"] = safety_reward
        self.previous_observation = observation

        terminated = (
            0 <= observation.p1_health <= 0
            or 0 <= observation.p2_health <= 0
            or observation.round_time == 0
        )
        truncated = False
        info = {
            "raw": observation,
            "training_profile": self.training_profile.value,
            "action": action_id,
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
            "action_14_hit": float(action_id == 14 and p2_damage > 0),
            "action_15": float(action_id == 15),
            "action_15_hit": float(action_id == 15 and p2_damage > 0),
            "action_16": float(action_id == 16),
            "action_16_hit": float(action_id == 16 and p2_damage > 0),
            "action_17": float(action_id == 17),
            "action_17_hit": float(action_id == 17 and p2_damage > 0),
            "action_18": float(action_id == 18),
            "action_18_hit": float(action_id == 18 and p2_damage > 0),
            "action_19": float(action_id == 19),
            "action_19_hit": float(action_id == 19 and p2_damage > 0),
            "action_20": float(action_id == 20),
            "action_20_hit": float(action_id == 20 and p2_damage > 0),
            "action_21": float(action_id == 21),
            "action_21_hit": float(action_id == 21 and p2_damage > 0),
            "action_22": float(action_id == 22),
            "action_22_hit": float(action_id == 22 and p2_damage > 0),
            "action_23": float(action_id == 23),
            "action_23_hit": float(action_id == 23 and p2_damage > 0),
            "action_24": float(action_id == 24),
            "action_24_hit": float(action_id == 24 and p2_damage > 0),
            "action_25": float(action_id == 25),
            "action_25_hit": float(action_id == 25 and p2_damage > 0),
            "action_26": float(action_id == SEVENTY_FIVE_SHIKI_KAI_ACTION_ID),
            "action_26_hit": float(action_id == SEVENTY_FIVE_SHIKI_KAI_ACTION_ID and p2_damage > 0),
            "distance_x_abs": float(abs(observation.distance_x)),
            "reward_hp": reward_parts["hp"],
            "reward_hitbox": reward_parts["hitbox"],
            "reward_distance": reward_parts["distance"],
            "reward_defense": reward_parts["defense"],
            "reward_combo": reward_parts["combo"],
            "reward_super": reward_parts["super"],
            "reward_anti_air": reward_parts["anti_air"],
            "reward_safety": reward_parts["safety"],
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
    ) -> tuple[float, dict[str, float], bool]:
        reward_parts = {
            "damage": 0.0,
            "phase": 0.0,
            "complete": 0.0,
            "phase_reset": 0.0,
            "timeout": 0.0,
            "ko_without_combo": 0.0,
            "time": -COMBO_PROFILE.time_penalty,
        }
        if previous is None:
            return 0.0, reward_parts, False

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
            followup_is_queued_or_started = (
                next_phase is not None
                and next_phase.queue_during_previous
                and self.pending_followup_action == next_phase.action_id
                and (
                    action_status.queued_action_id == next_phase.action_id
                    or action_status.active_action_id == next_phase.action_id
                    or action_status.last_started_action_id == next_phase.action_id
                )
            )
            if followup_is_queued_or_started:
                self.pending_chain_action = next_phase.action_id
                self.pending_action_age = self.pending_followup_age
            else:
                self.pending_chain_action = None
                self.pending_action_age = 0
                self.pending_action_power_stocks = None
            self.pending_followup_action = None
            self.pending_followup_age = 0
            self.frames_since_chain_hit = 0
        elif self.combo_phase > 0:
            self.frames_since_chain_hit += COMBO_PROFILE.action_repeat

        combo_success = self.combo_phase >= len(self.combo_scenario.phases)
        if combo_success:
            reward_parts["complete"] += self.combo_scenario.complete_reward
            return float(sum(reward_parts.values())), reward_parts, True

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

        if chain_expired:
            reward_parts["phase_reset"] -= COMBO_PROFILE.phase_reset_penalty
            self.combo_phase = 0
            self.pending_chain_action = None
            self.pending_action_age = 0
            self.pending_action_power_stocks = None
            self.pending_followup_action = None
            self.pending_followup_age = 0
            self.frames_since_chain_hit = 0
            self.phase_wait_remaining = 0

        timeout = self.episode_steps >= COMBO_PROFILE.max_episode_steps
        if timeout:
            reward_parts["timeout"] -= COMBO_PROFILE.episode_timeout_penalty

        return float(sum(reward_parts.values())), reward_parts, combo_success

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
        if action_id == expected_action:
            self.pending_chain_action = action_id
            self.pending_action_age = 0
            if self.previous_observation is not None:
                self.pending_action_power_stocks = max(
                    0,
                    self.previous_observation.p1_advanced_power_stocks,
                )
        elif action_id != IDLE_ACTION_ID:
            self.pending_chain_action = None
            self.pending_action_age = 0
            self.pending_action_power_stocks = None

    def _combo_input_ready(self) -> bool:
        if (
            not self.client.input_ready()
            or not self.client.p1_ready_for_action()
            or self.phase_wait_remaining > 0
        ):
            return False
        return self.pending_chain_action is None

    def _make_observation(self, observation: Kof98Observation) -> np.ndarray:
        input_ready = self.client.input_ready()
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

    def action_masks(self) -> np.ndarray:
        mask = np.ones(self.action_space.n, dtype=bool)
        if self.training_profile is not TrainingProfile.COMBO:
            return mask

        mask[:] = False
        mask[IDLE_ACTION_ID] = True
        queueable_followup_phase = self._queueable_followup_phase()
        if queueable_followup_phase is not None:
            mask[queueable_followup_phase.action_id] = True
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
            return mask

        phase = self._current_combo_phase()
        if phase is not None:
            mask[phase.action_id] = True
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
            self.pending_attack_risk["steps_left"] -= 1.0
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
            elif self.pending_attack_risk["steps_left"] <= 0.0:
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
                "steps_left": float(ATTACK_RISK_WINDOW_STEPS),
                "p2_damage": 0.0,
            }

        info["pending"] = float(self.pending_attack_risk is not None)
        return reward, info

    @staticmethod
    def _reward(
        previous: Optional[Kof98Observation],
        current: Kof98Observation,
        action_id: int,
        p1_damage: float,
        p2_damage: float,
        p1_attack_overlap: bool,
        p2_attack_overlap: bool,
        p2_attack_pressure: bool,
        super_available: bool,
        p2_airborne: bool,
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
            "time": -0.001,
        }
        if previous is None:
            return 0.0, reward_parts

        if previous.p2_health >= 0 and current.p2_health >= 0:
            p2_health_delta = float(previous.p2_health - current.p2_health)
            if action_id != ONIYAKI_ACTION_ID or p2_airborne:
                reward_parts["hp"] += p2_health_delta * 3.0
            if action_id == ONIYAKI_ACTION_ID and p2_airborne and p2_health_delta > 0.0:
                reward_parts["anti_air"] += ONIYAKI_ANTI_AIR_BONUS
        if previous.p1_health >= 0 and current.p1_health >= 0:
            reward_parts["hp"] -= float(previous.p1_health - current.p1_health) * 2.0

        if p1_attack_overlap:
            reward_parts["hitbox"] += P1_ATTACK_OVERLAP_REWARD
        if p2_attack_overlap:
            reward_parts["hitbox"] -= P2_ATTACK_OVERLAP_PENALTY

        guard_action = action_id in GUARD_ACTION_IDS
        if p2_attack_pressure:
            if guard_action and p1_damage <= 0.0:
                reward_parts["defense"] += DEFENSE_GUARD_REWARD
            elif guard_action:
                reward_parts["defense"] -= DEFENSE_BAD_GUARD_PENALTY
            else:
                reward_parts["defense"] -= DEFENSE_UNGUARDED_PRESSURE_PENALTY

        if previous.p1_combo_count >= 0 and current.p1_combo_count >= 0:
            combo_delta = max(0, current.p1_combo_count - previous.p1_combo_count)
            if combo_delta > 0:
                reward_parts["combo"] += combo_delta * COMBO_DELTA_REWARD
                reward_parts["combo"] += max(0, current.p1_combo_count - 1) * COMBO_LENGTH_REWARD
            if action_id in SUPER_ACTION_IDS and super_available and p2_damage > 0.0 and current.p1_combo_count >= 2:
                reward_parts["combo"] += SUPER_COMBO_BONUS

        if action_id in SUPER_ACTION_IDS and not super_available:
            reward_parts["super"] -= SUPER_NO_STOCK_PENALTY

        if previous.p2_health > 0 and 0 <= current.p2_health <= 0:
            remaining_time = max(0.0, min(99.0, float(current.round_time)))
            reward_parts["fast_win"] += FAST_WIN_BONUS_MAX * (remaining_time / 99.0)

        if current.p1_has_position and current.p2_has_position:
            distance = abs(current.distance_x)
            if EFFECTIVE_DISTANCE_MIN <= distance <= EFFECTIVE_DISTANCE_MAX:
                reward_parts["distance"] += DISTANCE_IN_RANGE_REWARD
            elif distance > DISTANCE_TOO_FAR_START:
                penalty_t = min(
                    1.0,
                    (distance - DISTANCE_TOO_FAR_START)
                    / max(1, DISTANCE_MAX_PENALTY_AT - DISTANCE_TOO_FAR_START),
                )
                reward_parts["distance"] -= DISTANCE_FAR_PENALTY * penalty_t

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
