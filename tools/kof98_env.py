from __future__ import annotations

import ctypes
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
GUARD_ACTION_IDS = {2, 3, 4}
DEFENSE_PRESSURE_MARGIN = 28
DEFENSE_GUARD_REWARD = 0.08
DEFENSE_UNGUARDED_PRESSURE_PENALTY = 0.04
DEFENSE_BAD_GUARD_PENALTY = 0.08
COMBO_DELTA_REWARD = 1.0
COMBO_LENGTH_REWARD = 0.25
SUPER_COMBO_BONUS = 3.0
SUPER_ACTION_IDS = {19, 20}
SUPER_POWER_STOCKS_REQUIRED = 1
SUPER_NO_STOCK_PENALTY = 0.5
ONIYAKI_ACTION_ID = 17
P2_AIRBORNE_Y_THRESHOLD = 185
ONIYAKI_ANTI_AIR_BONUS = 1.0
ATTACK_RISK_ACTION_IDS = set(range(14, 24))
ATTACK_RISK_WINDOW_STEPS = 4
ATTACK_RISK_CLOSE_DISTANCE = 55
ATTACK_RISK_PUNISH_PENALTY = 4.0
ATTACK_RISK_UNSAFE_CLOSE_PENALTY = 0.12
ATTACK_RISK_SAFE_REWARD = 0.05


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

        self.dll.kof_env_p1_ready_for_action.argtypes = [ctypes.c_void_p]
        self.dll.kof_env_p1_ready_for_action.restype = ctypes.c_int

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
    ):
        if gym is None or spaces is None:
            raise RuntimeError("Install gymnasium before using Kof98Env")

        super().__init__()
        self.client = KofEnvClient(dll_path)
        self.client.load_core(core_path)
        self.client.load_game(game_path, system_dir, save_dir)
        self.state_path = Path(state_path) if state_path else None
        self.action_repeat = action_repeat
        self.hitbox_reward = hitbox_reward
        self.previous_observation: Optional[Kof98Observation] = None
        self.pending_attack_risk: Optional[dict[str, float]] = None

        self.action_space = spaces.Discrete(24)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(23,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.state_path:
            self.client.load_state(self.state_path)
        else:
            self.client.reset()

        observation = self.client.observation()
        self.previous_observation = observation
        self.pending_attack_risk = None
        return observation_to_vector(observation), {"raw": observation}

    def step(self, action):
        action_id = int(action)
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
            "distance_x_abs": float(abs(observation.distance_x)),
            "reward_hp": reward_parts["hp"],
            "reward_hitbox": reward_parts["hitbox"],
            "reward_distance": reward_parts["distance"],
            "reward_defense": reward_parts["defense"],
            "reward_combo": reward_parts["combo"],
            "reward_super": reward_parts["super"],
            "reward_anti_air": reward_parts["anti_air"],
            "reward_safety": reward_parts["safety"],
            "reward_time": reward_parts["time"],
        }
        return observation_to_vector(observation), reward, terminated, truncated, info

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
            + reward_parts["time"]
        )
        return reward, reward_parts
