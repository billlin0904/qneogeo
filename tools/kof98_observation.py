"""Versioned KOF98 observation encoding and generic combat state machines.

V1 is the frozen 26-value ABI used by every existing checkpoint. V2 appends
measured state: raw player status bytes, controlled-player action lifecycle,
movement and screen-edge context. Opponent scripted-action internals are
deliberately neutral because they do not exist against a human player.
Existing fields never move, which lets us transplant a V1 policy into V2
without losing its learned behaviour.

V3 keeps the 140-value shape, but replaces V2's two neutral P2 timing scalars
and constant 30-value P2-NONE one-hot with symmetric reaction timing and
generic defense/confirm history. No curriculum or Oracle identity is exposed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Iterable, Optional, Protocol

import numpy as np


ACTION_COUNT = 29
ACTION_ONE_HOT_SIZE = ACTION_COUNT + 1
NONE_ACTION_INDEX = ACTION_COUNT
NATIVE_WIDTH = 320.0
NATIVE_HEIGHT = 224.0
AIRBORNE_Y_THRESHOLD = 185
ACTION_TIME_SCALE_FRAMES = 180.0

OBSERVATION_V1_SIZE = 26
OBSERVATION_V2_SCALAR_SIZE = 18
OBSERVATION_SCHEMA_V1_ID = "kof98-observation-v1-26"
OBSERVATION_SCHEMA_V2_ID = "kof98-observation-v2-140"
OBSERVATION_SCHEMA_V3_ID = "kof98-observation-v3-event-140"

# V3 reuses exactly the V2 fields that were deliberately neutral:
# scalar slots 14/15 and the full opponent-action NONE one-hot.
OBSERVATION_V3_REPURPOSED_INDICES = (
    40,
    41,
    *range(104, 134),
)
REACTION_TIME_SCALE_FRAMES = 60.0
RECOVERY_TIME_SCALE_FRAMES = 60.0
FRAME_ADVANTAGE_SCALE_FRAMES = 30.0
GENERIC_PHASE_AGE_SCALE_FRAMES = 180.0


class ObservationVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


class CombatPhase(IntEnum):
    NEUTRAL = 0
    APPROACH = 1
    CONFIRM = 2
    COMBO = 3
    DEFENSE = 4
    KNOCKDOWN = 5


COMBAT_PHASE_COUNT = len(CombatPhase)
OBSERVATION_V2_SIZE = (
    OBSERVATION_V1_SIZE
    + OBSERVATION_V2_SCALAR_SIZE
    + ACTION_ONE_HOT_SIZE * 3
    + COMBAT_PHASE_COUNT
)
OBSERVATION_V3_SIZE = OBSERVATION_V2_SIZE


class RawObservation(Protocol):
    round_time: int
    p1_health: int
    p2_health: int
    p1_power: int
    p2_power: int
    p1_power_state: int
    p2_power_state: int
    p1_advanced_power_value: int
    p1_advanced_power_stocks: int
    p2_advanced_power_value: int
    p2_advanced_power_stocks: int
    p1_stun: int
    p2_stun: int
    p1_combo_count: int
    p2_combo_count: int
    p1_x: int
    p1_y: int
    p2_x: int
    p2_y: int
    distance_x: int
    distance_y: int
    p1_has_position: int
    p2_has_position: int


class StrategyState(Protocol):
    p1_status: int
    p2_status: int
    p1_active_action_id: int
    p1_queued_action_id: int
    p1_action_elapsed_frames: int
    p1_action_remaining_frames: int
    p2_active_action_id: int
    p2_action_elapsed_frames: int
    p2_action_remaining_frames: int
    p1_ready: int
    p2_ready: int
    p1_facing_left: int
    p2_scripted: int


class PlayerTimingState(Protocol):
    reaction_kind: int
    reaction_remaining: int
    recovery_remaining: int
    reaction_valid: int
    reaction_remaining_valid: int
    actionable_valid: int
    actionable: int
    recovery_valid: int


class CombatTimingState(Protocol):
    frame_advantage: int
    frame_advantage_valid: int
    p1: PlayerTimingState
    p2: PlayerTimingState


class CombatEvent(Protocol):
    event_type: int
    action_id: int
    action_serial: int
    source_player: int
    target_player: int
    absolute_engine_frame: int


class DefensePhase(IntEnum):
    NEUTRAL = 0
    PRESSURE = 1
    BLOCK_CONTACT = 2
    BLOCK_REACTION = 3
    POST_BLOCK = 4


class ConfirmPhase(IntEnum):
    NEUTRAL = 0
    STARTER_PENDING = 1
    STARTER_HIT = 2
    STARTER_BLOCKED = 3
    FOLLOWUP_STARTED = 4


DEFENSE_PHASE_COUNT = len(DefensePhase)
CONFIRM_PHASE_COUNT = len(ConfirmPhase)


@dataclass(frozen=True)
class GenericCombatState:
    defense_phase: DefensePhase = DefensePhase.NEUTRAL
    confirm_phase: ConfirmPhase = ConfirmPhase.NEUTRAL
    defense_phase_age_frames: int = 0
    confirm_phase_age_frames: int = 0
    queue_window_open: bool = False
    last_action_accepted: bool = False


@dataclass(frozen=True)
class ObservationContext:
    input_ready: bool
    normalized_combo_phase: float
    combo_phase_age: float
    action_repeat: int
    combat_phase: CombatPhase = CombatPhase.NEUTRAL
    profile_is_fight: bool = False
    strategy_state: Optional[StrategyState] = None
    timing_state: Optional[CombatTimingState] = None
    generic_combat_state: GenericCombatState = GenericCombatState()
    event_features_enabled: bool = True
    previous_observation: Optional[RawObservation] = None


def observation_size(version: ObservationVersion | str) -> int:
    resolved = ObservationVersion(version)
    return OBSERVATION_V1_SIZE if resolved is ObservationVersion.V1 else OBSERVATION_V2_SIZE


def observation_schema_id(version: ObservationVersion | str) -> str:
    resolved = ObservationVersion(version)
    if resolved is ObservationVersion.V1:
        return OBSERVATION_SCHEMA_V1_ID
    if resolved is ObservationVersion.V2:
        return OBSERVATION_SCHEMA_V2_ID
    return OBSERVATION_SCHEMA_V3_ID


def _non_negative(value: int) -> int:
    return max(0, int(value))


def base_observation_vector(observation: RawObservation) -> np.ndarray:
    p1_x = observation.p1_x if observation.p1_has_position else 0
    p1_y = observation.p1_y if observation.p1_has_position else 0
    p2_x = observation.p2_x if observation.p2_has_position else 0
    p2_y = observation.p2_y if observation.p2_has_position else 0
    return np.array(
        [
            observation.round_time / 99.0,
            observation.p1_health / 103.0,
            observation.p2_health / 103.0,
            observation.p1_power / 128.0,
            observation.p2_power / 128.0,
            observation.p1_power_state / 128.0,
            observation.p2_power_state / 128.0,
            _non_negative(observation.p1_advanced_power_value) / 128.0,
            _non_negative(observation.p1_advanced_power_stocks) / 5.0,
            _non_negative(observation.p2_advanced_power_value) / 128.0,
            _non_negative(observation.p2_advanced_power_stocks) / 5.0,
            observation.p1_stun / 255.0,
            observation.p2_stun / 255.0,
            _non_negative(observation.p1_combo_count) / 99.0,
            _non_negative(observation.p2_combo_count) / 99.0,
            p1_x / NATIVE_WIDTH,
            p1_y / NATIVE_HEIGHT,
            p2_x / NATIVE_WIDTH,
            p2_y / NATIVE_HEIGHT,
            observation.distance_x / NATIVE_WIDTH,
            observation.distance_y / NATIVE_HEIGHT,
            float(observation.p1_has_position != 0),
            float(observation.p2_has_position != 0),
        ],
        dtype=np.float32,
    )


def _action_one_hot(action_id: int) -> np.ndarray:
    result = np.zeros(ACTION_ONE_HOT_SIZE, dtype=np.float32)
    index = action_id if 0 <= action_id < ACTION_COUNT else NONE_ACTION_INDEX
    result[index] = 1.0
    return result


def _velocity(
    current: RawObservation,
    previous: Optional[RawObservation],
    player: int,
    action_repeat: int,
) -> tuple[float, float]:
    if previous is None:
        return 0.0, 0.0
    has_current = current.p1_has_position if player == 1 else current.p2_has_position
    has_previous = previous.p1_has_position if player == 1 else previous.p2_has_position
    if not has_current or not has_previous:
        return 0.0, 0.0
    current_x = current.p1_x if player == 1 else current.p2_x
    current_y = current.p1_y if player == 1 else current.p2_y
    previous_x = previous.p1_x if player == 1 else previous.p2_x
    previous_y = previous.p1_y if player == 1 else previous.p2_y
    frame_count = max(1, action_repeat)
    return (
        float(np.clip((current_x - previous_x) / frame_count / 16.0, -1.0, 1.0)),
        float(np.clip((current_y - previous_y) / frame_count / 16.0, -1.0, 1.0)),
    )


def _screen_edge_distance(x: int, has_position: int) -> float:
    if not has_position:
        return 0.0
    clamped = float(np.clip(x, 0.0, NATIVE_WIDTH))
    return min(clamped, NATIVE_WIDTH - clamped) / (NATIVE_WIDTH * 0.5)


def _flag(value: int | bool) -> float:
    return float(bool(value))


def _normalized_non_negative(value: int, scale: float) -> float:
    return float(np.clip(max(0, int(value)) / scale, 0.0, 1.0))


def _generic_timing_vector(context: ObservationContext) -> np.ndarray:
    if not context.event_features_enabled:
        return np.zeros(
            len(OBSERVATION_V3_REPURPOSED_INDICES),
            dtype=np.float32,
        )
    timing = context.timing_state
    if timing is None:
        raise ValueError("StrategyV4 requires a combat timing state snapshot")

    def reaction_values(player: PlayerTimingState) -> list[float]:
        valid = bool(player.reaction_valid)
        remaining_valid = bool(player.reaction_remaining_valid)
        kind = int(player.reaction_kind) if valid else 0
        return [
            _flag(remaining_valid),
            _normalized_non_negative(
                player.reaction_remaining
                if valid and remaining_valid
                else 0,
                REACTION_TIME_SCALE_FRAMES,
            ),
            _flag(kind == 1),
            _flag(kind == 2),
        ]

    generic = context.generic_combat_state
    defense_one_hot = np.zeros(DEFENSE_PHASE_COUNT, dtype=np.float32)
    defense_one_hot[int(generic.defense_phase)] = 1.0
    confirm_one_hot = np.zeros(CONFIRM_PHASE_COUNT, dtype=np.float32)
    confirm_one_hot[int(generic.confirm_phase)] = 1.0

    result = np.array(
        [
            *reaction_values(timing.p1),
            *reaction_values(timing.p2),
            _flag(timing.p1.actionable_valid),
            _flag(timing.p1.actionable_valid and timing.p1.actionable),
            _flag(timing.p2.actionable_valid),
            _flag(timing.p2.actionable_valid and timing.p2.actionable),
            _flag(timing.p1.recovery_valid),
            _normalized_non_negative(
                timing.p1.recovery_remaining
                if timing.p1.recovery_valid
                else 0,
                RECOVERY_TIME_SCALE_FRAMES,
            ),
            _flag(timing.p2.recovery_valid),
            _normalized_non_negative(
                timing.p2.recovery_remaining
                if timing.p2.recovery_valid
                else 0,
                RECOVERY_TIME_SCALE_FRAMES,
            ),
            _flag(timing.frame_advantage_valid),
            float(
                np.clip(
                    timing.frame_advantage / FRAME_ADVANTAGE_SCALE_FRAMES,
                    -1.0,
                    1.0,
                )
                if timing.frame_advantage_valid
                else 0.0
            ),
            *defense_one_hot,
            *confirm_one_hot,
            _normalized_non_negative(
                generic.defense_phase_age_frames,
                GENERIC_PHASE_AGE_SCALE_FRAMES,
            ),
            _normalized_non_negative(
                generic.confirm_phase_age_frames,
                GENERIC_PHASE_AGE_SCALE_FRAMES,
            ),
            _flag(generic.queue_window_open),
            _flag(generic.last_action_accepted),
        ],
        dtype=np.float32,
    )
    if result.shape != (len(OBSERVATION_V3_REPURPOSED_INDICES),):
        raise AssertionError(
            f"StrategyV4 timing vector has unexpected shape {result.shape}"
        )
    return result


def encode_observation(
    version: ObservationVersion | str,
    observation: RawObservation,
    context: ObservationContext,
) -> np.ndarray:
    base = np.concatenate(
        [
            base_observation_vector(observation),
            np.array(
                [
                    float(context.input_ready),
                    context.normalized_combo_phase,
                    context.combo_phase_age,
                ],
                dtype=np.float32,
            ),
        ]
    )
    resolved_version = ObservationVersion(version)
    if resolved_version is ObservationVersion.V1:
        return base

    state = context.strategy_state
    if state is None:
        raise ValueError("StrategyV2 requires a strategy state snapshot")
    p1_vx, p1_vy = _velocity(
        observation,
        context.previous_observation,
        1,
        context.action_repeat,
    )
    p2_vx, p2_vy = _velocity(
        observation,
        context.previous_observation,
        2,
        context.action_repeat,
    )
    p1_airborne = bool(observation.p1_has_position) and 0 <= observation.p1_y < AIRBORNE_Y_THRESHOLD
    p2_airborne = bool(observation.p2_has_position) and 0 <= observation.p2_y < AIRBORNE_Y_THRESHOLD
    scalars = np.array(
        [
            _non_negative(state.p1_status) / 255.0,
            _non_negative(state.p2_status) / 255.0,
            float(state.p1_ready != 0),
            float(state.p2_ready != 0),
            float(p1_airborne),
            float(p2_airborne),
            p1_vx,
            p1_vy,
            p2_vx,
            p2_vy,
            _screen_edge_distance(observation.p1_x, observation.p1_has_position),
            _screen_edge_distance(observation.p2_x, observation.p2_has_position),
            min(_non_negative(state.p1_action_elapsed_frames) / ACTION_TIME_SCALE_FRAMES, 1.0),
            min(_non_negative(state.p1_action_remaining_frames) / ACTION_TIME_SCALE_FRAMES, 1.0),
            0.0,
            0.0,
            float(state.p1_facing_left != 0),
            float(context.profile_is_fight),
        ],
        dtype=np.float32,
    )
    phase = np.zeros(COMBAT_PHASE_COUNT, dtype=np.float32)
    phase[int(context.combat_phase)] = 1.0
    result = np.concatenate(
        [
            base,
            scalars,
            _action_one_hot(state.p1_active_action_id),
            _action_one_hot(state.p1_queued_action_id),
            _action_one_hot(-1),
            phase,
        ]
    ).astype(np.float32, copy=False)
    if result.shape != (OBSERVATION_V2_SIZE,):
        raise AssertionError(f"StrategyV2 observation has unexpected shape {result.shape}")
    if resolved_version is ObservationVersion.V3:
        result = result.copy()
        result[list(OBSERVATION_V3_REPURPOSED_INDICES)] = (
            _generic_timing_vector(context)
        )
    return result


class GenericCombatStateMachine:
    """Event-driven state shared by every profile and opponent controller.

    The machine records generic combat history only. It never sees a recipe,
    scenario id, Oracle action, or curriculum target.
    """

    ACTION_STARTED = 1
    COMBO_HIT = 2
    BLOCK_CONTACT = 4
    CLEAN_HIT = 7
    BLOCKSTUN_STARTED = 10
    BLOCKSTUN_ENDED = 11
    ATTACK_ACTION_MIN = 6
    PHASE_TIMEOUT_FRAMES = 60
    POST_BLOCK_TIMEOUT_FRAMES = 36

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.defense_phase = DefensePhase.NEUTRAL
        self.confirm_phase = ConfirmPhase.NEUTRAL
        self.defense_phase_age_frames = 0
        self.confirm_phase_age_frames = 0
        self.starter_action_id = -1
        self.starter_action_serial = 0
        self.pending_followup_action_id = -1
        self.pending_followup_action_serial = 0
        self.queue_window_open = False
        self.last_action_accepted = False
        self.last_update_starter_hit_count = 0
        self.last_update_starter_blocked_count = 0

    def _set_defense(self, phase: DefensePhase) -> None:
        if self.defense_phase is not phase:
            self.defense_phase = phase
            self.defense_phase_age_frames = 0

    def _set_confirm(self, phase: ConfirmPhase) -> None:
        if self.confirm_phase is not phase:
            self.confirm_phase = phase
            self.confirm_phase_age_frames = 0

    def update(
        self,
        events: Iterable[CombatEvent],
        timing_state: CombatTimingState,
        *,
        elapsed_frames: int,
        p2_attack_pressure: bool,
        queue_window_open: bool,
        last_action_accepted: bool,
    ) -> GenericCombatState:
        # These counters are telemetry for the current emulator step. They do
        # not participate in the state transition or encoded observation.
        self.last_update_starter_hit_count = 0
        self.last_update_starter_blocked_count = 0
        self.defense_phase_age_frames += max(0, int(elapsed_frames))
        self.confirm_phase_age_frames += max(0, int(elapsed_frames))
        self.queue_window_open = bool(queue_window_open)
        self.last_action_accepted = bool(last_action_accepted)

        ordered_events = sorted(
            events,
            key=lambda event: (
                int(getattr(event, "absolute_engine_frame", 0)),
                int(getattr(event, "frame_offset", 0)),
            ),
        )
        for event in ordered_events:
            event_type = int(event.event_type)
            source = int(event.source_player)
            target = int(event.target_player)

            if event_type == self.CLEAN_HIT and target == 1:
                self._set_defense(DefensePhase.NEUTRAL)
            elif event_type == self.BLOCK_CONTACT and target == 1:
                self._set_defense(DefensePhase.BLOCK_CONTACT)
            elif event_type == self.BLOCKSTUN_STARTED and target == 1:
                self._set_defense(DefensePhase.BLOCK_REACTION)
            elif event_type == self.BLOCKSTUN_ENDED and target == 1:
                self._set_defense(DefensePhase.POST_BLOCK)

            if (
                event_type == self.ACTION_STARTED
                and source == 1
                and int(event.action_id) >= self.ATTACK_ACTION_MIN
            ):
                if self.confirm_phase is ConfirmPhase.STARTER_HIT:
                    self._set_confirm(ConfirmPhase.FOLLOWUP_STARTED)
                elif self.confirm_phase is ConfirmPhase.STARTER_PENDING:
                    # Cancel/queue 可能在起手命中於 RAM 入帳前就啟動。
                    # 先保留 child serial，待同一 starter 的 COMBO_HIT
                    # 到達後再轉成 FOLLOWUP_STARTED。
                    if int(event.action_serial) != self.starter_action_serial:
                        self.pending_followup_action_id = int(event.action_id)
                        self.pending_followup_action_serial = int(
                            event.action_serial
                        )
                elif self.confirm_phase is ConfirmPhase.NEUTRAL:
                    self.starter_action_id = int(event.action_id)
                    self.starter_action_serial = int(event.action_serial)
                    self._set_confirm(ConfirmPhase.STARTER_PENDING)
            elif (
                event_type == self.COMBO_HIT
                and source == 1
                and target == 2
            ):
                if (
                    self.confirm_phase is ConfirmPhase.STARTER_PENDING
                    and int(event.action_id) == self.starter_action_id
                    and int(event.action_serial) == self.starter_action_serial
                ):
                    self.last_update_starter_hit_count += 1
                    if self.pending_followup_action_serial:
                        self._set_confirm(ConfirmPhase.FOLLOWUP_STARTED)
                    else:
                        self._set_confirm(ConfirmPhase.STARTER_HIT)
                elif self.confirm_phase is ConfirmPhase.FOLLOWUP_STARTED:
                    self.confirm_phase_age_frames = 0
            elif (
                event_type == self.BLOCK_CONTACT
                and source == 1
                and target == 2
                and self.confirm_phase is ConfirmPhase.STARTER_PENDING
                and int(event.action_id) == self.starter_action_id
                and int(event.action_serial) == self.starter_action_serial
            ):
                self.last_update_starter_blocked_count += 1
                self.pending_followup_action_id = -1
                self.pending_followup_action_serial = 0
                self._set_confirm(ConfirmPhase.STARTER_BLOCKED)

        if (
            timing_state.p1.reaction_valid
            and int(timing_state.p1.reaction_kind) == 1
            and self.defense_phase is not DefensePhase.POST_BLOCK
        ):
            self._set_defense(DefensePhase.BLOCK_REACTION)
        elif (
            self.defense_phase is DefensePhase.NEUTRAL
            and p2_attack_pressure
        ):
            self._set_defense(DefensePhase.PRESSURE)

        defense_timeout = (
            self.POST_BLOCK_TIMEOUT_FRAMES
            if self.defense_phase is DefensePhase.POST_BLOCK
            else self.PHASE_TIMEOUT_FRAMES
        )
        if self.defense_phase_age_frames > defense_timeout:
            self._set_defense(DefensePhase.NEUTRAL)
        if self.confirm_phase_age_frames > self.PHASE_TIMEOUT_FRAMES:
            self.starter_action_id = -1
            self.starter_action_serial = 0
            self.pending_followup_action_id = -1
            self.pending_followup_action_serial = 0
            self._set_confirm(ConfirmPhase.NEUTRAL)

        return self.snapshot()

    def snapshot(self) -> GenericCombatState:
        return GenericCombatState(
            defense_phase=self.defense_phase,
            confirm_phase=self.confirm_phase,
            defense_phase_age_frames=self.defense_phase_age_frames,
            confirm_phase_age_frames=self.confirm_phase_age_frames,
            queue_window_open=self.queue_window_open,
            last_action_accepted=self.last_action_accepted,
        )


class CombatPhaseMachine:
    """A compact reward-machine state exposed to the policy, not a reward hack."""

    def __init__(self) -> None:
        self.phase = CombatPhase.NEUTRAL

    def reset(self) -> None:
        self.phase = CombatPhase.NEUTRAL

    def update(
        self,
        observation: RawObservation,
        strategy_state: StrategyState,
        *,
        p1_damage: float = 0.0,
    ) -> CombatPhase:
        combo_count = _non_negative(observation.p1_combo_count)
        if p1_damage > 0.0:
            self.phase = CombatPhase.DEFENSE
        elif combo_count >= 2:
            self.phase = CombatPhase.COMBO
        elif combo_count == 1:
            self.phase = CombatPhase.CONFIRM
        elif (
            observation.p2_health > 0
            and not strategy_state.p2_ready
            and observation.p2_has_position
            and observation.p2_y >= AIRBORNE_Y_THRESHOLD
        ):
            self.phase = CombatPhase.KNOCKDOWN
        elif observation.p1_has_position and observation.p2_has_position and abs(observation.distance_x) > 100:
            self.phase = CombatPhase.APPROACH
        else:
            self.phase = CombatPhase.NEUTRAL
        return self.phase
