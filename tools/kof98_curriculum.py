"""Deterministic KOF98 curriculum levels built from safe save states.

A libretro save state contains the emulator core, but it does not contain the
frontend's C++ action queues or Python reward-machine state.  A level therefore
stores a safe base state and a deterministic prelude instead of serializing in
the middle of an action.  Replaying the recipe reconstructs the same tactical
window without restoring half of an input script.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Protocol


LEVEL_RECIPE_SCHEMA_VERSION = 1


class CurriculumTask(str, Enum):
    DEFENSE = "defense"
    ANTI_AIR = "anti_air"
    APPROACH = "approach"
    HIT_CONFIRM = "hit_confirm"


class RewardMachinePhase(IntEnum):
    WAITING = 0
    OPPORTUNITY = 1
    COMMITTED = 2
    SUCCEEDED = 3
    FAILED = 4


@dataclass(frozen=True)
class RewardMachineTransition:
    previous_phase: RewardMachinePhase
    phase: RewardMachinePhase
    success: bool = False
    failure: bool = False


class TacticalRewardMachine:
    """Event-history state for one reverse-curriculum tactical episode.

    An action press is never success by itself.  The environment must first
    report a real opportunity and then a real outcome (block without damage,
    airborne hit, P1-caused safe entry, or confirmed follow-up hit).
    """

    def __init__(self) -> None:
        self.phase = RewardMachinePhase.WAITING

    def reset(self) -> None:
        self.phase = RewardMachinePhase.WAITING

    def advance(
        self,
        *,
        opportunity: bool,
        committed: bool,
        success: bool,
        failure: bool,
    ) -> RewardMachineTransition:
        previous = self.phase
        if self.phase in (
            RewardMachinePhase.SUCCEEDED,
            RewardMachinePhase.FAILED,
        ):
            return RewardMachineTransition(previous, self.phase)

        if success:
            self.phase = RewardMachinePhase.SUCCEEDED
        elif failure and self.phase is not RewardMachinePhase.WAITING:
            self.phase = RewardMachinePhase.FAILED
        elif committed and self.phase is not RewardMachinePhase.WAITING:
            self.phase = RewardMachinePhase.COMMITTED
        elif opportunity:
            self.phase = RewardMachinePhase.OPPORTUNITY

        return RewardMachineTransition(
            previous_phase=previous,
            phase=self.phase,
            success=(
                self.phase is RewardMachinePhase.SUCCEEDED
                and previous is not RewardMachinePhase.SUCCEEDED
            ),
            failure=(
                self.phase is RewardMachinePhase.FAILED
                and previous is not RewardMachinePhase.FAILED
            ),
        )


@dataclass(frozen=True)
class OracleAction:
    action_id: int
    wait_before_frames: int = 0
    max_dispatch_wait_frames: int = 240
    settle_after_frames: int = 0

    def validate(self) -> None:
        if self.action_id < 0:
            raise ValueError("Oracle action id must be non-negative")
        if self.wait_before_frames < 0:
            raise ValueError("Oracle wait_before_frames must be non-negative")
        if self.max_dispatch_wait_frames < 0:
            raise ValueError("Oracle max_dispatch_wait_frames must be non-negative")
        if self.settle_after_frames < 0:
            raise ValueError("Oracle settle_after_frames must be non-negative")


@dataclass(frozen=True)
class LevelRecipe:
    name: str
    task: CurriculumTask
    base_state: Path
    p2_action_id: int | None = None
    p2_start_delay_frames: int = 0
    p2_prelude_frames: int = 0
    oracle_actions: tuple[OracleAction, ...] = ()
    settle_frames: int = 120
    level: int = 0
    trigger_frame: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = LEVEL_RECIPE_SCHEMA_VERSION

    def validate(self, action_count: int | None = None) -> None:
        if self.schema_version != LEVEL_RECIPE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported level recipe schema {self.schema_version}; "
                f"expected {LEVEL_RECIPE_SCHEMA_VERSION}"
            )
        if not self.name.strip():
            raise ValueError("Level recipe name cannot be empty")
        if (
            self.p2_start_delay_frames < 0
            or self.p2_prelude_frames < 0
            or self.settle_frames < 0
            or self.level < 0
        ):
            raise ValueError("Level recipe frame counts and level must be non-negative")
        if self.p2_start_delay_frames and self.p2_prelude_frames:
            raise ValueError(
                "A recipe cannot both delay P2 after policy control and replay a "
                "P2 prelude before policy control"
            )
        if self.p2_action_id is not None and self.p2_action_id < 0:
            raise ValueError("P2 action id must be non-negative")
        if action_count is not None:
            action_ids = [action.action_id for action in self.oracle_actions]
            if self.p2_action_id is not None:
                action_ids.append(self.p2_action_id)
            invalid = [action_id for action_id in action_ids if action_id >= action_count]
            if invalid:
                raise ValueError(
                    f"Recipe action id is outside Discrete({action_count}): {invalid}"
                )
        for action in self.oracle_actions:
            action.validate()

    def resolved(self, root: Path) -> "LevelRecipe":
        base_state = self.base_state
        if not base_state.is_absolute():
            base_state = root / base_state
        return LevelRecipe(
            name=self.name,
            task=self.task,
            base_state=base_state.resolve(),
            p2_action_id=self.p2_action_id,
            p2_start_delay_frames=self.p2_start_delay_frames,
            p2_prelude_frames=self.p2_prelude_frames,
            oracle_actions=self.oracle_actions,
            settle_frames=self.settle_frames,
            level=self.level,
            trigger_frame=self.trigger_frame,
            metadata=dict(self.metadata),
            schema_version=self.schema_version,
        )

    def to_dict(self, base_dir: Path | None = None) -> dict[str, Any]:
        data = asdict(self)
        data["task"] = self.task.value
        data["base_state"] = _portable_path(self.base_state, base_dir)
        data["oracle_actions"] = [asdict(action) for action in self.oracle_actions]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LevelRecipe":
        recipe = cls(
            name=str(data["name"]),
            task=CurriculumTask(data["task"]),
            base_state=Path(data["base_state"]),
            p2_action_id=(
                None if data.get("p2_action_id") is None else int(data["p2_action_id"])
            ),
            p2_start_delay_frames=int(data.get("p2_start_delay_frames", 0)),
            p2_prelude_frames=int(data.get("p2_prelude_frames", 0)),
            oracle_actions=tuple(
                OracleAction(
                    action_id=int(action["action_id"]),
                    wait_before_frames=int(action.get("wait_before_frames", 0)),
                    max_dispatch_wait_frames=int(
                        action.get("max_dispatch_wait_frames", 240)
                    ),
                    settle_after_frames=int(action.get("settle_after_frames", 0)),
                )
                for action in data.get("oracle_actions", [])
            ),
            settle_frames=int(data.get("settle_frames", 120)),
            level=int(data.get("level", 0)),
            trigger_frame=(
                None if data.get("trigger_frame") is None else int(data["trigger_frame"])
            ),
            metadata=dict(data.get("metadata", {})),
            schema_version=int(
                data.get("schema_version", LEVEL_RECIPE_SCHEMA_VERSION)
            ),
        )
        recipe.validate()
        return recipe


class CurriculumClient(Protocol):
    action_count: int

    def load_state(self, state_path: str | Path) -> None: ...
    def snapshot_safe(self) -> bool: ...
    def set_p2_training_ai(self, enabled: bool) -> None: ...
    def set_p2_action_ai(self, enabled: bool) -> None: ...
    def set_p2_action(self, action_id: int) -> None: ...
    def run_frames(self, frames: int) -> None: ...
    def observation(self): ...


def load_level_recipes(path: str | Path, root: Path | None = None) -> list[LevelRecipe]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    items = payload.get("recipes", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("Level recipe JSON must be a list or contain a 'recipes' list")
    base = root if root is not None else source.parent
    return [LevelRecipe.from_dict(item).resolved(base) for item in items]


def save_level_recipes(
    path: str | Path,
    recipes: list[LevelRecipe],
    metadata: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": LEVEL_RECIPE_SCHEMA_VERSION,
        "metadata": metadata or {},
        "recipes": [recipe.to_dict(destination.parent) for recipe in recipes],
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_level(client: CurriculumClient, recipe: LevelRecipe):
    """Restore a safe base and replay the deterministic P2 prelude."""

    recipe.validate(client.action_count)
    client.set_p2_training_ai(False)
    client.set_p2_action_ai(False)
    client.load_state(recipe.base_state)
    if not client.snapshot_safe():
        raise RuntimeError(
            f"Level recipe '{recipe.name}' base state is not a safe neutral snapshot: "
            f"{recipe.base_state}"
        )

    if recipe.p2_action_id is not None:
        client.set_p2_action_ai(True)
        if recipe.p2_start_delay_frames == 0:
            client.set_p2_action(recipe.p2_action_id)
    if recipe.p2_prelude_frames:
        client.run_frames(recipe.p2_prelude_frames)
    return client.observation()


def state_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Path, base_dir: Path | None) -> str:
    if base_dir is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path)
