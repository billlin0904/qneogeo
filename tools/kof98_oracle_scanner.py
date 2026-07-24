"""Scan deterministic KOF98 tactical windows and emit curriculum recipes.

The scanner is intentionally outside PPO.  It first proves that a tactic is
physically executable from a safe state, then records successful trigger
frames as replayable LevelRecipe entries for reverse curriculum training.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from kof98_curriculum import (
    CurriculumTask,
    LevelRecipe,
    OracleAction,
    prepare_level,
    save_level_recipes,
    state_sha256,
)
from kof98_env import (
    ACTION_COUNT,
    CLOSE_C_ACTION_ID,
    FORWARD_ACTION_ID,
    KofEnvClient,
    ONIYAKI_ACTION_ID,
    RED_KICK_ACTION_ID,
    SEVENTY_FIVE_SHIKI_KAI_ACTION_ID,
    STEP_EVENT_COMBO_HIT,
)


JUMP_C_ACTION_ID = 20
JUMP_D_ACTION_ID = 21
STAND_GUARD_ACTION_ID = 2
TACTICAL_EFFECTIVE_DISTANCE = 90
MIN_APPROACH_PROGRESS = 12
POLICY_DECISION_FRAMES = 4


def scheduled_start_delay(trigger: int, requested_lead: int) -> int:
    raw_delay = max(0, requested_lead - trigger)
    if raw_delay == 0:
        return 0
    return (
        (raw_delay + POLICY_DECISION_FRAMES - 1)
        // POLICY_DECISION_FRAMES
        * POLICY_DECISION_FRAMES
    )


@dataclass
class OracleTrace:
    dispatched_actions: list[int]
    dispatch_frames: list[int]
    events: list[dict]
    elapsed_frames: int
    initial_p1_health: int
    initial_p2_health: int
    final_p1_health: int
    final_p2_health: int
    initial_p1_x: int
    initial_p2_x: int
    final_p1_x: int
    final_p2_x: int
    initial_distance: int
    final_distance: int
    max_combo: int
    action_error: str = ""

    @property
    def p1_damage(self) -> int:
        return max(0, self.initial_p1_health - self.final_p1_health)

    @property
    def p2_damage(self) -> int:
        return max(0, self.initial_p2_health - self.final_p2_health)


def default_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = default_root()
    parser = argparse.ArgumentParser(
        description="Scan deterministic KOF98 tactical Oracle windows."
    )
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--task",
        action="append",
        choices=("all", *(task.value for task in CurriculumTask)),
        default=None,
        help="Task to scan. Repeat for multiple tasks; defaults to all.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "ai_logs" / "oracle" / "kof98_v3a2_oracle.json",
    )
    parser.add_argument(
        "--recipes-output",
        type=Path,
        default=root / "ai_logs" / "oracle" / "kof98_v3a2_recipes.json",
    )
    return parser.parse_args()


def append_current_events(client: KofEnvClient, events: list[dict], frame: int) -> int:
    for event in client.step_events():
        events.append(
            {
                "absolute_frame": frame + int(event.frame_offset),
                "event_type": int(event.event_type),
                "action_id": int(event.action_id),
                "action_serial": int(event.action_serial),
                "combo_before": int(event.combo_before),
                "combo_after": int(event.combo_after),
                "p2_hp_delta": int(event.p2_hp_delta),
                "p1_hp_delta": int(event.p1_hp_delta),
                "target_y_at_event": int(event.target_y_at_event),
                "target_airborne_at_event": bool(event.target_airborne_at_event),
                "target_airborne_after_event": bool(
                    event.target_airborne_after_event
                ),
                "hit_contact": bool(event.hit_contact),
                "block_contact": bool(event.block_contact),
            }
        )
    return frame


def advance_one_frame(client: KofEnvClient, events: list[dict], frame: int) -> int:
    client.run_frames(1)
    append_current_events(client, events, frame)
    return frame + 1


def run_oracle(client: KofEnvClient, recipe: LevelRecipe) -> OracleTrace:
    initial = prepare_level(client, recipe)
    events: list[dict] = []
    dispatched_actions: list[int] = []
    dispatch_frames: list[int] = []
    frame = 0
    max_combo = max(0, int(initial.p1_combo_count))
    action_error = ""

    for oracle_action in recipe.oracle_actions:
        for _ in range(oracle_action.wait_before_frames):
            frame = advance_one_frame(client, events, frame)

        dispatched = False
        for _ in range(oracle_action.max_dispatch_wait_frames + 1):
            can_start = client.input_ready() and client.p1_ready_for_action()
            can_queue = client.can_queue_action(oracle_action.action_id)
            if can_start or can_queue:
                client.step(oracle_action.action_id, 1)
                append_current_events(client, events, frame)
                status = client.action_status()
                if not status.action_accepted:
                    action_error = (
                        f"Action {oracle_action.action_id} was dispatchable but rejected"
                    )
                    break
                dispatched_actions.append(oracle_action.action_id)
                dispatch_frames.append(frame)
                frame += 1
                dispatched = True
                break
            frame = advance_one_frame(client, events, frame)

        if action_error:
            break
        if not dispatched:
            action_error = (
                f"Action {oracle_action.action_id} did not become dispatchable within "
                f"{oracle_action.max_dispatch_wait_frames} frames"
            )
            break
        for _ in range(oracle_action.settle_after_frames):
            frame = advance_one_frame(client, events, frame)
        max_combo = max(max_combo, int(client.observation().p1_combo_count))

    for _ in range(recipe.settle_frames):
        frame = advance_one_frame(client, events, frame)
        max_combo = max(max_combo, int(client.observation().p1_combo_count))

    final = client.observation()
    max_combo = max(
        [max_combo, *(int(event["combo_after"]) for event in events)],
    )
    return OracleTrace(
        dispatched_actions=dispatched_actions,
        dispatch_frames=dispatch_frames,
        events=events,
        elapsed_frames=frame,
        initial_p1_health=int(initial.p1_health),
        initial_p2_health=int(initial.p2_health),
        final_p1_health=int(final.p1_health),
        final_p2_health=int(final.p2_health),
        initial_p1_x=int(initial.p1_x),
        initial_p2_x=int(initial.p2_x),
        final_p1_x=int(final.p1_x),
        final_p2_x=int(final.p2_x),
        initial_distance=abs(int(initial.distance_x)),
        final_distance=abs(int(final.distance_x)),
        max_combo=max_combo,
        action_error=action_error,
    )


def scan_approach(client: KofEnvClient, state: Path) -> tuple[list[dict], list[LevelRecipe]]:
    results: list[dict] = []
    recipes: list[LevelRecipe] = []
    for chunks in range(1, 13):
        recipe = LevelRecipe(
            name=f"approach_forward_{chunks}_chunks",
            task=CurriculumTask.APPROACH,
            base_state=state,
            oracle_actions=tuple(OracleAction(FORWARD_ACTION_ID) for _ in range(chunks)),
            # Hold chunks are four frames.  Keep enough time for all chunks
            # plus a correction window instead of truncating after one step.
            settle_frames=max(48, chunks * 4 + 16),
            level=max(0, 12 - chunks),
        )
        trace = run_oracle(client, recipe)
        direction = 1 if trace.initial_p2_x >= trace.initial_p1_x else -1
        p1_progress = direction * (trace.final_p1_x - trace.initial_p1_x)
        success = (
            not trace.action_error
            and trace.final_distance <= TACTICAL_EFFECTIVE_DISTANCE
            and p1_progress >= MIN_APPROACH_PROGRESS
            and trace.p1_damage == 0
        )
        results.append(
            {
                "recipe": recipe.to_dict(),
                "success": success,
                "p1_progress": p1_progress,
                "trace": asdict(trace),
            }
        )
        if success:
            recipes.append(recipe)
            break
    return results, recipes


def scan_anti_air(client: KofEnvClient, state: Path) -> tuple[list[dict], list[LevelRecipe]]:
    results: list[dict] = []
    successful_triggers: list[tuple[int, int]] = []
    for p2_action in (JUMP_C_ACTION_ID, JUMP_D_ACTION_ID):
        for prelude in range(0, 37, 2):
            recipe = LevelRecipe(
                name=f"anti_air_p2_{p2_action}_trigger_{prelude}",
                task=CurriculumTask.ANTI_AIR,
                base_state=state,
                p2_action_id=p2_action,
                p2_prelude_frames=prelude,
                oracle_actions=(OracleAction(ONIYAKI_ACTION_ID),),
                settle_frames=80,
                trigger_frame=prelude,
            )
            trace = run_oracle(client, recipe)
            anti_air_hits = [
                event
                for event in trace.events
                if event["event_type"] == STEP_EVENT_COMBO_HIT
                and event["action_id"] == ONIYAKI_ACTION_ID
                and event["p2_hp_delta"] > 0
                and event["target_airborne_at_event"]
            ]
            success = not trace.action_error and bool(anti_air_hits)
            results.append(
                {
                    "recipe": recipe.to_dict(),
                    "success": success,
                    "anti_air_hits": anti_air_hits,
                    "trace": asdict(trace),
                }
            )
            if success:
                successful_triggers.append((p2_action, prelude))

    recipes: list[LevelRecipe] = []
    latest_trigger_by_action = {
        p2_action: max(
            trigger
            for candidate_action, trigger in successful_triggers
            if candidate_action == p2_action
        )
        for p2_action in {action for action, _trigger in successful_triggers}
    }
    for p2_action, trigger in sorted(latest_trigger_by_action.items()):
        for level, lead in enumerate((0, 4, 8, 12)):
            prelude = max(0, trigger - lead)
            start_delay = scheduled_start_delay(trigger, lead)
            actual_lead = lead if start_delay == 0 else trigger + start_delay
            candidate = LevelRecipe(
                name=f"anti_air_p2_{p2_action}_level_{level}",
                task=CurriculumTask.ANTI_AIR,
                base_state=state,
                p2_action_id=p2_action,
                p2_start_delay_frames=start_delay,
                p2_prelude_frames=prelude,
                oracle_actions=(
                    OracleAction(ONIYAKI_ACTION_ID, wait_before_frames=trigger - prelude),
                ),
                settle_frames=80,
                level=level,
                trigger_frame=trigger,
                metadata={
                    "requested_policy_lead_frames": lead,
                    "policy_lead_frames": actual_lead,
                },
            )
            recipes.append(candidate)
    return results, recipes


def scan_defense(client: KofEnvClient, state: Path) -> tuple[list[dict], list[LevelRecipe]]:
    results: list[dict] = []
    successful_triggers: list[tuple[int, int]] = []
    for p2_action in (CLOSE_C_ACTION_ID, 9, 11, 13):
        for prelude in range(0, 21, 2):
            baseline = LevelRecipe(
                name=f"defense_baseline_p2_{p2_action}_{prelude}",
                task=CurriculumTask.DEFENSE,
                base_state=state,
                p2_action_id=p2_action,
                p2_prelude_frames=prelude,
                settle_frames=50,
            )
            guarded = LevelRecipe(
                name=f"defense_p2_{p2_action}_trigger_{prelude}",
                task=CurriculumTask.DEFENSE,
                base_state=state,
                p2_action_id=p2_action,
                p2_prelude_frames=prelude,
                oracle_actions=tuple(
                    OracleAction(STAND_GUARD_ACTION_ID) for _ in range(4)
                ),
                settle_frames=50,
                trigger_frame=prelude,
            )
            baseline_trace = run_oracle(client, baseline)
            guarded_trace = run_oracle(client, guarded)
            block_contacts = [
                event for event in guarded_trace.events if event["block_contact"]
            ]
            success = (
                baseline_trace.p1_damage > 0
                and guarded_trace.p1_damage < baseline_trace.p1_damage
                and not guarded_trace.action_error
                and bool(block_contacts)
            )
            results.append(
                {
                    "recipe": guarded.to_dict(),
                    "success": success,
                    "baseline_p1_damage": baseline_trace.p1_damage,
                    "guarded_p1_damage": guarded_trace.p1_damage,
                    "block_contacts": block_contacts,
                    "trace": asdict(guarded_trace),
                }
            )
            if success:
                successful_triggers.append((p2_action, prelude))
    recipes: list[LevelRecipe] = []
    latest_trigger_by_action = {
        p2_action: max(
            trigger
            for candidate_action, trigger in successful_triggers
            if candidate_action == p2_action
        )
        for p2_action in {action for action, _trigger in successful_triggers}
    }
    for p2_action, trigger in sorted(latest_trigger_by_action.items()):
        for level, lead in enumerate((0, 4, 8, 12)):
            guard_actions = tuple(
                OracleAction(
                    STAND_GUARD_ACTION_ID,
                    wait_before_frames=(
                        trigger - max(0, trigger - lead)
                        if index == 0
                        else 0
                    ),
                )
                for index in range(4)
            )
            common = {
                "base_state": state,
                "p2_action_id": p2_action,
                "p2_start_delay_frames": scheduled_start_delay(trigger, lead),
                "p2_prelude_frames": max(0, trigger - lead),
                "level": level,
                "trigger_frame": trigger,
                "metadata": {
                    "requested_policy_lead_frames": lead,
                    "policy_lead_frames": (
                        lead
                        if lead <= trigger
                        else trigger + scheduled_start_delay(trigger, lead)
                    ),
                    "lesson": "guard_only",
                },
            }
            recipes.append(LevelRecipe(
                name=f"defense_p2_{p2_action}_level_{level}",
                task=CurriculumTask.DEFENSE,
                oracle_actions=guard_actions,
                settle_frames=50,
                **common,
            ))

            # Try fast normals first, then advancing/special attacks.  A block
            # counter recipe is emitted only when the deterministic trace
            # proves that the post-block action really hits.
            for counter_action in (11, 10, 7, CLOSE_C_ACTION_ID, 14, 16, 17):
                counter_recipe = LevelRecipe(
                    name=(
                        f"block_counter_p2_{p2_action}_a{counter_action}_"
                        f"level_{level}"
                    ),
                    task=CurriculumTask.DEFENSE,
                    oracle_actions=guard_actions + (OracleAction(counter_action),),
                    settle_frames=100,
                    **{
                        **common,
                        "metadata": {
                            **common["metadata"],
                            "lesson": "block_counter",
                            "counter_action_id": counter_action,
                        },
                    },
                )
                trace = run_oracle(client, counter_recipe)
                block_frames = [
                    int(event["absolute_frame"])
                    for event in trace.events
                    if event["block_contact"]
                ]
                counter_hits = [
                    event
                    for event in trace.events
                    if event["event_type"] == STEP_EVENT_COMBO_HIT
                    and event["action_id"] == counter_action
                    and block_frames
                    and int(event["absolute_frame"]) > min(block_frames)
                ]
                counter_success = (
                    not trace.action_error
                    and trace.p1_damage == 0
                    and bool(counter_hits)
                )
                results.append(
                    {
                        "recipe": counter_recipe.to_dict(),
                        "success": counter_success,
                        "block_contacts": block_frames,
                        "counter_hits": counter_hits,
                        "trace": asdict(trace),
                    }
                )
                if counter_success:
                    recipes.append(counter_recipe)
                    break
    return results, recipes


def scan_hit_confirm(client: KofEnvClient, state: Path) -> tuple[list[dict], list[LevelRecipe]]:
    plans = (
        ("close_c_75_red", (CLOSE_C_ACTION_ID, SEVENTY_FIVE_SHIKI_KAI_ACTION_ID, RED_KICK_ACTION_ID), 4),
        ("close_c_75_orochinagi", (CLOSE_C_ACTION_ID, SEVENTY_FIVE_SHIKI_KAI_ACTION_ID, 18), 4),
    )
    results: list[dict] = []
    recipes: list[LevelRecipe] = []
    for name, actions, required_combo in plans:
        recipe = LevelRecipe(
            name=f"hit_confirm_{name}",
            task=CurriculumTask.HIT_CONFIRM,
            base_state=state,
            oracle_actions=tuple(OracleAction(action_id) for action_id in actions),
            settle_frames=180,
            metadata={"required_combo": required_combo},
        )
        trace = run_oracle(client, recipe)
        success = not trace.action_error and trace.max_combo >= required_combo
        results.append(
            {"recipe": recipe.to_dict(), "success": success, "trace": asdict(trace)}
        )
        if success:
            recipes.append(recipe)
    return results, recipes


def requested_tasks(values: Iterable[str] | None) -> list[CurriculumTask]:
    selected = list(values or ["all"])
    if "all" in selected:
        return list(CurriculumTask)
    return list(dict.fromkeys(CurriculumTask(value) for value in selected))


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    states = {
        CurriculumTask.APPROACH: root / "saves" / "states" / "kof98.slot2.state",
        CurriculumTask.ANTI_AIR: root / "saves" / "states" / "kof98.slot2.state",
        CurriculumTask.DEFENSE: root / "saves" / "states" / "kof98.slot1.state",
        CurriculumTask.HIT_CONFIRM: root / "saves" / "states" / "kof98.slot1.state",
    }
    missing = [str(path) for path in states.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Oracle base state(s): " + ", ".join(sorted(set(missing))))

    client = KofEnvClient(root / "build-vs2026-x64" / "Release" / "fbneo_training.dll")
    client.load_core(root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll")
    client.load_game(
        root / "roms" / "fbneo" / "kof98.zip",
        root / "system",
        root / "saves",
    )
    scanners = {
        CurriculumTask.APPROACH: scan_approach,
        CurriculumTask.ANTI_AIR: scan_anti_air,
        CurriculumTask.DEFENSE: scan_defense,
        CurriculumTask.HIT_CONFIRM: scan_hit_confirm,
    }
    report: dict = {
        "runtime_contract": client.contract_metadata(),
        "action_count": ACTION_COUNT,
        "states": {
            task.value: {
                "path": str(path),
                "sha256": state_sha256(path),
            }
            for task, path in states.items()
        },
        "tasks": {},
    }
    recipes: list[LevelRecipe] = []
    try:
        for task in requested_tasks(args.task):
            results, task_recipes = scanners[task](client, states[task])
            report["tasks"][task.value] = {
                "candidate_count": len(results),
                "success_count": sum(bool(result["success"]) for result in results),
                "results": results,
            }
            recipes.extend(task_recipes)
            print(
                f"{task.value}: {report['tasks'][task.value]['success_count']} / "
                f"{len(results)} Oracle candidates succeeded; "
                f"{len(task_recipes)} recipe(s) emitted"
            )
    finally:
        client.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_level_recipes(
        args.recipes_output,
        recipes,
        metadata={
            "oracle_report": str(args.output),
            "runtime_contract": report["runtime_contract"],
        },
    )
    print(f"Oracle report: {args.output}")
    print(f"Level recipes: {args.recipes_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
