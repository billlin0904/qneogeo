"""Deterministic held-out evaluation for KOF98 Physical Fight policies."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from kof98_env import (
    ACTION_COUNT,
    STEP_EVENT_ACTION_STARTED,
    STEP_EVENT_COMBO_HIT,
    ActionMaskLevel,
    FightCurriculum,
    FightRewardVersion,
    Kof98Env,
    P2Style,
    TrainingProfile,
)
from kof98_observation import (
    OBSERVATION_SCHEMA_V2_ID,
    OBSERVATION_SCHEMA_V3_ID,
    OBSERVATION_V1_SIZE,
    OBSERVATION_V2_SIZE,
    ObservationVersion,
)


ATTACK_ACTION_MIN = 6
BLOCK_CONFIRM_WINDOW_FRAMES = 60


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a KOF98 PPO model in held-out Physical fights.",
    )
    parser.add_argument("--root", type=Path, default=default_project_root())
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("saves/states/kof98.slot2.state"),
    )
    parser.add_argument("--episodes-per-style", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4098)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--disable-observation-event-features",
        action="store_true",
        help="Evaluate the V3-neutral B arm.",
    )
    parser.add_argument(
        "--single-style",
        choices=[style.value for style in P2Style],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


@dataclass
class EvaluationStats:
    episodes: int = 0
    wins: int = 0
    time_limit_episodes: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    hp_differential: float = 0.0
    max_combo_total: float = 0.0
    combo_4plus_episodes: int = 0
    guard_opportunities: float = 0.0
    guard_successes: float = 0.0
    post_block_opportunities: float = 0.0
    post_block_successes: float = 0.0
    starter_hit_opportunities: float = 0.0
    hit_followup_successes: float = 0.0
    starter_blocked_opportunities: float = 0.0
    blocked_stop_successes: float = 0.0
    free_decisions: float = 0.0
    free_action_counts: list[float] = field(
        default_factory=lambda: [0.0] * ACTION_COUNT
    )

    def add(self, other: "EvaluationStats") -> None:
        for name in (
            "episodes",
            "wins",
            "time_limit_episodes",
            "damage_dealt",
            "damage_taken",
            "hp_differential",
            "max_combo_total",
            "combo_4plus_episodes",
            "guard_opportunities",
            "guard_successes",
            "post_block_opportunities",
            "post_block_successes",
            "starter_hit_opportunities",
            "hit_followup_successes",
            "starter_blocked_opportunities",
            "blocked_stop_successes",
            "free_decisions",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.free_action_counts = [
            left + right
            for left, right in zip(
                self.free_action_counts,
                other.free_action_counts,
            )
        ]

    def summary(self) -> dict:
        episodes = max(1, self.episodes)
        guard_opportunities = max(1.0, self.guard_opportunities)
        post_block_opportunities = max(1.0, self.post_block_opportunities)
        starter_hit_opportunities = max(1.0, self.starter_hit_opportunities)
        starter_blocked_opportunities = max(
            1.0,
            self.starter_blocked_opportunities,
        )
        hit_followup_rate = (
            self.hit_followup_successes / starter_hit_opportunities
        )
        blocked_stop_rate = (
            self.blocked_stop_successes / starter_blocked_opportunities
        )
        free_decisions = max(1.0, self.free_decisions)
        action_rates = [
            count / free_decisions for count in self.free_action_counts
        ]
        top_actions = sorted(
            enumerate(action_rates),
            key=lambda item: item[1],
            reverse=True,
        )[:8]
        return {
            "episodes": self.episodes,
            "win_rate": self.wins / episodes,
            "damage_dealt_mean": self.damage_dealt / episodes,
            "damage_taken_mean": self.damage_taken / episodes,
            "hp_differential_mean": self.hp_differential / episodes,
            "max_combo_mean": self.max_combo_total / episodes,
            "combo_4plus_episode_rate": self.combo_4plus_episodes / episodes,
            "pressure_opportunities": self.guard_opportunities,
            "manual_guard_successes": self.guard_successes,
            "manual_guard_given_pressure": (
                self.guard_successes / guard_opportunities
            ),
            "post_block_opportunities": self.post_block_opportunities,
            "post_block_successes": self.post_block_successes,
            "post_block_response_hit_rate": (
                self.post_block_successes / post_block_opportunities
            ),
            "starter_hit_opportunities": self.starter_hit_opportunities,
            "hit_followup_successes": self.hit_followup_successes,
            "followup_given_starter_hit": hit_followup_rate,
            "starter_blocked_opportunities": self.starter_blocked_opportunities,
            "blocked_stop_successes": self.blocked_stop_successes,
            "stop_given_starter_blocked": blocked_stop_rate,
            "confirm_discrimination": 0.5 * (
                hit_followup_rate + blocked_stop_rate
            ),
            "free_decisions": self.free_decisions,
            "top_free_actions": [
                {"action": action_id, "rate": rate}
                for action_id, rate in top_actions
            ],
        }


def resolve_observation_version(model) -> ObservationVersion:
    shape = tuple(model.observation_space.shape)
    if shape == (OBSERVATION_V1_SIZE,):
        return ObservationVersion.V1
    if shape != (OBSERVATION_V2_SIZE,):
        raise ValueError(f"Unsupported observation shape: {shape}")
    schema = getattr(model, "kof_observation_schema_id", OBSERVATION_SCHEMA_V2_ID)
    if schema == OBSERVATION_SCHEMA_V3_ID:
        return ObservationVersion.V3
    if schema == OBSERVATION_SCHEMA_V2_ID:
        return ObservationVersion.V2
    raise ValueError(f"Unsupported observation schema: {schema!r}")


def evaluate_style(
    model,
    *,
    root: Path,
    state_path: Path,
    style: P2Style,
    observation_version: ObservationVersion,
    event_features: bool,
    episodes: int,
    max_steps: int,
    action_repeat: int,
    seed: int,
) -> EvaluationStats:
    env = Kof98Env(
        dll_path=root / "build-vs2026-x64" / "Release" / "fbneo_training.dll",
        core_path=root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll",
        game_path=root / "roms" / "fbneo" / "kof98.zip",
        system_dir=root / "system",
        save_dir=root / "saves",
        fight_state_path=state_path,
        action_repeat=action_repeat,
        hitbox_reward=False,
        p2_training_ai=True,
        p2_style=style,
        fight_curriculum=FightCurriculum.NONE,
        training_profile=TrainingProfile.FIGHT,
        action_mask_level=ActionMaskLevel.PHYSICAL,
        observation_version=observation_version,
        observation_event_features=event_features,
        fight_reward_version=FightRewardVersion.SYMMETRIC_TACTICAL_V3,
    )
    stats = EvaluationStats()
    try:
        for episode_index in range(episodes):
            obs, _ = env.reset(seed=seed + episode_index)
            episode_damage_dealt = 0.0
            episode_damage_taken = 0.0
            episode_max_combo = 0.0
            pending_block_deadlines: list[int] = []
            seen_counter_windows: set[int] = set()
            successful_counter_windows: set[int] = set()

            for _step_index in range(max_steps):
                action_mask = env.action_masks()
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                    action_masks=action_mask,
                )
                action_id = int(np.asarray(action).item())
                obs, _reward, terminated, truncated, info = env.step(action_id)

                episode_damage_dealt += float(info.get("p2_damage", 0.0))
                episode_damage_taken += float(info.get("p1_damage", 0.0))
                episode_max_combo = max(
                    episode_max_combo,
                    float(info.get("p1_combo_count", 0.0)),
                )
                stats.guard_opportunities += float(
                    info.get("tactical_guard_opportunity", 0.0)
                )
                stats.guard_successes += float(
                    info.get("tactical_guard_success", 0.0)
                )
                counter_window_start = int(
                    info.get("defense_counter_window_start_frame", 0.0)
                )
                counter_window_deadline = int(
                    info.get("defense_counter_window_deadline_frame", 0.0)
                )
                counter_action_serial = int(
                    info.get("defense_counter_action_serial", 0.0)
                )
                if (
                    counter_window_start > 0
                    and counter_window_start not in seen_counter_windows
                ):
                    seen_counter_windows.add(counter_window_start)
                    stats.post_block_opportunities += 1.0

                counter_hit = any(
                    int(event.event_type) == STEP_EVENT_COMBO_HIT
                    and int(event.source_player) == 1
                    and int(event.action_serial) == counter_action_serial
                    and counter_action_serial > 0
                    and counter_window_start
                    <= int(event.absolute_engine_frame)
                    < counter_window_deadline
                    for event in env.last_step_events
                )
                if (
                    counter_hit
                    and counter_window_start not in successful_counter_windows
                ):
                    successful_counter_windows.add(counter_window_start)
                    stats.post_block_successes += 1.0
                stats.starter_hit_opportunities += float(
                    info.get("step_starter_hit_count", 0.0)
                )
                stats.hit_followup_successes += float(
                    info.get("tactical_confirm_success", 0.0)
                )

                engine_frame = int(env.client.combat_timing_state().engine_frame)
                blocked_count = int(info.get("step_starter_blocked_count", 0.0))
                for _ in range(blocked_count):
                    pending_block_deadlines.append(
                        engine_frame + BLOCK_CONFIRM_WINDOW_FRAMES
                    )
                    stats.starter_blocked_opportunities += 1.0

                p1_attack_started_frames = [
                    int(event.absolute_engine_frame)
                    for event in env.last_step_events
                    if int(event.event_type) == STEP_EVENT_ACTION_STARTED
                    and int(event.source_player) == 1
                    and int(event.action_id) >= ATTACK_ACTION_MIN
                ]
                remaining_deadlines = []
                for deadline in pending_block_deadlines:
                    attacked_early = any(
                        started_frame < deadline
                        for started_frame in p1_attack_started_frames
                    )
                    if attacked_early:
                        continue
                    if engine_frame >= deadline:
                        stats.blocked_stop_successes += 1.0
                    else:
                        remaining_deadlines.append(deadline)
                pending_block_deadlines = remaining_deadlines

                if float(info.get("free_decision", 0.0)) > 0.0:
                    stats.free_decisions += 1.0
                    if 0 <= action_id < ACTION_COUNT:
                        stats.free_action_counts[action_id] += 1.0

                if terminated or truncated:
                    break
            else:
                info = dict(info)
                info["fight_outcome"] = ""
                stats.time_limit_episodes += 1

            stats.episodes += 1
            if str(info.get("fight_outcome", "")) in ("win_ko", "win_timeout"):
                stats.wins += 1
            stats.damage_dealt += episode_damage_dealt
            stats.damage_taken += episode_damage_taken
            stats.hp_differential += (
                float(info.get("p1_health", 0.0))
                - float(info.get("p2_health", 0.0))
            )
            stats.max_combo_total += episode_max_combo
            if episode_max_combo >= 4.0:
                stats.combo_4plus_episodes += 1
    finally:
        env.close()
    return stats


def write_result(
    result: dict,
    *,
    output: Path | None,
    root: Path,
    print_json: bool,
) -> None:
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if print_json:
        print(rendered, flush=True)
    if output is not None:
        output_path = output if output.is_absolute() else root / output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        if print_json:
            print(f"Saved evaluation: {output_path}", flush=True)


def run_isolated_evaluation(
    args: argparse.Namespace,
    *,
    root: Path,
    model_path: Path,
    state_path: Path,
) -> int:
    aggregate = EvaluationStats()
    by_style: dict[str, dict] = {}
    observation_version = ""
    event_features = not args.disable_observation_event_features

    with tempfile.TemporaryDirectory(prefix="kof98_eval_") as temp_dir:
        temp_root = Path(temp_dir)
        for style_index, style in enumerate(P2Style):
            partial_path = temp_root / f"{style.value}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--root",
                str(root),
                "--model",
                str(model_path),
                "--state",
                str(state_path),
                "--episodes-per-style",
                str(args.episodes_per_style),
                "--max-steps",
                str(args.max_steps),
                "--action-repeat",
                str(args.action_repeat),
                "--seed",
                str(args.seed + style_index * 10_000),
                "--device",
                args.device,
                "--single-style",
                style.value,
                "--output",
                str(partial_path),
            ]
            if args.disable_observation_event_features:
                command.append("--disable-observation-event-features")

            print(f"Evaluating {style.value}...", flush=True)
            completed = None
            for attempt in range(2):
                completed = subprocess.run(command, cwd=root, check=False)
                if completed.returncode == 0:
                    break
                print(
                    f"Retrying {style.value} after evaluator exit "
                    f"{completed.returncode} (attempt {attempt + 2}/2)...",
                    flush=True,
                )
            if completed is None or completed.returncode != 0:
                return int(completed.returncode if completed is not None else 1)

            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            raw_stats = EvaluationStats(**partial["raw_stats"])
            aggregate.add(raw_stats)
            by_style[style.value] = partial["summary"]
            observation_version = str(partial["observation_version"])

    result = {
        "model": str(model_path),
        "state": str(state_path),
        "seed": args.seed,
        "episodes_per_style": args.episodes_per_style,
        "observation_version": observation_version,
        "event_features": event_features,
        "raw_stats": asdict(aggregate),
        "aggregate": aggregate.summary(),
        "styles": by_style,
    }
    write_result(result, output=args.output, root=root, print_json=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.episodes_per_style <= 0:
        raise ValueError("--episodes-per-style must be greater than zero")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be greater than zero")

    root = args.root.resolve()
    model_path = args.model if args.model.is_absolute() else root / args.model
    state_path = args.state if args.state.is_absolute() else root / args.state
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not state_path.is_file():
        raise FileNotFoundError(f"State not found: {state_path}")

    if args.single_style is None:
        return run_isolated_evaluation(
            args,
            root=root,
            model_path=model_path,
            state_path=state_path,
        )

    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(model_path), device=args.device)
    observation_version = resolve_observation_version(model)
    event_features = not args.disable_observation_event_features
    if (
        args.disable_observation_event_features
        and observation_version is not ObservationVersion.V3
    ):
        raise ValueError("Event feature gating requires a V3 model")

    style = P2Style(args.single_style)
    stats = evaluate_style(
        model,
        root=root,
        state_path=state_path,
        style=style,
        observation_version=observation_version,
        event_features=event_features,
        episodes=args.episodes_per_style,
        max_steps=args.max_steps,
        action_repeat=args.action_repeat,
        seed=args.seed,
    )

    result = {
        "model": str(model_path),
        "state": str(state_path),
        "seed": args.seed,
        "episodes_per_style": args.episodes_per_style,
        "observation_version": observation_version.value,
        "event_features": event_features,
        "style": style.value,
        "raw_stats": asdict(stats),
        "summary": stats.summary(),
    }
    write_result(result, output=args.output, root=root, print_json=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
