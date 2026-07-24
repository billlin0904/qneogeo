"""Run the paired StrategyV4 B/C pilot from one shared checkpoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PILOT_SEEDS = (98, 198, 298)


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired B/C StrategyV4 pilots sequentially.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_project_root(),
        help="qneogeo project root.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Training timesteps per arm and seed.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Stable-Baselines3 device.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Ask the first run to start TensorBoard on port 6006.",
    )
    parser.add_argument(
        "--eval-episodes-per-style",
        type=int,
        default=20,
        help="Held-out Physical episodes per P2 style after each arm.",
    )
    parser.add_argument(
        "--experiment-tag",
        default="v4b_directionfix",
        help="Name segment used to keep independent pilot revisions separate.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when an arm's final model already exists.",
    )
    return parser.parse_args()


def pilot_save_name(
    arm: str,
    seed: int,
    timesteps: int,
    experiment_tag: str,
) -> str:
    field_name = "neutral" if arm == "b" else "event"
    return (
        f"kof98_strategy_{experiment_tag}_{arm}_v3_{field_name}_seed{seed}_"
        f"pilot{timesteps // 1000}k_ppo"
    )


def pilot_command(
    root: Path,
    *,
    arm: str,
    seed: int,
    timesteps: int,
    device: str,
    start_tensorboard: bool,
    experiment_tag: str,
) -> list[str]:
    save_name = pilot_save_name(arm, seed, timesteps, experiment_tag)
    command = [
        sys.executable,
        str(root / "tools" / "train_kof98_ppo.py"),
        "--timesteps",
        str(timesteps),
        "--resume",
        "trained_models\\kof98_strategy_v4a_shared_v3_seed98_final.zip",
        "--observation-version",
        "v3",
        "--fight-reward-version",
        "symmetric_tactical_rm_v3",
        "--profile",
        "mixed",
        "--preset",
        "repeat4",
        "--combo-suite",
        "kyo29",
        "--combo-ratio",
        "0.3",
        "--level-recipe-bank",
        "ai_logs\\oracle\\kof98_v3c_recipes.json",
        "--level-recipe-envs",
        "2",
        "--mask-level",
        "guided",
        "--fight-state",
        "saves\\states\\kof98.slot2.state",
        "--p2-training-ai",
        "--p2-style",
        "oniyaki",
        "--p2-style",
        "guard",
        "--p2-style",
        "jump_in",
        "--p2-style",
        "poke",
        "--num-envs",
        "10",
        "--seed",
        str(seed),
        "--save-name",
        save_name,
        "--tensorboard-run-name",
        save_name,
        "--checkpoint-every",
        "25000",
        "--relative-checkpoints",
        "--no-hitbox-reward",
        "--device",
        device,
    ]
    if arm == "b":
        command.append("--disable-observation-event-features")
    if start_tensorboard:
        command.extend(("--tensorboard", "--tensorboard-port", "6006"))
    return command


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be greater than zero")

    shared_checkpoint = (
        root
        / "trained_models"
        / "kof98_strategy_v4a_shared_v3_seed98_final.zip"
    )
    if not shared_checkpoint.is_file():
        raise FileNotFoundError(f"Shared checkpoint not found: {shared_checkpoint}")

    first_run = True
    for seed in PILOT_SEEDS:
        for arm in ("b", "c"):
            save_name = pilot_save_name(
                arm,
                seed,
                args.timesteps,
                args.experiment_tag,
            )
            final_model = root / "trained_models" / f"{save_name}_final.zip"
            if final_model.is_file() and not args.force:
                print(f"Skipping completed arm: {final_model}", flush=True)
                first_run = False
                continue

            command = pilot_command(
                root,
                arm=arm,
                seed=seed,
                timesteps=args.timesteps,
                device=args.device,
                start_tensorboard=bool(args.tensorboard and first_run),
                experiment_tag=args.experiment_tag,
            )
            print(
                f"\n=== StrategyV4 arm={arm.upper()} seed={seed} "
                f"timesteps={args.timesteps} ===",
                flush=True,
            )
            completed = subprocess.run(command, cwd=root, check=False)
            if completed.returncode != 0:
                return int(completed.returncode)

            evaluation_output = (
                root
                / "ai_logs"
                / "evaluations"
                / f"{save_name}.json"
            )
            evaluation_command = [
                sys.executable,
                str(root / "tools" / "evaluate_kof98_physical.py"),
                "--model",
                str(final_model),
                "--state",
                "saves\\states\\kof98.slot2.state",
                "--episodes-per-style",
                str(args.eval_episodes_per_style),
                "--seed",
                str(10_000 + seed),
                "--device",
                args.device,
                "--output",
                str(evaluation_output),
            ]
            if arm == "b":
                evaluation_command.append(
                    "--disable-observation-event-features"
                )
            evaluation = subprocess.run(
                evaluation_command,
                cwd=root,
                check=False,
            )
            if evaluation.returncode != 0:
                return int(evaluation.returncode)
            first_run = False

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
