from __future__ import annotations

import argparse
import atexit
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional
from importlib.metadata import PackageNotFoundError, version

from kof98_env import (
    COMBO_SCENARIOS,
    DEFAULT_COMBO_SCENARIO_NAME,
    IDLE_ACTION_ID,
    Kof98Env,
    TrainingProfile,
)


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def make_env(
    root: Path,
    combo_state_path: Optional[Path],
    fight_state_path: Optional[Path],
    training_profile: TrainingProfile,
    combo_scenario: str,
    action_repeat: int,
    seed: int,
    p2_training_ai: bool = False,
    hitbox_reward: bool = True,
    viewer: bool = False,
    viewer_scale: int = 3,
    viewer_fps: int = 30,
    viewer_speed: float = 1.0,
    viewer_hitboxes: bool = False,
    viewer_terminal_tail_frames: int = 90,
) -> Callable[[], Kof98Env]:
    def _init() -> Kof98Env:
        env = Kof98Env(
            dll_path=root / "build-vs2026-x64" / "Release" / "fbneo_training.dll",
            core_path=root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll",
            game_path=root / "roms" / "fbneo" / "kof98.zip",
            system_dir=root / "system",
            save_dir=root / "saves",
            combo_state_path=combo_state_path,
            fight_state_path=fight_state_path,
            action_repeat=action_repeat,
            hitbox_reward=hitbox_reward,
            p2_training_ai=p2_training_ai,
            training_profile=training_profile,
            combo_scenario=combo_scenario,
        )
        if viewer:
            env = TrainingViewerWrapper(
                env,
                viewer_scale,
                viewer_fps,
                viewer_speed,
                env.action_repeat,
                viewer_hitboxes,
                viewer_terminal_tail_frames,
            )
        env.reset(seed=seed)
        return env

    return _init


class TrainingViewerWrapper:
    def __init__(
        self,
        env: Kof98Env,
        scale: int,
        fps: int,
        speed: float,
        action_repeat: int,
        hitboxes: bool,
        terminal_tail_frames: int,
    ):
        from watch_kof98_ppo import Frame, FrameSink, InputHistory, OpenGlViewer

        self.env = env
        self.sink = FrameSink()
        self.input_history = InputHistory()
        self.startup_frame = Frame(pixels=bytes(320 * 224 * 2), width=320, height=224)
        self.viewer = OpenGlViewer(self.startup_frame.width, self.startup_frame.height, max(1, scale))
        self.viewer.draw(self.startup_frame)
        self.pygame = self.viewer.pygame
        self.clock = self.pygame.time.Clock()
        self.fps = max(1, fps)
        self.step_fps = max(1.0, self.fps * max(0.01, speed) / max(1, action_repeat))
        self.hitboxes = hitboxes
        self.terminal_tail_frames = max(0, terminal_tail_frames)
        self.enabled = True
        self.env.client.set_video_refresh_callback(self.sink.receive)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        self.input_history.clear()
        self._pump_viewer()
        return result

    def step(self, action):
        result = self.env.step(action)
        self._pump_viewer()
        if (result[2] or result[3]) and self.enabled:
            self._play_terminal_tail()
        return result

    def close(self) -> None:
        self.env.close()
        if self.enabled:
            self.enabled = False
            self.pygame.quit()

    def _pump_viewer(self) -> None:
        if not self.enabled:
            return

        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.enabled = False
                self.pygame.quit()
                return
            if event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_ESCAPE:
                self.enabled = False
                self.pygame.quit()
                return

        frame = self.sink.frame if self.sink.frame is not None else self.startup_frame
        overlay = None
        if self.hitboxes:
            from watch_kof98_ppo import build_hitbox_overlay_from_client

            overlay = build_hitbox_overlay_from_client(self.env.client, frame.width, frame.height)
        self.input_history.push(self.env.client.last_joypad())
        self.viewer.draw(frame, overlay, self.input_history.entries)
        self.clock.tick(self.step_fps)

    def _play_terminal_tail(self) -> None:
        for _ in range(self.terminal_tail_frames):
            if not self.enabled:
                return

            self.env.client.step(IDLE_ACTION_ID, 1)
            self._pump_viewer()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a KOF98 PPO agent through fbneo_training.dll.")
    parser.add_argument(
        "--root",
        type=Path,
        default=default_project_root(),
        help="qneogeo project root.",
    )
    parser.add_argument(
        "--combo-state",
        type=Path,
        default=None,
        help="Combo training save state. Defaults to saves/states/kof98.slot1.state.",
    )
    parser.add_argument(
        "--combo-scenario",
        action="append",
        default=None,
        metavar="NAME=STATE",
        help=(
            "Combo scenario and state assignment. Repeat to distribute Combo "
            "environments across multiple scenarios."
        ),
    )
    parser.add_argument(
        "--fight-state",
        type=Path,
        default=None,
        help="Fight training save state. Defaults to saves/states/kof98.slot2.state.",
    )
    parser.add_argument(
        "--profile",
        choices=("combo", "fight", "mixed"),
        default="combo",
        help="Training environment composition.",
    )
    parser.add_argument(
        "--combo-ratio",
        type=float,
        default=0.5,
        help="Fraction of parallel environments assigned to Combo in mixed mode.",
    )
    parser.add_argument(
        "--p2-training-ai",
        action="store_true",
        help="Enable the DLL P2 training AI in Fight environments.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total PPO training timesteps.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments. Start with 1, then increase after it is stable.",
    )
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=6,
        help="Fight profile setting retained for later. Combo training always uses its profile value of 1.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Stable-Baselines3 device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Model checkpoint directory.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from an existing PPO .zip model.",
    )
    parser.add_argument(
        "--save-name",
        default=None,
        help="Model file prefix. Defaults to kof98_<profile>_ppo.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open an OpenGL viewer while training. Requires --num-envs 1.",
    )
    parser.add_argument(
        "--viewer-scale",
        type=int,
        default=3,
        help="OpenGL viewer initial window scale.",
    )
    parser.add_argument(
        "--viewer-fps",
        type=int,
        default=60,
        help="Target emulated game FPS for training viewer throttling.",
    )
    parser.add_argument(
        "--viewer-speed",
        type=float,
        default=1.0,
        help="Training viewer playback multiplier. 1.0 is near real-time; 0.5 is half speed.",
    )
    parser.add_argument(
        "--viewer-hitboxes",
        action="store_true",
        help="Draw KOF98 hitboxes in the training OpenGL viewer.",
    )
    parser.add_argument(
        "--viewer-terminal-tail-frames",
        type=int,
        default=90,
        help="Play this many extra frames after terminal before the viewer resets.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Start TensorBoard together with training.",
    )
    parser.add_argument(
        "--tensorboard-port",
        type=int,
        default=6006,
        help="TensorBoard HTTP port.",
    )
    parser.add_argument(
        "--no-hitbox-reward",
        action="store_true",
        help="Ignored by the Combo profile, which always disables hitbox reward shaping.",
    )
    return parser.parse_args()


def validate_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def build_training_profiles(
    mode: str,
    num_envs: int,
    combo_ratio: float,
) -> list[TrainingProfile]:
    if num_envs < 1:
        raise ValueError("--num-envs must be at least 1")

    if mode == "combo":
        return [TrainingProfile.COMBO] * num_envs
    if mode == "fight":
        return [TrainingProfile.FIGHT] * num_envs
    if mode != "mixed":
        raise ValueError(f"Unknown training profile: {mode}")
    if num_envs < 2:
        raise ValueError("Mixed training requires --num-envs 2 or greater")
    if not 0.0 < combo_ratio < 1.0:
        raise ValueError("Mixed training requires --combo-ratio between 0 and 1")

    combo_env_count = int(num_envs * combo_ratio + 0.5)
    combo_env_count = max(1, min(num_envs - 1, combo_env_count))
    fight_env_count = num_envs - combo_env_count
    return (
        [TrainingProfile.COMBO] * combo_env_count
        + [TrainingProfile.FIGHT] * fight_env_count
    )


def resolve_combo_scenario_specs(
    root: Path,
    values: Optional[list[str]],
    default_state_path: Path,
) -> list[tuple[str, Path]]:
    if not values:
        return [(DEFAULT_COMBO_SCENARIO_NAME, default_state_path)]

    specs: list[tuple[str, Path]] = []
    for value in values:
        name, separator, state_text = value.partition("=")
        if not separator or not name or not state_text:
            raise ValueError(
                f"Invalid --combo-scenario '{value}'. Expected NAME=STATE."
            )
        if name not in COMBO_SCENARIOS:
            available = ", ".join(sorted(COMBO_SCENARIOS))
            raise ValueError(
                f"Unknown combo scenario '{name}'. Available: {available}"
            )

        state_path = Path(state_text)
        if not state_path.is_absolute():
            state_path = root / state_path
        specs.append((name, state_path))

    return specs


def stable_baselines3_major_version() -> int:
    try:
        text = version("stable-baselines3")
    except PackageNotFoundError:
        return 0

    major = text.split(".", 1)[0]
    return int(major) if major.isdigit() else 0


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_tensorboard(log_dir: Path, port: int) -> None:
    if is_port_open(port):
        print(f"TensorBoard already appears to be running: http://localhost:{port}/")
        return

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(log_dir),
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(log_dir.parent),
    )

    def stop_tensorboard() -> None:
        if process.poll() is None:
            process.terminate()

    atexit.register(stop_tensorboard)
    print(f"TensorBoard started: http://localhost:{port}/")


def close_vec_env_safely(env) -> None:
    try:
        env.close()
        return
    except (BrokenPipeError, EOFError, OSError) as error:
        processes = list(getattr(env, "processes", []))
        exit_codes = [process.exitcode for process in processes]
        print(
            f"Warning: vector environment was already closing ({error}). "
            f"Worker exit codes: {exit_codes}",
            file=sys.stderr,
        )

        for remote in getattr(env, "remotes", []):
            try:
                remote.close()
            except OSError:
                pass

        for process in processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

        env.closed = True


def main() -> int:
    args = parse_args()

    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as error:
        print("Install training dependencies first:", file=sys.stderr)
        print("  pip install stable-baselines3 sb3-contrib gymnasium tensorboard", file=sys.stderr)
        print(f"Import error: {error}", file=sys.stderr)
        return 1

    if stable_baselines3_major_version() < 2:
        print("stable-baselines3 2.x is required because Kof98Env uses the Gymnasium API.", file=sys.stderr)
        print("Upgrade in your KofAI environment:", file=sys.stderr)
        print("  python -m pip install -U \"stable-baselines3[extra]>=2.3.0\" \"sb3-contrib>=2.3.0\" gymnasium tensorboard", file=sys.stderr)
        return 1

    class MaskableSubprocVecEnv(SubprocVecEnv):
        def get_attr(self, attr_name, indices=None):
            if attr_name == "action_masks":
                # sb3-contrib 2.3 probes support with get_attr(), which tries to
                # pickle the bound method and its ctypes.CDLL-backed environment.
                return [True] * len(self._get_target_remotes(indices))

            return super().get_attr(attr_name, indices)

    class MaskableMonitor(Monitor):
        def action_masks(self):
            return self.env.action_masks()

    class TrainingMetricsCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.reset_rollout_metrics()

        def reset_rollout_metrics(self) -> None:
            self.step_count = 0
            self.combo_step_count = 0
            self.fight_step_count = 0
            self.p1_damage = 0.0
            self.p2_damage = 0.0
            self.p1_attack_overlap = 0.0
            self.p2_attack_overlap = 0.0
            self.p2_attack_pressure = 0.0
            self.guard_action = 0.0
            self.guard_success = 0.0
            self.super_available = 0.0
            self.super_without_stock = 0.0
            self.p2_airborne = 0.0
            self.oniyaki_anti_air_hit = 0.0
            self.attack_safety_pending = 0.0
            self.attack_safety_punished = 0.0
            self.attack_safety_unsafe_close = 0.0
            self.attack_safety_safe = 0.0
            self.p1_advanced_power_value = 0.0
            self.p1_advanced_power_stocks = 0.0
            self.p2_advanced_power_value = 0.0
            self.p2_advanced_power_stocks = 0.0
            self.p1_combo_count = 0.0
            self.max_p1_combo_count = 0.0
            self.action_14 = 0.0
            self.action_14_hit = 0.0
            self.action_15 = 0.0
            self.action_15_hit = 0.0
            self.action_16 = 0.0
            self.action_16_hit = 0.0
            self.action_17 = 0.0
            self.action_17_hit = 0.0
            self.action_18 = 0.0
            self.action_18_hit = 0.0
            self.action_19 = 0.0
            self.action_19_hit = 0.0
            self.action_20 = 0.0
            self.action_20_hit = 0.0
            self.action_21 = 0.0
            self.action_21_hit = 0.0
            self.action_22 = 0.0
            self.action_22_hit = 0.0
            self.action_23 = 0.0
            self.action_23_hit = 0.0
            self.action_24 = 0.0
            self.action_24_hit = 0.0
            self.action_25 = 0.0
            self.action_25_hit = 0.0
            self.action_26 = 0.0
            self.action_26_hit = 0.0
            self.distance_x_abs = 0.0
            self.reward_hp = 0.0
            self.reward_hitbox = 0.0
            self.reward_distance = 0.0
            self.reward_defense = 0.0
            self.reward_combo = 0.0
            self.reward_super = 0.0
            self.reward_anti_air = 0.0
            self.reward_safety = 0.0
            self.reward_fast_win = 0.0
            self.reward_time = 0.0
            self.combo_episodes = 0.0
            self.combo_successes = 0.0
            self.combo_episode_max_total = 0.0
            self.combo_scenario_metrics: dict[str, dict[str, float]] = {}
            self.reward_damage = 0.0
            self.reward_combo_milestone = 0.0
            self.reward_combo_target = 0.0
            self.reward_timeout = 0.0
            self.reward_ko_without_combo = 0.0
            self.input_ready = 0.0
            self.combo_phase = 0.0
            self.reward_phase = 0.0
            self.reward_complete = 0.0
            self.reward_phase_reset = 0.0

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            for index, info in enumerate(infos):
                self.step_count += 1
                training_profile = info.get("training_profile")
                is_combo_profile = training_profile == TrainingProfile.COMBO.value
                if is_combo_profile:
                    self.combo_step_count += 1
                    scenario_name = str(info.get("combo_scenario", "unknown"))
                    scenario_metrics = self.combo_scenario_metrics.setdefault(
                        scenario_name,
                        {
                            "episodes": 0.0,
                            "successes": 0.0,
                            "episode_max_total": 0.0,
                        },
                    )
                elif training_profile == TrainingProfile.FIGHT.value:
                    self.fight_step_count += 1
                self.p1_damage += float(info.get("p1_damage", 0.0))
                self.p2_damage += float(info.get("p2_damage", 0.0))
                self.p1_attack_overlap += float(info.get("p1_attack_overlap", 0.0))
                self.p2_attack_overlap += float(info.get("p2_attack_overlap", 0.0))
                self.p2_attack_pressure += float(info.get("p2_attack_pressure", 0.0))
                self.guard_action += float(info.get("guard_action", 0.0))
                self.guard_success += float(info.get("guard_success", 0.0))
                self.super_available += float(info.get("super_available", 0.0))
                self.super_without_stock += float(info.get("super_without_stock", 0.0))
                self.p2_airborne += float(info.get("p2_airborne", 0.0))
                self.oniyaki_anti_air_hit += float(info.get("oniyaki_anti_air_hit", 0.0))
                self.attack_safety_pending += float(info.get("attack_safety_pending", 0.0))
                self.attack_safety_punished += float(info.get("attack_safety_punished", 0.0))
                self.attack_safety_unsafe_close += float(info.get("attack_safety_unsafe_close", 0.0))
                self.attack_safety_safe += float(info.get("attack_safety_safe", 0.0))
                self.p1_advanced_power_value += float(info.get("p1_advanced_power_value", 0.0))
                self.p1_advanced_power_stocks += float(info.get("p1_advanced_power_stocks", 0.0))
                self.p2_advanced_power_value += float(info.get("p2_advanced_power_value", 0.0))
                self.p2_advanced_power_stocks += float(info.get("p2_advanced_power_stocks", 0.0))
                p1_combo_count = float(info.get("p1_combo_count", 0.0))
                self.p1_combo_count += p1_combo_count
                self.max_p1_combo_count = max(self.max_p1_combo_count, p1_combo_count)
                self.action_14 += float(info.get("action_14", 0.0))
                self.action_14_hit += float(info.get("action_14_hit", 0.0))
                self.action_15 += float(info.get("action_15", 0.0))
                self.action_15_hit += float(info.get("action_15_hit", 0.0))
                self.action_16 += float(info.get("action_16", 0.0))
                self.action_16_hit += float(info.get("action_16_hit", 0.0))
                self.action_17 += float(info.get("action_17", 0.0))
                self.action_17_hit += float(info.get("action_17_hit", 0.0))
                self.action_18 += float(info.get("action_18", 0.0))
                self.action_18_hit += float(info.get("action_18_hit", 0.0))
                self.action_19 += float(info.get("action_19", 0.0))
                self.action_19_hit += float(info.get("action_19_hit", 0.0))
                self.action_20 += float(info.get("action_20", 0.0))
                self.action_20_hit += float(info.get("action_20_hit", 0.0))
                self.action_21 += float(info.get("action_21", 0.0))
                self.action_21_hit += float(info.get("action_21_hit", 0.0))
                self.action_22 += float(info.get("action_22", 0.0))
                self.action_22_hit += float(info.get("action_22_hit", 0.0))
                self.action_23 += float(info.get("action_23", 0.0))
                self.action_23_hit += float(info.get("action_23_hit", 0.0))
                self.action_24 += float(info.get("action_24", 0.0))
                self.action_24_hit += float(info.get("action_24_hit", 0.0))
                self.action_25 += float(info.get("action_25", 0.0))
                self.action_25_hit += float(info.get("action_25_hit", 0.0))
                self.action_26 += float(info.get("action_26", 0.0))
                self.action_26_hit += float(info.get("action_26_hit", 0.0))
                self.distance_x_abs += float(info.get("distance_x_abs", 0.0))
                self.reward_hp += float(info.get("reward_hp", 0.0))
                self.reward_hitbox += float(info.get("reward_hitbox", 0.0))
                self.reward_distance += float(info.get("reward_distance", 0.0))
                self.reward_defense += float(info.get("reward_defense", 0.0))
                self.reward_combo += float(info.get("reward_combo", 0.0))
                self.reward_super += float(info.get("reward_super", 0.0))
                self.reward_anti_air += float(info.get("reward_anti_air", 0.0))
                self.reward_safety += float(info.get("reward_safety", 0.0))
                self.reward_fast_win += float(info.get("reward_fast_win", 0.0))
                self.reward_time += float(info.get("reward_time", 0.0))
                self.reward_damage += float(info.get("reward_damage", 0.0))
                self.reward_combo_milestone += float(info.get("reward_combo_milestone", 0.0))
                self.reward_combo_target += float(info.get("reward_combo_target", 0.0))
                self.reward_timeout += float(info.get("reward_timeout", 0.0))
                self.reward_ko_without_combo += float(info.get("reward_ko_without_combo", 0.0))
                if is_combo_profile:
                    self.input_ready += float(info.get("input_ready", 0.0))
                    self.combo_phase += float(info.get("combo_phase", 0.0))
                    self.reward_phase += float(info.get("reward_phase", 0.0))
                    self.reward_complete += float(info.get("reward_complete", 0.0))
                    self.reward_phase_reset += float(info.get("reward_phase_reset", 0.0))

                if is_combo_profile and index < len(dones) and dones[index]:
                    self.combo_episodes += 1.0
                    self.combo_successes += float(info.get("combo_success", 0.0))
                    self.combo_episode_max_total += float(info.get("episode_max_combo", 0.0))
                    scenario_metrics["episodes"] += 1.0
                    scenario_metrics["successes"] += float(info.get("combo_success", 0.0))
                    scenario_metrics["episode_max_total"] += float(info.get("episode_max_combo", 0.0))

            return True

        def _on_rollout_end(self) -> None:
            count = max(1, self.step_count)
            combo_count = max(1, self.combo_step_count)
            self.logger.record("kof/profile_combo_step_rate", self.combo_step_count / count)
            self.logger.record("kof/profile_fight_step_rate", self.fight_step_count / count)
            self.logger.record("kof/p1_damage_total", self.p1_damage)
            self.logger.record("kof/p2_damage_total", self.p2_damage)
            self.logger.record("kof/p1_attack_overlap_rate", self.p1_attack_overlap / count)
            self.logger.record("kof/p2_attack_overlap_rate", self.p2_attack_overlap / count)
            self.logger.record("kof/p2_attack_pressure_rate", self.p2_attack_pressure / count)
            self.logger.record("kof/guard_action_rate", self.guard_action / count)
            self.logger.record("kof/guard_success_rate", self.guard_success / max(1.0, self.p2_attack_pressure))
            self.logger.record("kof/super_available_rate", self.super_available / count)
            self.logger.record("kof/super_without_stock_total", self.super_without_stock)
            self.logger.record("kof/p2_airborne_rate", self.p2_airborne / count)
            self.logger.record("kof/oniyaki_anti_air_hit_total", self.oniyaki_anti_air_hit)
            self.logger.record("kof/oniyaki_anti_air_hit_rate", self.oniyaki_anti_air_hit / max(1.0, self.action_17))
            self.logger.record("kof/attack_safety_pending_rate", self.attack_safety_pending / count)
            self.logger.record("kof/attack_safety_punished_total", self.attack_safety_punished)
            self.logger.record("kof/attack_safety_unsafe_close_total", self.attack_safety_unsafe_close)
            self.logger.record("kof/attack_safety_safe_total", self.attack_safety_safe)
            self.logger.record("kof/mean_p1_advanced_power_value", self.p1_advanced_power_value / count)
            self.logger.record("kof/mean_p1_advanced_power_stocks", self.p1_advanced_power_stocks / count)
            self.logger.record("kof/mean_p2_advanced_power_value", self.p2_advanced_power_value / count)
            self.logger.record("kof/mean_p2_advanced_power_stocks", self.p2_advanced_power_stocks / count)
            self.logger.record("kof/mean_p1_combo_count", self.p1_combo_count / count)
            self.logger.record("kof/max_p1_combo_count", self.max_p1_combo_count)
            self.logger.record("kof/action_14_rate", self.action_14 / count)
            self.logger.record("kof/action_14_hit_total", self.action_14_hit)
            self.logger.record("kof/action_14_hit_rate", self.action_14_hit / max(1.0, self.action_14))
            self.logger.record("kof/action_15_rate", self.action_15 / count)
            self.logger.record("kof/action_15_hit_total", self.action_15_hit)
            self.logger.record("kof/action_15_hit_rate", self.action_15_hit / max(1.0, self.action_15))
            self.logger.record("kof/action_16_rate", self.action_16 / count)
            self.logger.record("kof/action_16_hit_total", self.action_16_hit)
            self.logger.record("kof/action_16_hit_rate", self.action_16_hit / max(1.0, self.action_16))
            self.logger.record("kof/action_17_rate", self.action_17 / count)
            self.logger.record("kof/action_17_hit_total", self.action_17_hit)
            self.logger.record("kof/action_17_hit_rate", self.action_17_hit / max(1.0, self.action_17))
            self.logger.record("kof/action_18_rate", self.action_18 / count)
            self.logger.record("kof/action_18_hit_total", self.action_18_hit)
            self.logger.record("kof/action_18_hit_rate", self.action_18_hit / max(1.0, self.action_18))
            self.logger.record("kof/action_19_rate", self.action_19 / count)
            self.logger.record("kof/action_19_hit_total", self.action_19_hit)
            self.logger.record("kof/action_19_hit_rate", self.action_19_hit / max(1.0, self.action_19))
            self.logger.record("kof/action_20_rate", self.action_20 / count)
            self.logger.record("kof/action_20_hit_total", self.action_20_hit)
            self.logger.record("kof/action_20_hit_rate", self.action_20_hit / max(1.0, self.action_20))
            self.logger.record("kof/action_21_rate", self.action_21 / count)
            self.logger.record("kof/action_21_hit_total", self.action_21_hit)
            self.logger.record("kof/action_21_hit_rate", self.action_21_hit / max(1.0, self.action_21))
            self.logger.record("kof/action_22_rate", self.action_22 / count)
            self.logger.record("kof/action_22_hit_total", self.action_22_hit)
            self.logger.record("kof/action_22_hit_rate", self.action_22_hit / max(1.0, self.action_22))
            self.logger.record("kof/action_23_rate", self.action_23 / count)
            self.logger.record("kof/action_23_hit_total", self.action_23_hit)
            self.logger.record("kof/action_23_hit_rate", self.action_23_hit / max(1.0, self.action_23))
            self.logger.record("kof/action_24_rate", self.action_24 / count)
            self.logger.record("kof/action_24_hit_total", self.action_24_hit)
            self.logger.record("kof/action_24_hit_rate", self.action_24_hit / max(1.0, self.action_24))
            self.logger.record("kof/action_25_rate", self.action_25 / count)
            self.logger.record("kof/action_25_hit_total", self.action_25_hit)
            self.logger.record("kof/action_25_hit_rate", self.action_25_hit / max(1.0, self.action_25))
            self.logger.record("kof/action_26_rate", self.action_26 / count)
            self.logger.record("kof/action_26_hit_total", self.action_26_hit)
            self.logger.record("kof/action_26_hit_rate", self.action_26_hit / max(1.0, self.action_26))
            self.logger.record("kof/mean_distance_x_abs", self.distance_x_abs / count)
            self.logger.record("kof/reward_hp_total", self.reward_hp)
            self.logger.record("kof/reward_hitbox_total", self.reward_hitbox)
            self.logger.record("kof/reward_distance_total", self.reward_distance)
            self.logger.record("kof/reward_defense_total", self.reward_defense)
            self.logger.record("kof/reward_combo_total", self.reward_combo)
            self.logger.record("kof/reward_super_total", self.reward_super)
            self.logger.record("kof/reward_anti_air_total", self.reward_anti_air)
            self.logger.record("kof/reward_safety_total", self.reward_safety)
            self.logger.record("kof/reward_fast_win_total", self.reward_fast_win)
            self.logger.record("kof/reward_time_total", self.reward_time)
            self.logger.record("kof/combo_episodes_total", self.combo_episodes)
            self.logger.record("kof/combo_success_rate", self.combo_successes / max(1.0, self.combo_episodes))
            self.logger.record("kof/combo_episode_max_mean", self.combo_episode_max_total / max(1.0, self.combo_episodes))
            for scenario_name, metrics in self.combo_scenario_metrics.items():
                episode_count = max(1.0, metrics["episodes"])
                self.logger.record(
                    f"kof_combo/{scenario_name}/success_rate",
                    metrics["successes"] / episode_count,
                )
                self.logger.record(
                    f"kof_combo/{scenario_name}/episode_max_mean",
                    metrics["episode_max_total"] / episode_count,
                )
            self.logger.record("kof/reward_damage_total", self.reward_damage)
            self.logger.record("kof/reward_combo_milestone_total", self.reward_combo_milestone)
            self.logger.record("kof/reward_combo_target_total", self.reward_combo_target)
            self.logger.record("kof/reward_timeout_total", self.reward_timeout)
            self.logger.record("kof/reward_ko_without_combo_total", self.reward_ko_without_combo)
            self.logger.record("kof/input_ready_rate", self.input_ready / combo_count)
            self.logger.record("kof/mean_combo_phase", self.combo_phase / combo_count)
            self.logger.record("kof/reward_phase_total", self.reward_phase)
            self.logger.record("kof/reward_complete_total", self.reward_complete)
            self.logger.record("kof/reward_phase_reset_total", self.reward_phase_reset)
            self.reset_rollout_metrics()

    root = args.root.resolve()
    if args.viewer and args.profile == "mixed":
        print("--viewer cannot display mixed parallel environments. Use combo or fight.", file=sys.stderr)
        return 2
    if args.viewer and args.num_envs != 1:
        print("--viewer requires --num-envs 1. Forcing num-envs to 1.", file=sys.stderr)
        args.num_envs = 1

    try:
        training_profiles = build_training_profiles(
            args.profile,
            args.num_envs,
            args.combo_ratio,
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    combo_state_path = args.combo_state
    if combo_state_path is None:
        combo_state_path = root / "saves" / "states" / "kof98.slot1.state"
    elif not combo_state_path.is_absolute():
        combo_state_path = root / combo_state_path
    try:
        combo_scenario_specs = resolve_combo_scenario_specs(
            root,
            args.combo_scenario,
            combo_state_path,
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    fight_state_path = args.fight_state
    if fight_state_path is None:
        fight_state_path = root / "saves" / "states" / "kof98.slot2.state"
    elif not fight_state_path.is_absolute():
        fight_state_path = root / fight_state_path

    validate_file(root / "build-vs2026-x64" / "Release" / "fbneo_training.dll", "fbneo_training.dll")
    validate_file(root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll", "fbneo_libretro.dll")
    validate_file(root / "roms" / "fbneo" / "kof98.zip", "kof98.zip")
    combo_env_count = training_profiles.count(TrainingProfile.COMBO)
    fight_env_count = training_profiles.count(TrainingProfile.FIGHT)
    if combo_env_count:
        for scenario_name, scenario_state_path in combo_scenario_specs:
            validate_file(scenario_state_path, f"Combo state for {scenario_name}")
            print(f"Combo scenario: {scenario_name} => {scenario_state_path}")
    if fight_env_count:
        validate_file(fight_state_path, "Fight save state")
        print(f"Fight profile state: {fight_state_path}")
    print(f"Training environments: Combo={combo_env_count}, Fight={fight_env_count}")

    if args.save_name is None:
        args.save_name = f"kof98_{args.profile}_ppo"

    log_dir = (args.log_dir or root / "ai_logs").resolve()
    save_dir = (args.save_dir or root / "trained_models").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.tensorboard:
        start_tensorboard(log_dir, args.tensorboard_port)

    environment_specs: list[tuple[TrainingProfile, str, Path]] = []
    combo_spec_index = 0
    for training_profile in training_profiles:
        if training_profile is TrainingProfile.COMBO:
            scenario_name, scenario_state_path = combo_scenario_specs[
                combo_spec_index % len(combo_scenario_specs)
            ]
            combo_spec_index += 1
        else:
            scenario_name, scenario_state_path = combo_scenario_specs[0]
        environment_specs.append(
            (training_profile, scenario_name, scenario_state_path)
        )

    def monitored_env(
        seed: int,
        training_profile: TrainingProfile,
        combo_scenario: str,
        scenario_state_path: Path,
    ) -> Callable:
        def _init():
            return MaskableMonitor(make_env(
                root=root,
                combo_state_path=scenario_state_path,
                fight_state_path=fight_state_path,
                training_profile=training_profile,
                combo_scenario=combo_scenario,
                action_repeat=args.action_repeat,
                seed=seed,
                p2_training_ai=args.p2_training_ai,
                hitbox_reward=not args.no_hitbox_reward,
                viewer=args.viewer,
                viewer_scale=args.viewer_scale,
                viewer_fps=args.viewer_fps,
                viewer_speed=args.viewer_speed,
                viewer_hitboxes=args.viewer_hitboxes,
                viewer_terminal_tail_frames=args.viewer_terminal_tail_frames,
            )())

        return _init

    env_fns = [
        monitored_env(seed, training_profile, combo_scenario, scenario_state_path)
        for seed, (training_profile, combo_scenario, scenario_state_path)
        in enumerate(environment_specs)
    ]
    if len(env_fns) == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = MaskableSubprocVecEnv(env_fns, start_method="spawn")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 31_250 // max(1, len(env_fns))),
        save_path=str(save_dir),
        name_prefix=args.save_name,
    )
    callback = CallbackList([checkpoint_callback, TrainingMetricsCallback()])

    if args.resume:
        resume_path = args.resume if args.resume.is_absolute() else root / args.resume
        validate_file(resume_path, "Resume model")
        model = MaskablePPO.load(
            str(resume_path),
            env=env,
            device=args.device,
            tensorboard_log=str(log_dir),
        )
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            device=args.device,
            verbose=1,
            n_steps=1024,
            batch_size=128,
            n_epochs=6,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=2.5e-4,
            ent_coef=0.01,
            clip_range=0.2,
            target_kl=0.03,
            tensorboard_log=str(log_dir),
        )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            reset_num_timesteps=args.resume is None,
            tb_log_name=args.save_name,
        )
        final_path = save_dir / f"{args.save_name}_final.zip"
        model.save(str(final_path))
        print(f"Saved final model: {final_path}")
    finally:
        close_vec_env_safely(env)

    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())
