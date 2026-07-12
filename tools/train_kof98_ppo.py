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

from kof98_env import Kof98Env


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def make_env(
    root: Path,
    state_path: Optional[Path],
    action_repeat: int,
    seed: int,
    hitbox_reward: bool = True,
    viewer: bool = False,
    viewer_scale: int = 3,
    viewer_fps: int = 30,
    viewer_speed: float = 1.0,
    viewer_hitboxes: bool = False,
) -> Callable[[], Kof98Env]:
    def _init() -> Kof98Env:
        env = Kof98Env(
            dll_path=root / "build-vs2026-x64" / "Release" / "fbneo_training.dll",
            core_path=root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll",
            game_path=root / "roms" / "fbneo" / "kof98.zip",
            system_dir=root / "system",
            save_dir=root / "saves",
            state_path=state_path,
            action_repeat=action_repeat,
            hitbox_reward=hitbox_reward,
        )
        if viewer:
            env = TrainingViewerWrapper(env, viewer_scale, viewer_fps, viewer_speed, action_repeat, viewer_hitboxes)
        env.reset(seed=seed)
        return env

    return _init


class TrainingViewerWrapper:
    def __init__(self, env: Kof98Env, scale: int, fps: int, speed: float, action_repeat: int, hitboxes: bool):
        from watch_kof98_ppo import Frame, FrameSink, OpenGlViewer

        self.env = env
        self.sink = FrameSink()
        self.startup_frame = Frame(pixels=bytes(320 * 224 * 2), width=320, height=224)
        self.viewer = OpenGlViewer(self.startup_frame.width, self.startup_frame.height, max(1, scale))
        self.viewer.draw(self.startup_frame)
        self.pygame = self.viewer.pygame
        self.clock = self.pygame.time.Clock()
        self.fps = max(1, fps)
        self.step_fps = max(1.0, self.fps * max(0.01, speed) / max(1, action_repeat))
        self.hitboxes = hitboxes
        self.enabled = True
        self.env.client.set_video_refresh_callback(self.sink.receive)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        self._pump_viewer()
        return result

    def step(self, action):
        result = self.env.step(action)
        self._pump_viewer()
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
        self.viewer.draw(frame, overlay)
        self.clock.tick(self.step_fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a KOF98 PPO agent through fbneo_training.dll.")
    parser.add_argument(
        "--root",
        type=Path,
        default=default_project_root(),
        help="qneogeo project root.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=None,
        help="Initial save state. Defaults to saves/states/kof98.slot1.state if it exists.",
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
        help="Frames to run for each action.",
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
        default="kof98_ppo",
        help="Model file prefix.",
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
        help="Disable per-step hitbox reward shaping for faster training.",
    )
    return parser.parse_args()


def validate_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


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


def main() -> int:
    args = parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as error:
        print("Install training dependencies first:", file=sys.stderr)
        print("  pip install stable-baselines3 gymnasium tensorboard", file=sys.stderr)
        print(f"Import error: {error}", file=sys.stderr)
        return 1

    if stable_baselines3_major_version() < 2:
        print("stable-baselines3 2.x is required because Kof98Env uses the Gymnasium API.", file=sys.stderr)
        print("Upgrade in your KofAI environment:", file=sys.stderr)
        print("  python -m pip install -U \"stable-baselines3[extra]>=2.3.0\" gymnasium tensorboard", file=sys.stderr)
        return 1

    class TrainingMetricsCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.reset_rollout_metrics()

        def reset_rollout_metrics(self) -> None:
            self.step_count = 0
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
            self.distance_x_abs = 0.0
            self.reward_hp = 0.0
            self.reward_hitbox = 0.0
            self.reward_distance = 0.0
            self.reward_defense = 0.0
            self.reward_combo = 0.0
            self.reward_super = 0.0
            self.reward_anti_air = 0.0
            self.reward_safety = 0.0
            self.reward_time = 0.0

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                self.step_count += 1
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
                self.distance_x_abs += float(info.get("distance_x_abs", 0.0))
                self.reward_hp += float(info.get("reward_hp", 0.0))
                self.reward_hitbox += float(info.get("reward_hitbox", 0.0))
                self.reward_distance += float(info.get("reward_distance", 0.0))
                self.reward_defense += float(info.get("reward_defense", 0.0))
                self.reward_combo += float(info.get("reward_combo", 0.0))
                self.reward_super += float(info.get("reward_super", 0.0))
                self.reward_anti_air += float(info.get("reward_anti_air", 0.0))
                self.reward_safety += float(info.get("reward_safety", 0.0))
                self.reward_time += float(info.get("reward_time", 0.0))

            return True

        def _on_rollout_end(self) -> None:
            count = max(1, self.step_count)
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
            self.logger.record("kof/mean_distance_x_abs", self.distance_x_abs / count)
            self.logger.record("kof/reward_hp_total", self.reward_hp)
            self.logger.record("kof/reward_hitbox_total", self.reward_hitbox)
            self.logger.record("kof/reward_distance_total", self.reward_distance)
            self.logger.record("kof/reward_defense_total", self.reward_defense)
            self.logger.record("kof/reward_combo_total", self.reward_combo)
            self.logger.record("kof/reward_super_total", self.reward_super)
            self.logger.record("kof/reward_anti_air_total", self.reward_anti_air)
            self.logger.record("kof/reward_safety_total", self.reward_safety)
            self.logger.record("kof/reward_time_total", self.reward_time)
            self.reset_rollout_metrics()

    root = args.root.resolve()
    state_path = args.state
    if state_path is None:
        default_state = root / "saves" / "states" / "kof98.slot1.state"
        state_path = default_state if default_state.exists() else None
    elif not state_path.is_absolute():
        state_path = root / state_path

    validate_file(root / "build-vs2026-x64" / "Release" / "fbneo_training.dll", "fbneo_training.dll")
    validate_file(root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll", "fbneo_libretro.dll")
    validate_file(root / "roms" / "fbneo" / "kof98.zip", "kof98.zip")
    if state_path is not None:
        validate_file(state_path, "Initial save state")

    if args.viewer and args.num_envs != 1:
        print("--viewer requires --num-envs 1. Forcing num-envs to 1.", file=sys.stderr)
        args.num_envs = 1

    log_dir = (args.log_dir or root / "ai_logs").resolve()
    save_dir = (args.save_dir or root / "trained_models").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.tensorboard:
        start_tensorboard(log_dir, args.tensorboard_port)

    def monitored_env(seed: int) -> Callable:
        def _init():
            return Monitor(make_env(
                root,
                state_path,
                args.action_repeat,
                seed,
                not args.no_hitbox_reward,
                args.viewer,
                args.viewer_scale,
                args.viewer_fps,
                args.viewer_speed,
                args.viewer_hitboxes,
            )())

        return _init

    env_fns = [monitored_env(seed) for seed in range(max(1, args.num_envs))]
    if len(env_fns) == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns, start_method="spawn")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 31_250 // max(1, len(env_fns))),
        save_path=str(save_dir),
        name_prefix=args.save_name,
    )
    callback = CallbackList([checkpoint_callback, TrainingMetricsCallback()])

    if args.resume:
        resume_path = args.resume if args.resume.is_absolute() else root / args.resume
        validate_file(resume_path, "Resume model")
        model = PPO.load(str(resume_path), env=env, device=args.device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            device=args.device,
            verbose=1,
            n_steps=1024,
            batch_size=128,
            n_epochs=4,
            gamma=0.97,
            gae_lambda=0.95,
            learning_rate=2.5e-4,
            tensorboard_log=str(log_dir),
        )

    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        final_path = save_dir / f"{args.save_name}_final.zip"
        model.save(str(final_path))
        print(f"Saved final model: {final_path}")
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())
