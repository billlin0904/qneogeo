"""KOF98 PPO 訓練入口:環境編組、超參數 preset、指標記錄、run manifest。

訓練流水線(每階段 resume 上一階段的模型):
    strict combo → guided combo → (physical combo, 已因觀測混疊結案)
    → mixed(combo 留存 + fight 實戰)→ reward/curriculum 迭代

Mixed 模式的環境編組(--num-envs N, --combo-ratio R):
    Combo 環境  = N×R 個，每次 reset 以錯開的 round-robin 輪替全部
                 --combo-scenario，因此 4 個 env 也能覆蓋 11 套課表。
    Fight 環境  = 其餘。依序配置舊 combo-route teacher、可重複指定的
                 targeted curriculum，剩下的才是真正 physical 實戰；
                 指定 --targeted-fight-envs 時，Targeted curriculum 與
                 Physical P2 style 都會在 reset 時輪替，降低 FBNeo 記憶體用量。

可重現性:每次 run 寫出 run_manifest_*.json(git commit + dirty、
套件版本、DLL/ROM/state 的 SHA-256、全部生效參數)。比較實驗時
應以「相同模擬幀數」為準,不是相同 timesteps(見 kof/emulated_frames)。
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from kof98_env import (
    ACTION_COUNT,
    ActionMaskLevel,
    COMBO_SCENARIOS,
    DEFAULT_COMBO_SCENARIO_NAME,
    FightCurriculum,
    FightRewardVersion,
    IDLE_ACTION_ID,
    Kof98Env,
    KofEnvClient,
    P2Style,
    TrainingProfile,
)
from kof98_model_transfer import (
    assert_legacy_policy_equivalence,
    assert_v2_v3_policy_equivalence,
    transplant_policy_observation_inputs,
    transplant_v2_policy_to_v3,
)
from kof98_curriculum import LevelRecipe, load_level_recipes
from kof98_observation import (
    OBSERVATION_V1_SIZE,
    OBSERVATION_V2_SIZE,
    OBSERVATION_SCHEMA_V2_ID,
    ObservationVersion,
    observation_schema_id,
    observation_size,
)

# Hyperparameters were tuned at action_repeat=6. The repeat4 preset keeps the
# per-emulated-frame semantics: gamma = 0.99^(4/6), gae_lambda = 0.95^(4/6),
# and n_steps scaled so one rollout covers the same emulated time.
TRAINING_PRESETS = {
    "repeat6": {"action_repeat": 6, "gamma": 0.99, "gae_lambda": 0.95, "n_steps": 1024},
    "repeat4": {"action_repeat": 4, "gamma": 0.99331, "gae_lambda": 0.96638, "n_steps": 1536},
}

FIGHT_OUTCOME_NAMES = (
    "win_ko",
    "loss_ko",
    "win_timeout",
    "loss_timeout",
    "draw_timeout",
    "draw_ko",
)
TACTICAL_ACTION_IDS = (1, 2, 3, 4, 5, 8, 11, 16)
TACTICAL_CONDITION_TAGS = {
    "guard": "guard_given_pressure",
    "anti_air": "oniyaki_hit_given_airborne",
    "approach": "safe_entry_given_far",
    "confirm": "followup_hit_given_confirm",
}

DEFAULT_GUIDED_FIGHT_SCENARIOS = (
    "kyo_close_c_seventy_five_shiki_kai_red_kick",
    "kyo_close_c_seventy_five_shiki_kai_kototsuki",
    "kyo_close_c_seventy_five_shiki_kai_aragami",
    "kyo_forward_b_red_kick",
)

KYO29_COMBO_SUITE = (
    ("kyo_corner_dokugami", 1),
    ("kyo_forward_b_kototsuki", 3),
    ("kyo_forward_b_orochinagi", 1),
    ("kyo_forward_b_red_kick", 1),
    ("kyo_forward_b_aragami", 1),
    ("kyo_close_c_seventy_five_shiki_kai_orochinagi", 1),
    ("kyo_close_c_seventy_five_shiki_kai_red_kick", 1),
    ("kyo_close_c_seventy_five_shiki_kai_kototsuki", 3),
    ("kyo_close_c_seventy_five_shiki_kai_aragami", 1),
    ("kyo_corner_seventy_five_shiki_kai_aragami_chain", 1),
    ("kyo_crouch_b_crouch_a_mushiki", 1),
)


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# 單一環境工廠。SubprocVecEnv 用 spawn 啟動子行程,每個子行程各載一份
# fbneo_training.dll(完整模擬器實例),因此環境數受記憶體/CPU 限制。
def make_env(
    root: Path,
    combo_state_path: Optional[Path],
    fight_state_path: Optional[Path],
    training_profile: TrainingProfile,
    combo_scenario: str,
    action_mask_level: ActionMaskLevel,
    action_repeat: int,
    seed: int,
    combo_scenario_rotation: Optional[list[tuple[str, Path]]] = None,
    combo_rotation_offset: int = 0,
    combo_rotation_stride: int = 1,
    p2_training_ai: bool = False,
    p2_style: P2Style = P2Style.ONIYAKI,
    fight_guided: bool = False,
    fight_curriculum: FightCurriculum = FightCurriculum.NONE,
    fight_rotation: Optional[list[tuple[FightCurriculum, P2Style]]] = None,
    fight_rotation_offset: int = 0,
    fight_rotation_stride: int = 1,
    hitbox_reward: bool = True,
    viewer: bool = False,
    viewer_scale: int = 3,
    viewer_fps: int = 30,
    viewer_speed: float = 1.0,
    viewer_hitboxes: bool = False,
    viewer_terminal_tail_frames: int = 90,
    observation_version: ObservationVersion = ObservationVersion.V1,
    observation_event_features: bool = True,
    fight_reward_version: FightRewardVersion | None = None,
    level_recipe_rotation: Optional[list[LevelRecipe]] = None,
    level_recipe_rotation_offset: int = 0,
    level_recipe_rotation_stride: int = 1,
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
            p2_style=p2_style,
            fight_guided=fight_guided,
            fight_curriculum=fight_curriculum,
            fight_rotation=fight_rotation,
            fight_rotation_offset=fight_rotation_offset,
            fight_rotation_stride=fight_rotation_stride,
            training_profile=training_profile,
            combo_scenario=combo_scenario,
            combo_scenario_rotation=combo_scenario_rotation,
            combo_rotation_offset=combo_rotation_offset,
            combo_rotation_stride=combo_rotation_stride,
            action_mask_level=action_mask_level,
            observation_version=observation_version,
            observation_event_features=observation_event_features,
            fight_reward_version=fight_reward_version,
            level_recipe_rotation=level_recipe_rotation,
            level_recipe_rotation_offset=level_recipe_rotation_offset,
            level_recipe_rotation_stride=level_recipe_rotation_stride,
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
        "--combo-suite",
        choices=("single", "kyo29"),
        default="single",
        help="Named combo scenario/state bundle. Explicit --combo-scenario entries take precedence.",
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
        "--guided-fight-envs",
        type=int,
        default=0,
        help=(
            "Number of Fight environments that expose a combo-route action "
            "mask while keeping the real Fight state, opponent and rewards."
        ),
    )
    parser.add_argument(
        "--guided-fight-scenario",
        action="append",
        choices=tuple(COMBO_SCENARIOS),
        default=None,
        help=(
            "Combo route assigned to Guided Fight environments. Repeat to "
            "cycle multiple routes."
        ),
    )
    parser.add_argument(
        "--targeted-fight-curriculum",
        action="append",
        choices=tuple(
            curriculum.value
            for curriculum in FightCurriculum
            if curriculum not in (FightCurriculum.NONE, FightCurriculum.COMBO_ROUTE)
        ),
        default=None,
        help=(
            "Targeted Fight curriculum assigned to the first Fight environments. "
            "Repeat to add defense, anti_air, approach and hit_confirm teachers."
        ),
    )
    parser.add_argument(
        "--targeted-fight-envs",
        type=int,
        default=None,
        help=(
            "Resident Targeted Fight worker count. When set, the curricula "
            "from --targeted-fight-curriculum rotate on reset instead of "
            "requiring one FBNeo process per curriculum."
        ),
    )
    parser.add_argument(
        "--level-recipe-bank",
        type=Path,
        default=None,
        help=(
            "V3 reverse-curriculum Level Recipe JSON. Recipe workers replay "
            "safe states with deterministic P2 preludes and always use the "
            "full physical action mask."
        ),
    )
    parser.add_argument(
        "--level-recipe-envs",
        type=int,
        default=None,
        help=(
            "Fight worker count assigned to --level-recipe-bank. Defaults to "
            "the smaller of recipe count and available Fight workers."
        ),
    )
    parser.add_argument(
        "--mask-level",
        choices=tuple(level.value for level in ActionMaskLevel),
        default=ActionMaskLevel.STRICT.value,
        help="Combo curriculum mask. Fight always uses physical legality.",
    )
    parser.add_argument(
        "--p2-training-ai",
        action="store_true",
        help="Enable the DLL P2 training AI in Fight environments.",
    )
    parser.add_argument(
        "--p2-style",
        action="append",
        choices=tuple(style.value for style in P2Style),
        default=None,
        help=(
            "P2 behavior styles for Fight environments. They are assigned "
            "round-robin normally and rotate on reset when "
            "--targeted-fight-envs enables resident-worker rotation."
        ),
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
        "--preset",
        choices=tuple(TRAINING_PRESETS),
        default="repeat6",
        help="Bundled action_repeat/gamma/gae_lambda/n_steps. repeat6 is the reproducible baseline.",
    )
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=None,
        help="Fight profile frames per decision. Defaults to the preset value. Combo training always uses 1.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="PPO discount factor. Defaults to the preset value.",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=None,
        help="PPO GAE lambda. Defaults to the preset value.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="PPO rollout length per environment. Defaults to the preset value.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Stable-Baselines3 device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=98,
        help="Random seed used by PPO and vector environments (default: 98).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--tensorboard-run-name",
        default=None,
        help=(
            "TensorBoard run name. Defaults to a timestamped name so resumed "
            "training never merges events with an older run."
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Model checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=31_250,
        help=(
            "Checkpoint interval in aggregate PPO timesteps "
            "(default: 31250)."
        ),
    )
    parser.add_argument(
        "--relative-checkpoints",
        action="store_true",
        help=(
            "Name checkpoints by timesteps elapsed in this run and preserve "
            "a step-0 copy. Intended for paired pilot experiments."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from an existing PPO .zip model.",
    )
    parser.add_argument(
        "--allow-action-set-migration",
        action="store_true",
        help=(
            "Explicitly accept a checkpoint created before the current DLL "
            "action-set/schema metadata. Use only for an intentional fine-tune."
        ),
    )
    parser.add_argument(
        "--observation-version",
        choices=tuple(version.value for version in ObservationVersion),
        default=ObservationVersion.V1.value,
        help=(
            "Observation ABI. v1 is 26 values, v2 is the legacy 140-value "
            "strategy state, and v3 reuses 140 values for generic timing/events."
        ),
    )
    parser.add_argument(
        "--disable-observation-event-features",
        action="store_true",
        help=(
            "Keep V3's 32 repurposed event/timing columns at zero. This is "
            "the StrategyV4-A migration control group, not a deployment mode."
        ),
    )
    parser.add_argument(
        "--fight-reward-version",
        choices=("auto",) + tuple(version.value for version in FightRewardVersion),
        default="auto",
        help="Fight objective. auto selects legacy for v1 and symmetric HP/outcome for v2.",
    )
    parser.add_argument(
        "--migrate-from",
        type=Path,
        default=None,
        help=(
            "Create V2 from a V1 checkpoint, or V3 from a V2 checkpoint. "
            "Both migrations preserve policy output before fine-tuning."
        ),
    )
    parser.add_argument(
        "--teacher-model",
        type=Path,
        default=None,
        help="Legacy teacher used for temporary kickstart policy regularisation.",
    )
    parser.add_argument(
        "--teacher-weight",
        type=float,
        default=0.10,
        help="Initial kickstart cross-entropy weight.",
    )
    parser.add_argument(
        "--teacher-decay-steps",
        type=int,
        default=2_000_000,
        help="Linearly decay teacher regularisation to zero over this many steps.",
    )
    parser.add_argument(
        "--teacher-batch-size",
        type=int,
        default=2048,
        help="Maximum rollout samples used for each teacher update.",
    )
    parser.add_argument(
        "--oracle-teacher-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary recipe-Oracle imitation weight. It keeps the full "
            "physical mask and decays to zero; use only with a level bank."
        ),
    )
    parser.add_argument(
        "--oracle-teacher-decay-steps",
        type=int,
        default=600_000,
        help="Linearly decay recipe-Oracle imitation over this many steps.",
    )
    parser.add_argument(
        "--oracle-teacher-batch-size",
        type=int,
        default=2048,
        help="Maximum recipe observations used per auxiliary update.",
    )
    parser.add_argument(
        "--oracle-teacher-updates",
        type=int,
        default=4,
        help="Auxiliary Oracle gradient updates after each PPO rollout.",
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


def file_sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def git_command_output(root: Path, *arguments: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def package_version_or_none(name: str) -> Optional[str]:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


# Run manifest:實驗的身分證。工作樹常是 dirty 狀態,單靠 commit hash
# 無法重現 —— 所以連檔案雜湊、套件版本、生效參數全部落盤。
def write_run_manifest(
    manifest_path: Path,
    args: argparse.Namespace,
    root: Path,
    effective: dict,
    manifest_files: dict[str, Path],
) -> None:
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = None

    porcelain = git_command_output(root, "status", "--porcelain")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "effective": effective,
        "git": {
            "commit": git_command_output(root, "rev-parse", "HEAD"),
            "dirty": None if porcelain is None else bool(porcelain),
        },
        "versions": {
            "python": sys.version,
            "torch": torch_version,
            "stable-baselines3": package_version_or_none("stable-baselines3"),
            "sb3-contrib": package_version_or_none("sb3-contrib"),
            "gymnasium": package_version_or_none("gymnasium"),
            "numpy": package_version_or_none("numpy"),
        },
        "files": {
            label: {"path": str(path), "sha256": file_sha256(path)}
            for label, path in manifest_files.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# 依 --profile/--combo-ratio 決定每個平行環境是 COMBO 還是 FIGHT。
# combo_env_count = int(N*R+0.5),夾在 [1, N-1]。
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
    observation_version = ObservationVersion(args.observation_version)
    fight_reward_version = (
        None
        if args.fight_reward_version == "auto"
        else FightRewardVersion(args.fight_reward_version)
    )
    effective_fight_reward_version = fight_reward_version or (
        FightRewardVersion.SYMMETRIC_V2
        if observation_version is not ObservationVersion.V1
        else FightRewardVersion.LEGACY_COMBO4
    )
    if args.resume is not None and args.migrate_from is not None:
        print("Use either --resume or --migrate-from, not both.", file=sys.stderr)
        return 2
    if (
        args.migrate_from is not None
        and observation_version is ObservationVersion.V1
    ):
        print(
            "--migrate-from requires --observation-version v2 or v3.",
            file=sys.stderr,
        )
        return 2
    if (
        args.teacher_model is not None
        and observation_version is ObservationVersion.V1
    ):
        print(
            "--teacher-model requires --observation-version v2 or v3.",
            file=sys.stderr,
        )
        return 2

    print("Loading Stable-Baselines3 and training dependencies...", flush=True)
    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from kof98_kickstart import KickstartTeacherCallback
        from kof98_oracle_teacher import OracleCurriculumCallback
    except ImportError as error:
        print("Install training dependencies first:", file=sys.stderr)
        print("  pip install stable-baselines3 sb3-contrib gymnasium tensorboard", file=sys.stderr)
        print(f"Import error: {error}", file=sys.stderr)
        return 1
    print("Training dependencies loaded.", flush=True)

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

    class RelativeCheckpointCallback(BaseCallback):
        """Save run-relative 0/N checkpoints even when resuming a model."""

        def __init__(
            self,
            interval_timesteps: int,
            total_timesteps: int,
            save_path: Path,
            name_prefix: str,
        ):
            super().__init__()
            self.interval_timesteps = max(1, int(interval_timesteps))
            self.total_timesteps = max(0, int(total_timesteps))
            self.save_path = Path(save_path)
            self.name_prefix = str(name_prefix)
            self.start_num_timesteps = 0
            self.next_checkpoint = self.interval_timesteps

        def _save_relative_checkpoint(self, relative_timesteps: int) -> None:
            checkpoint_path = self.save_path / (
                f"{self.name_prefix}_pilot_"
                f"{int(relative_timesteps):06d}_steps.zip"
            )
            self.model.save(str(checkpoint_path))
            print(f"Saved relative checkpoint: {checkpoint_path}", flush=True)

        def _on_training_start(self) -> None:
            self.start_num_timesteps = int(self.model.num_timesteps)
            self.next_checkpoint = self.interval_timesteps
            self._save_relative_checkpoint(0)

        def _on_step(self) -> bool:
            relative_timesteps = (
                int(self.model.num_timesteps) - self.start_num_timesteps
            )
            while (
                self.next_checkpoint <= self.total_timesteps
                and relative_timesteps >= self.next_checkpoint
            ):
                self._save_relative_checkpoint(self.next_checkpoint)
                self.next_checkpoint += self.interval_timesteps
            return True

    class TrainingMetricsCallback(BaseCallback):
        """自訂 TensorBoard 指標。命名空間:

        kof/          全體共用(reward 分項總和、模擬幀數累計…)
        kof_combo/<scenario>/   各連段課程的成功率/替代率
        kof_fight_physical/     自由實戰(真正的考試成績)
        kof_fight_guided/       teacher 引導環境(上課成績,不可混讀!)
        kof_fight_targeted/     防禦/對空/接近/hit-confirm 局部課程
        kof_fight_*/<style>/    再按 P2 風格細分(勝率、4plus、teacher 完成數)

        讀圖注意:
        - per-rollout 局數很少(個位數),單點劇烈跳動是小樣本雜訊,
          看 last-N 平均或平滑曲線。
        - action rate 的分母要用 available_steps(mask 開放時),
          不是總步數 —— 否則不同 action_repeat 的 run 無法互比。
        - 每 rollout 結束歸零(reset_rollout_metrics);cumulative_* 例外。
        """

        def __init__(self, fight_assignments: list[dict]):
            super().__init__()
            self.active_mode_styles: dict[str, set[str]] = {
                "guided": set(),
                "targeted": set(),
                "physical": set(),
            }
            self.active_targeted_curricula: set[str] = set()
            for assignment in fight_assignments:
                mode = str(assignment["mode"])
                style = str(assignment["style"])
                if mode in self.active_mode_styles:
                    self.active_mode_styles[mode].add(style)
                if mode == "targeted":
                    self.active_targeted_curricula.add(
                        str(assignment["curriculum"]),
                    )
            self.cumulative_frames = 0.0
            self.cumulative_fight_frames = 0.0
            self.cumulative_mode_tactical_metrics = {
                mode: {
                    condition: {"opportunities": 0.0, "successes": 0.0}
                    for condition in TACTICAL_CONDITION_TAGS
                }
                for mode, styles in self.active_mode_styles.items()
                if styles
            }
            self.cumulative_mode_approach_progress = {
                mode: 0.0
                for mode, styles in self.active_mode_styles.items()
                if styles
            }
            self.cumulative_mode_approach_forward_frames = {
                mode: 0.0
                for mode, styles in self.active_mode_styles.items()
                if styles
            }
            self.env_fight_max_combo: dict[int, float] = {}
            self.env_fight_damage_dealt: dict[int, float] = {}
            self.env_fight_damage_taken: dict[int, float] = {}
            self.reset_rollout_metrics()

        @staticmethod
        def _empty_event_metrics() -> dict[str, float]:
            return {
                "steps": 0.0,
                "frames": 0.0,
                "p1_reaction_active": 0.0,
                "p1_reaction_remaining_valid": 0.0,
                "p2_reaction_active": 0.0,
                "p2_reaction_remaining_valid": 0.0,
                "defense_non_neutral": 0.0,
                "confirm_non_neutral": 0.0,
                "block_contact": 0.0,
                "clean_hit": 0.0,
                "blockstun_end": 0.0,
                "starter_hit": 0.0,
                "starter_blocked": 0.0,
            }

        def reset_rollout_metrics(self) -> None:
            self.step_count = 0
            self.combo_step_count = 0
            self.fight_step_count = 0
            self.frames_total = 0.0
            self.fight_frames_total = 0.0
            self.fight_free_decision_steps = 0.0
            self.fight_queue_decision_steps = 0.0
            self.fight_forced_idle_steps = 0.0
            self.fight_action_available = np.zeros(ACTION_COUNT, dtype=np.float64)
            self.fight_action_selected = np.zeros(ACTION_COUNT, dtype=np.float64)
            self.fight_followup_queued = np.zeros(ACTION_COUNT, dtype=np.float64)
            self.fight_followup_started = np.zeros(ACTION_COUNT, dtype=np.float64)
            self.fight_followup_hit = np.zeros(ACTION_COUNT, dtype=np.float64)
            self.fight_episodes = 0.0
            self.fight_outcome_counts = {name: 0.0 for name in FIGHT_OUTCOME_NAMES}
            self.fight_episode_max_combo_total = 0.0
            self.fight_combo_2plus_episodes = 0.0
            self.fight_combo_4plus_episodes = 0.0
            self.fight_episode_damage_dealt_total = 0.0
            self.fight_episode_damage_taken_total = 0.0
            self.fight_mode_metrics = {
                mode: {
                    "episodes": 0.0,
                    "wins": 0.0,
                    "combo_4plus": 0.0,
                    "episode_max_total": 0.0,
                    "teacher_completions": 0.0,
                }
                for mode, styles in self.active_mode_styles.items()
                if styles
            }
            self.fight_mode_action_available = {
                mode: np.zeros(ACTION_COUNT, dtype=np.float64)
                for mode in self.fight_mode_metrics
            }
            self.fight_mode_action_selected = {
                mode: np.zeros(ACTION_COUNT, dtype=np.float64)
                for mode in self.fight_mode_metrics
            }
            self.fight_mode_free_decision_steps = {
                mode: 0.0
                for mode in self.fight_mode_metrics
            }
            self.fight_mode_free_action_selected = {
                mode: np.zeros(ACTION_COUNT, dtype=np.float64)
                for mode in self.fight_mode_metrics
            }
            self.fight_style_metrics = {
                mode: {
                    style: {
                        "episodes": 0.0,
                        "wins": 0.0,
                        "combo_4plus": 0.0,
                        "episode_max_total": 0.0,
                        "teacher_completions": 0.0,
                    }
                    for style in styles
                }
                for mode, styles in self.active_mode_styles.items()
                if styles
            }
            self.fight_curriculum_metrics = {
                curriculum: {
                    "opportunities": 0.0,
                    "successes": 0.0,
                    "guard_successes": 0.0,
                    "counter_opportunities": 0.0,
                    "counter_successes": 0.0,
                    "reward_machine_successes": 0.0,
                    "reward_machine_failures": 0.0,
                    "reward_machine_timeouts": 0.0,
                    "reward_total": 0.0,
                    "episodes": 0.0,
                    "wins": 0.0,
                    "combo_4plus": 0.0,
                    "episode_max_total": 0.0,
                }
                for curriculum in self.active_targeted_curricula
            }
            self.level_recipe_metrics: dict[str, dict[str, float]] = {}
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
            self.action_27 = 0.0
            self.action_27_hit = 0.0
            self.action_28 = 0.0
            self.action_28_hit = 0.0
            self.distance_x_abs = 0.0
            self.reward_hp = 0.0
            self.reward_hitbox = 0.0
            self.reward_distance = 0.0
            self.reward_defense = 0.0
            self.reward_combo = 0.0
            self.reward_fight_combo_4plus_milestone = 0.0
            self.reward_super = 0.0
            self.reward_anti_air = 0.0
            self.reward_safety = 0.0
            self.reward_cancel = 0.0
            self.reward_fast_win = 0.0
            self.reward_outcome = 0.0
            self.reward_time = 0.0
            self.reward_curriculum = 0.0
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
            self.reward_wrong_action = 0.0
            self.reward_alternate = 0.0
            self.combo_alternate_successes = 0.0
            self.runtime_step_event_dropped = 0.0
            self.runtime_batch_epoch_mismatch = 0.0
            self.runtime_chip_hit_block_conflict = 0.0
            self.runtime_nonfinite_observation = 0.0
            self.runtime_event_feature_enabled_steps = 0.0
            self.runtime_event_feature_nonzero_steps = 0.0
            self.runtime_event_feature_nonzero_total = 0.0
            self.fight_event_metrics = self._empty_event_metrics()
            self.fight_mode_event_metrics = {
                mode: self._empty_event_metrics()
                for mode in self.fight_mode_metrics
            }

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            for index, info in enumerate(infos):
                self.step_count += 1
                frame_count = float(info.get("frame_count", 0.0))
                self.frames_total += frame_count
                self.cumulative_frames += frame_count
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
                            "alternate_successes": 0.0,
                            "episode_max_total": 0.0,
                        },
                    )
                elif training_profile == TrainingProfile.FIGHT.value:
                    curriculum_name = str(
                        info.get("fight_curriculum", FightCurriculum.NONE.value)
                    )
                    if curriculum_name == FightCurriculum.COMBO_ROUTE.value:
                        fight_mode = "guided"
                    elif curriculum_name != FightCurriculum.NONE.value:
                        fight_mode = "targeted"
                    else:
                        fight_mode = "physical"
                    p2_style = str(info.get("p2_style", P2Style.ONIYAKI.value))
                    if p2_style not in self.fight_style_metrics[fight_mode]:
                        p2_style = P2Style.ONIYAKI.value
                    teacher_completion = float(
                        info.get("fight_teacher_complete", 0.0)
                    )
                    self.fight_mode_metrics[fight_mode]["teacher_completions"] += (
                        teacher_completion
                    )
                    self.fight_style_metrics[fight_mode][p2_style][
                        "teacher_completions"
                    ] += teacher_completion
                    curriculum_metrics = self.fight_curriculum_metrics.get(
                        curriculum_name
                    )
                    recipe_name = str(info.get("level_recipe", ""))
                    recipe_metrics = None
                    if recipe_name:
                        recipe_metrics = self.level_recipe_metrics.setdefault(
                            recipe_name,
                            {
                                "episodes": 0.0,
                                "successes": 0.0,
                                "failures": 0.0,
                                "timeouts": 0.0,
                                "reward_total": 0.0,
                                "frames_total": 0.0,
                            },
                        )
                        recipe_metrics["successes"] += float(
                            info.get("reward_machine_success", 0.0)
                        )
                        recipe_metrics["failures"] += float(
                            info.get("reward_machine_failure", 0.0)
                        )
                        recipe_metrics["timeouts"] += float(
                            info.get("reward_machine_timeout", 0.0)
                        )
                        recipe_metrics["reward_total"] += float(
                            info.get("reward_curriculum", 0.0)
                        )
                        recipe_metrics["frames_total"] += frame_count
                    if curriculum_metrics is not None:
                        curriculum_metrics["opportunities"] += float(
                            info.get("curriculum_opportunity", 0.0)
                        )
                        curriculum_metrics["successes"] += float(
                            info.get("curriculum_success", 0.0)
                        )
                        curriculum_metrics["guard_successes"] += float(
                            info.get("curriculum_guard_success", 0.0)
                        )
                        curriculum_metrics["counter_opportunities"] += float(
                            info.get("curriculum_counter_opportunity", 0.0)
                        )
                        curriculum_metrics["counter_successes"] += float(
                            info.get("curriculum_counter_success", 0.0)
                        )
                        curriculum_metrics["reward_machine_successes"] += float(
                            info.get("reward_machine_success", 0.0)
                        )
                        curriculum_metrics["reward_machine_failures"] += float(
                            info.get("reward_machine_failure", 0.0)
                        )
                        curriculum_metrics["reward_machine_timeouts"] += float(
                            info.get("reward_machine_timeout", 0.0)
                        )
                        curriculum_metrics["reward_total"] += float(
                            info.get("reward_curriculum", 0.0)
                        )
                    self.fight_step_count += 1
                    self.fight_frames_total += frame_count
                    self.cumulative_fight_frames += frame_count
                    for event_metrics in (
                        self.fight_event_metrics,
                        self.fight_mode_event_metrics[fight_mode],
                    ):
                        event_metrics["steps"] += 1.0
                        event_metrics["frames"] += frame_count
                        for metric_name in (
                            "p1_reaction_active",
                            "p1_reaction_remaining_valid",
                            "p2_reaction_active",
                            "p2_reaction_remaining_valid",
                            "defense_non_neutral",
                            "confirm_non_neutral",
                        ):
                            event_metrics[metric_name] += float(
                                info.get(metric_name, 0.0)
                            )
                        event_metrics["block_contact"] += float(
                            info.get("step_block_contact_count", 0.0)
                        )
                        event_metrics["clean_hit"] += float(
                            info.get("step_clean_hit_count", 0.0)
                        )
                        event_metrics["blockstun_end"] += float(
                            info.get("step_blockstun_ended_count", 0.0)
                        )
                        event_metrics["starter_hit"] += float(
                            info.get("step_starter_hit_count", 0.0)
                        )
                        event_metrics["starter_blocked"] += float(
                            info.get("step_starter_blocked_count", 0.0)
                        )
                    free_decision = float(info.get("free_decision", 0.0))
                    self.fight_free_decision_steps += free_decision
                    self.fight_mode_free_decision_steps[fight_mode] += free_decision
                    self.fight_queue_decision_steps += float(info.get("queue_decision", 0.0))
                    self.fight_forced_idle_steps += float(info.get("forced_idle", 0.0))
                    availability = info.get("action_availability")
                    if availability is not None:
                        availability_array = np.asarray(
                            availability,
                            dtype=np.float64,
                        )
                        self.fight_action_available += availability_array
                        self.fight_mode_action_available[
                            fight_mode
                        ] += availability_array
                    selected_action = int(info.get("action", -1))
                    if 0 <= selected_action < ACTION_COUNT:
                        self.fight_action_selected[selected_action] += 1.0
                        self.fight_mode_action_selected[
                            fight_mode
                        ][selected_action] += 1.0
                        if free_decision > 0.0:
                            self.fight_mode_free_action_selected[
                                fight_mode
                            ][selected_action] += free_decision
                    tactical_metrics = self.cumulative_mode_tactical_metrics[
                        fight_mode
                    ]
                    for condition in TACTICAL_CONDITION_TAGS:
                        tactical_metrics[condition]["opportunities"] += float(
                            info.get(f"tactical_{condition}_opportunity", 0.0)
                        )
                        tactical_metrics[condition]["successes"] += float(
                            info.get(f"tactical_{condition}_success", 0.0)
                        )
                    self.cumulative_mode_approach_progress[fight_mode] += float(
                        info.get("tactical_approach_step_progress", 0.0)
                    )
                    self.cumulative_mode_approach_forward_frames[fight_mode] += float(
                        info.get("tactical_approach_step_forward_frames", 0.0)
                    )
                    queued_actions = info.get("queued_followup_actions")
                    started_actions = info.get("started_followup_actions")
                    hit_actions = info.get("hit_followup_actions")
                    if queued_actions is None or started_actions is None or hit_actions is None:
                        followup_action = int(info.get("followup_action", -1.0))
                        queued_actions = (
                            [followup_action]
                            if info.get("queued_followup", 0.0)
                            else []
                        )
                        started_actions = (
                            [followup_action]
                            if info.get("started_followup", 0.0)
                            else []
                        )
                        hit_actions = (
                            [followup_action]
                            if info.get("followup_hit", 0.0)
                            else []
                        )
                    for followup_action in queued_actions:
                        if 0 <= int(followup_action) < ACTION_COUNT:
                            self.fight_followup_queued[int(followup_action)] += 1.0
                    for followup_action in started_actions:
                        if 0 <= int(followup_action) < ACTION_COUNT:
                            self.fight_followup_started[int(followup_action)] += 1.0
                    for followup_action in hit_actions:
                        if 0 <= int(followup_action) < ACTION_COUNT:
                            self.fight_followup_hit[int(followup_action)] += 1.0
                    fight_combo_count = float(info.get("p1_combo_count", 0.0))
                    self.env_fight_max_combo[index] = max(
                        self.env_fight_max_combo.get(index, 0.0),
                        fight_combo_count,
                    )
                    self.env_fight_damage_dealt[index] = (
                        self.env_fight_damage_dealt.get(index, 0.0)
                        + float(info.get("p2_damage", 0.0))
                    )
                    self.env_fight_damage_taken[index] = (
                        self.env_fight_damage_taken.get(index, 0.0)
                        + float(info.get("p1_damage", 0.0))
                    )
                    if index < len(dones) and dones[index]:
                        if recipe_metrics is not None:
                            recipe_metrics["episodes"] += 1.0
                        self.fight_episodes += 1.0
                        outcome = str(info.get("fight_outcome", ""))
                        if outcome in self.fight_outcome_counts:
                            self.fight_outcome_counts[outcome] += 1.0
                        episode_max_combo = self.env_fight_max_combo.pop(index, 0.0)
                        mode_metrics = self.fight_mode_metrics[fight_mode]
                        style_metrics = self.fight_style_metrics[fight_mode][p2_style]
                        mode_metrics["episodes"] += 1.0
                        mode_metrics["episode_max_total"] += episode_max_combo
                        style_metrics["episodes"] += 1.0
                        style_metrics["episode_max_total"] += episode_max_combo
                        if outcome in ("win_ko", "win_timeout"):
                            mode_metrics["wins"] += 1.0
                            style_metrics["wins"] += 1.0
                        if episode_max_combo >= 4.0:
                            mode_metrics["combo_4plus"] += 1.0
                            style_metrics["combo_4plus"] += 1.0
                        if curriculum_metrics is not None:
                            curriculum_metrics["episodes"] += 1.0
                            curriculum_metrics["episode_max_total"] += episode_max_combo
                            if outcome in ("win_ko", "win_timeout"):
                                curriculum_metrics["wins"] += 1.0
                            if episode_max_combo >= 4.0:
                                curriculum_metrics["combo_4plus"] += 1.0
                        self.fight_episode_max_combo_total += episode_max_combo
                        if episode_max_combo >= 2.0:
                            self.fight_combo_2plus_episodes += 1.0
                        if episode_max_combo >= 4.0:
                            self.fight_combo_4plus_episodes += 1.0
                        self.fight_episode_damage_dealt_total += self.env_fight_damage_dealt.pop(index, 0.0)
                        self.fight_episode_damage_taken_total += self.env_fight_damage_taken.pop(index, 0.0)
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
                self.action_27 += float(info.get("action_27", 0.0))
                self.action_27_hit += float(info.get("action_27_hit", 0.0))
                self.action_28 += float(info.get("action_28", 0.0))
                self.action_28_hit += float(info.get("action_28_hit", 0.0))
                self.distance_x_abs += float(info.get("distance_x_abs", 0.0))
                self.reward_hp += float(info.get("reward_hp", 0.0))
                self.reward_hitbox += float(info.get("reward_hitbox", 0.0))
                self.reward_distance += float(info.get("reward_distance", 0.0))
                self.reward_defense += float(info.get("reward_defense", 0.0))
                self.reward_combo += float(info.get("reward_combo", 0.0))
                self.reward_fight_combo_4plus_milestone += float(
                    info.get("reward_fight_combo_4plus_milestone", 0.0)
                )
                self.reward_super += float(info.get("reward_super", 0.0))
                self.reward_anti_air += float(info.get("reward_anti_air", 0.0))
                self.reward_safety += float(info.get("reward_safety", 0.0))
                self.reward_cancel += float(info.get("reward_cancel", 0.0))
                self.reward_fast_win += float(info.get("reward_fast_win", 0.0))
                self.reward_outcome += float(info.get("reward_outcome", 0.0))
                self.reward_time += float(info.get("reward_time", 0.0))
                self.reward_curriculum += float(
                    info.get("reward_curriculum", 0.0)
                )
                self.reward_damage += float(info.get("reward_damage", 0.0))
                self.reward_combo_milestone += float(info.get("reward_combo_milestone", 0.0))
                self.reward_combo_target += float(info.get("reward_combo_target", 0.0))
                self.reward_timeout += float(info.get("reward_timeout", 0.0))
                self.reward_ko_without_combo += float(info.get("reward_ko_without_combo", 0.0))
                self.runtime_step_event_dropped += float(
                    info.get("step_event_dropped", 0.0)
                )
                self.runtime_batch_epoch_mismatch += float(
                    info.get("step_event_batch_epoch_mismatch", 0.0)
                )
                self.runtime_chip_hit_block_conflict += float(
                    info.get("step_event_chip_hit_block_conflict", 0.0)
                )
                if "observation_finite" in info:
                    self.runtime_nonfinite_observation += float(
                        not bool(info["observation_finite"])
                    )
                event_features_enabled = float(
                    info.get("observation_event_features_enabled", 0.0)
                )
                event_feature_nonzero_count = float(
                    info.get("observation_event_nonzero_count", 0.0)
                )
                self.runtime_event_feature_enabled_steps += (
                    event_features_enabled
                )
                self.runtime_event_feature_nonzero_steps += float(
                    event_feature_nonzero_count > 0.0
                )
                self.runtime_event_feature_nonzero_total += (
                    event_feature_nonzero_count
                )
                if is_combo_profile:
                    self.input_ready += float(info.get("input_ready", 0.0))
                    self.combo_phase += float(info.get("combo_phase", 0.0))
                    self.reward_phase += float(info.get("reward_phase", 0.0))
                    self.reward_complete += float(info.get("reward_complete", 0.0))
                    self.reward_phase_reset += float(info.get("reward_phase_reset", 0.0))
                    self.reward_wrong_action += float(info.get("reward_wrong_action", 0.0))
                    self.reward_alternate += float(info.get("reward_alternate", 0.0))

                if is_combo_profile and index < len(dones) and dones[index]:
                    self.combo_episodes += 1.0
                    self.combo_successes += float(info.get("combo_success", 0.0))
                    self.combo_episode_max_total += float(info.get("episode_max_combo", 0.0))
                    scenario_metrics["episodes"] += 1.0
                    scenario_metrics["successes"] += float(info.get("combo_success", 0.0))
                    scenario_metrics["alternate_successes"] += float(info.get("combo_alternate_success", 0.0))
                    scenario_metrics["episode_max_total"] += float(info.get("episode_max_combo", 0.0))
                    self.combo_alternate_successes += float(info.get("combo_alternate_success", 0.0))

            return True

        def _on_rollout_end(self) -> None:
            count = max(1, self.step_count)
            combo_count = max(1, self.combo_step_count)
            self.logger.record("kof/profile_combo_step_rate", self.combo_step_count / count)
            self.logger.record("kof/profile_fight_step_rate", self.fight_step_count / count)
            self.logger.record(
                "kof_runtime/dropped_event_total",
                self.runtime_step_event_dropped,
            )
            self.logger.record(
                "kof_runtime/batch_epoch_mismatch_total",
                self.runtime_batch_epoch_mismatch,
            )
            self.logger.record(
                "kof_runtime/chip_hit_block_conflict_total",
                self.runtime_chip_hit_block_conflict,
            )
            self.logger.record(
                "kof_runtime/nonfinite_observation_total",
                self.runtime_nonfinite_observation,
            )
            self.logger.record(
                "kof_runtime/event_vector_enabled_step_rate",
                self.runtime_event_feature_enabled_steps / count,
            )
            self.logger.record(
                "kof_runtime/event_vector_nonzero_step_rate",
                self.runtime_event_feature_nonzero_steps / count,
            )
            self.logger.record(
                "kof_runtime/event_vector_nonzero_mean",
                self.runtime_event_feature_nonzero_total / count,
            )
            fight_event_steps = max(1.0, self.fight_event_metrics["steps"])
            fight_event_frames = max(1.0, self.fight_event_metrics["frames"])
            for metric_name in (
                "p1_reaction_active",
                "p1_reaction_remaining_valid",
                "p2_reaction_active",
                "p2_reaction_remaining_valid",
                "defense_non_neutral",
                "confirm_non_neutral",
            ):
                self.logger.record(
                    f"kof_event_source/{metric_name}_rate",
                    self.fight_event_metrics[metric_name] / fight_event_steps,
                )
            for metric_name in (
                "block_contact",
                "clean_hit",
                "blockstun_end",
                "starter_hit",
                "starter_blocked",
            ):
                self.logger.record(
                    f"kof_event_source/{metric_name}_per_1k_frames",
                    self.fight_event_metrics[metric_name]
                    * 1000.0
                    / fight_event_frames,
                )
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
            self.logger.record("kof/oniyaki_anti_air_hit_rate", self.oniyaki_anti_air_hit / max(1.0, self.action_16))
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
            self.logger.record("kof/action_27_rate", self.action_27 / count)
            self.logger.record("kof/action_27_hit_total", self.action_27_hit)
            self.logger.record("kof/action_27_hit_rate", self.action_27_hit / max(1.0, self.action_27))
            self.logger.record("kof/action_28_rate", self.action_28 / count)
            self.logger.record("kof/action_28_hit_total", self.action_28_hit)
            self.logger.record("kof/action_28_hit_rate", self.action_28_hit / max(1.0, self.action_28))
            self.logger.record("kof/mean_distance_x_abs", self.distance_x_abs / count)
            self.logger.record("kof/reward_hp_total", self.reward_hp)
            self.logger.record("kof/reward_hitbox_total", self.reward_hitbox)
            self.logger.record("kof/reward_distance_total", self.reward_distance)
            self.logger.record("kof/reward_defense_total", self.reward_defense)
            self.logger.record("kof/reward_combo_total", self.reward_combo)
            self.logger.record(
                "kof_fight/reward_4plus_milestone_total",
                self.reward_fight_combo_4plus_milestone,
            )
            self.logger.record("kof/reward_super_total", self.reward_super)
            self.logger.record("kof/reward_anti_air_total", self.reward_anti_air)
            self.logger.record("kof/reward_safety_total", self.reward_safety)
            self.logger.record("kof/reward_cancel_total", self.reward_cancel)
            self.logger.record("kof/reward_fast_win_total", self.reward_fast_win)
            self.logger.record("kof/reward_outcome_total", self.reward_outcome)
            self.logger.record("kof/reward_time_total", self.reward_time)
            self.logger.record(
                "kof/reward_curriculum_total",
                self.reward_curriculum,
            )
            self.logger.record("kof/combo_episodes_total", self.combo_episodes)
            self.logger.record("kof/combo_success_rate", self.combo_successes / max(1.0, self.combo_episodes))
            self.logger.record("kof/combo_episode_max_mean", self.combo_episode_max_total / max(1.0, self.combo_episodes))
            for scenario_name, metrics in self.combo_scenario_metrics.items():
                episode_count = max(1.0, metrics["episodes"])
                self.logger.record(
                    f"kof_combo/{scenario_name}/success_rate",
                    metrics["successes"] / episode_count,
                    exclude="stdout",
                )
                self.logger.record(
                    f"kof_combo/{scenario_name}/alternate_rate",
                    metrics.get("alternate_successes", 0.0) / episode_count,
                    exclude="stdout",
                )
                self.logger.record(
                    f"kof_combo/{scenario_name}/episode_max_mean",
                    metrics["episode_max_total"] / episode_count,
                    exclude="stdout",
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
            self.logger.record("kof/reward_wrong_action_total", self.reward_wrong_action)
            self.logger.record("kof/reward_alternate_total", self.reward_alternate)
            self.logger.record("kof/combo_alternate_success_total", self.combo_alternate_successes)
            self.logger.record("kof/emulated_frames", self.frames_total)
            self.logger.record("kof/emulated_frames_cumulative", self.cumulative_frames)

            fight_episode_count = max(1.0, self.fight_episodes)
            fight_wins = (
                self.fight_outcome_counts["win_ko"]
                + self.fight_outcome_counts["win_timeout"]
            )
            followup_queued_total = float(self.fight_followup_queued.sum())
            followup_started_total = float(self.fight_followup_started.sum())
            followup_hit_total = float(self.fight_followup_hit.sum())
            self.logger.record("kof_fight/episodes_total", self.fight_episodes)
            self.logger.record("kof_fight/win_rate", fight_wins / fight_episode_count)
            for outcome_name, outcome_count in self.fight_outcome_counts.items():
                self.logger.record(f"kof_fight/{outcome_name}_total", outcome_count)
            self.logger.record(
                "kof_fight/episode_max_combo_mean",
                self.fight_episode_max_combo_total / fight_episode_count,
            )
            self.logger.record(
                "kof_fight/combo_2plus_episode_rate",
                self.fight_combo_2plus_episodes / fight_episode_count,
            )
            self.logger.record(
                "kof_fight/combo_4plus_episode_rate",
                self.fight_combo_4plus_episodes / fight_episode_count,
            )
            self.logger.record(
                "kof_fight/damage_dealt_mean",
                self.fight_episode_damage_dealt_total / fight_episode_count,
            )
            self.logger.record(
                "kof_fight/damage_taken_mean",
                self.fight_episode_damage_taken_total / fight_episode_count,
            )
            self.logger.record("kof_fight/free_decision_steps", self.fight_free_decision_steps)
            self.logger.record("kof_fight/queue_decision_steps", self.fight_queue_decision_steps)
            self.logger.record("kof_fight/forced_idle_steps", self.fight_forced_idle_steps)
            self.logger.record("kof_fight/queued_followup_total", followup_queued_total)
            self.logger.record("kof_fight/started_followup_total", followup_started_total)
            self.logger.record("kof_fight/followup_hit_total", followup_hit_total)
            self.logger.record(
                "kof_fight/followup_hit_rate",
                followup_hit_total / max(1.0, followup_started_total),
            )
            self.logger.record("kof_fight/emulated_frames", self.fight_frames_total)
            self.logger.record("kof_fight/emulated_frames_cumulative", self.cumulative_fight_frames)
            for fight_mode, metrics in self.fight_mode_metrics.items():
                mode_episode_count = max(1.0, metrics["episodes"])
                prefix = f"kof_fight_{fight_mode}"
                mode_event_metrics = self.fight_mode_event_metrics[fight_mode]
                mode_event_steps = max(1.0, mode_event_metrics["steps"])
                mode_event_frames = max(1.0, mode_event_metrics["frames"])
                self.logger.record(f"{prefix}/episodes_total", metrics["episodes"])
                self.logger.record(
                    f"{prefix}/win_rate",
                    metrics["wins"] / mode_episode_count,
                )
                self.logger.record(
                    f"{prefix}/combo_4plus_episode_rate",
                    metrics["combo_4plus"] / mode_episode_count,
                )
                self.logger.record(
                    f"{prefix}/episode_max_combo_mean",
                    metrics["episode_max_total"] / mode_episode_count,
                )
                self.logger.record(
                    f"{prefix}/teacher_completions_total",
                    metrics["teacher_completions"],
                )
                for metric_name in (
                    "p1_reaction_active",
                    "p1_reaction_remaining_valid",
                    "p2_reaction_active",
                    "p2_reaction_remaining_valid",
                    "defense_non_neutral",
                    "confirm_non_neutral",
                ):
                    self.logger.record(
                        f"{prefix}/{metric_name}_rate",
                        mode_event_metrics[metric_name] / mode_event_steps,
                        exclude="stdout",
                    )
                for metric_name in (
                    "block_contact",
                    "clean_hit",
                    "blockstun_end",
                    "starter_hit",
                    "starter_blocked",
                ):
                    self.logger.record(
                        f"{prefix}/{metric_name}_per_1k_frames",
                        mode_event_metrics[metric_name]
                        * 1000.0
                        / mode_event_frames,
                        exclude="stdout",
                    )
                free_decision_count = max(
                    1.0,
                    self.fight_mode_free_decision_steps[fight_mode],
                )
                self.logger.record(
                    f"{prefix}/free_decision_steps",
                    self.fight_mode_free_decision_steps[fight_mode],
                    exclude="stdout",
                )
                for action_id in TACTICAL_ACTION_IDS:
                    available_steps = self.fight_mode_action_available[
                        fight_mode
                    ][action_id]
                    selected_total = self.fight_mode_action_selected[
                        fight_mode
                    ][action_id]
                    self.logger.record(
                        f"{prefix}/action_{action_id}_selection_rate_when_available",
                        selected_total / max(1.0, available_steps),
                        exclude="stdout",
                    )
                    self.logger.record(
                        f"{prefix}/action_{action_id}_free_decision_rate",
                        self.fight_mode_free_action_selected[
                            fight_mode
                        ][action_id] / free_decision_count,
                        exclude="stdout",
                    )
                self.logger.record(
                    f"{prefix}/crouch_b_close_c_free_decision_rate",
                    (
                        self.fight_mode_free_action_selected[fight_mode][11]
                        + self.fight_mode_free_action_selected[fight_mode][8]
                    ) / free_decision_count,
                    exclude="stdout",
                )
                for condition, tag_name in TACTICAL_CONDITION_TAGS.items():
                    tactical_metrics = self.cumulative_mode_tactical_metrics[
                        fight_mode
                    ][condition]
                    opportunity_count = max(
                        1.0,
                        tactical_metrics["opportunities"],
                    )
                    self.logger.record(
                        f"{prefix}/{tag_name}_opportunities_cumulative",
                        tactical_metrics["opportunities"],
                        exclude="stdout",
                    )
                    self.logger.record(
                        f"{prefix}/{tag_name}_rate",
                        tactical_metrics["successes"] / opportunity_count,
                        exclude="stdout",
                    )
                approach_opportunities = max(
                    1.0,
                    self.cumulative_mode_tactical_metrics[fight_mode]["approach"][
                        "opportunities"
                    ],
                )
                self.logger.record(
                    f"{prefix}/approach_p1_progress_mean",
                    self.cumulative_mode_approach_progress[fight_mode]
                    / approach_opportunities,
                    exclude="stdout",
                )
                self.logger.record(
                    f"{prefix}/approach_forward_frames_mean",
                    self.cumulative_mode_approach_forward_frames[fight_mode]
                    / approach_opportunities,
                    exclude="stdout",
                )
                for p2_style, style_metrics in self.fight_style_metrics[
                    fight_mode
                ].items():
                    style_episode_count = max(1.0, style_metrics["episodes"])
                    style_prefix = f"{prefix}/{p2_style}"
                    self.logger.record(
                        f"{style_prefix}/episodes_total",
                        style_metrics["episodes"],
                    )
                    self.logger.record(
                        f"{style_prefix}/win_rate",
                        style_metrics["wins"] / style_episode_count,
                    )
                    self.logger.record(
                        f"{style_prefix}/combo_4plus_episode_rate",
                        style_metrics["combo_4plus"] / style_episode_count,
                    )
                    self.logger.record(
                        f"{style_prefix}/episode_max_combo_mean",
                        style_metrics["episode_max_total"] / style_episode_count,
                    )
                    self.logger.record(
                        f"{style_prefix}/teacher_completions_total",
                        style_metrics["teacher_completions"],
                    )
            for curriculum_name, metrics in self.fight_curriculum_metrics.items():
                prefix = f"kof_fight_targeted/{curriculum_name}"
                episode_count = max(1.0, metrics["episodes"])
                opportunity_count = max(1.0, metrics["opportunities"])
                self.logger.record(
                    f"{prefix}/opportunities_total",
                    metrics["opportunities"],
                    exclude="stdout",
                )
                self.logger.record(
                    f"{prefix}/success_per_opportunity_step",
                    metrics["successes"] / opportunity_count,
                    exclude="stdout",
                )
                self.logger.record(
                    f"{prefix}/episode_success_rate",
                    metrics["reward_machine_successes"] / episode_count,
                    exclude="stdout",
                )
                self.logger.record(
                    f"{prefix}/episode_failure_rate",
                    metrics["reward_machine_failures"] / episode_count,
                    exclude="stdout",
                )
                self.logger.record(
                    f"{prefix}/episode_timeout_rate",
                    metrics["reward_machine_timeouts"] / episode_count,
                    exclude="stdout",
                )
                self.logger.record(
                    f"{prefix}/reward_total",
                    metrics["reward_total"],
                    exclude="stdout",
                )
                if curriculum_name == FightCurriculum.DEFENSE.value:
                    counter_opportunity_count = max(
                        1.0,
                        metrics["counter_opportunities"],
                    )
                    self.logger.record(
                        f"{prefix}/block_contact_success_per_opportunity_step",
                        metrics["guard_successes"] / opportunity_count,
                        exclude="stdout",
                    )
                    self.logger.record(
                        f"{prefix}/counter_hit_success_per_opportunity_step",
                        metrics["counter_successes"] / counter_opportunity_count,
                        exclude="stdout",
                    )
                self.logger.record(
                    f"{prefix}/win_rate",
                    metrics["wins"] / episode_count,
                    exclude="stdout",
                )
                self.logger.record(
                    f"{prefix}/combo_4plus_episode_rate",
                    metrics["combo_4plus"] / episode_count,
                    exclude="stdout",
                )
                self.logger.record(
                    f"{prefix}/episode_max_combo_mean",
                    metrics["episode_max_total"] / episode_count,
                    exclude="stdout",
                )
            for recipe_name, metrics in self.level_recipe_metrics.items():
                prefix = f"kof_level/{recipe_name}"
                episode_count = max(1.0, metrics["episodes"])
                self.logger.record(
                    f"{prefix}/episode_success_rate",
                    metrics["successes"] / episode_count,
                    exclude="stdout",
                )
                self.logger.record(
                    f"{prefix}/failure_rate",
                    metrics["failures"] / episode_count,
                    exclude="stdout",
                )
                self.logger.record(
                    f"{prefix}/timeout_rate",
                    metrics["timeouts"] / episode_count,
                    exclude="stdout",
                )
                self.logger.record(
                    f"{prefix}/episode_frames_mean",
                    metrics["frames_total"] / episode_count,
                    exclude="stdout",
                )
                self.logger.record(
                    f"{prefix}/reward_total",
                    metrics["reward_total"],
                    exclude="stdout",
                )
            for action_id in range(ACTION_COUNT):
                available_steps = float(self.fight_action_available[action_id])
                selected_total = float(self.fight_action_selected[action_id])
                self.logger.record(
                    f"kof_fight/action_{action_id}_available_steps",
                    available_steps,
                    exclude="stdout",
                )
                self.logger.record(
                    f"kof_fight/action_{action_id}_selected_total",
                    selected_total,
                    exclude="stdout",
                )
                self.logger.record(
                    f"kof_fight/action_{action_id}_selection_rate_when_available",
                    selected_total / max(1.0, available_steps),
                    exclude="stdout",
                )
                self.logger.record(
                    f"kof_fight/followup_{action_id}_queued_total",
                    float(self.fight_followup_queued[action_id]),
                    exclude="stdout",
                )
                self.logger.record(
                    f"kof_fight/followup_{action_id}_started_total",
                    float(self.fight_followup_started[action_id]),
                    exclude="stdout",
                )
                self.logger.record(
                    f"kof_fight/followup_{action_id}_hit_total",
                    float(self.fight_followup_hit[action_id]),
                    exclude="stdout",
                )
            self.reset_rollout_metrics()

    root = args.root.resolve()
    preset = TRAINING_PRESETS[args.preset]
    effective_action_repeat = (
        args.action_repeat if args.action_repeat is not None else preset["action_repeat"]
    )
    effective_gamma = args.gamma if args.gamma is not None else preset["gamma"]
    effective_gae_lambda = (
        args.gae_lambda if args.gae_lambda is not None else preset["gae_lambda"]
    )
    effective_n_steps = args.n_steps if args.n_steps is not None else preset["n_steps"]
    print(
        f"Preset {args.preset}: action_repeat={effective_action_repeat}, "
        f"gamma={effective_gamma}, gae_lambda={effective_gae_lambda}, "
        f"n_steps={effective_n_steps}"
    )

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
    combo_scenario_arguments = args.combo_scenario
    if combo_scenario_arguments is None and args.combo_suite == "kyo29":
        combo_scenario_arguments = [
            f"{scenario_name}=saves\\states\\kof98.slot{slot}.state"
            for scenario_name, slot in KYO29_COMBO_SUITE
        ]
    try:
        combo_scenario_specs = resolve_combo_scenario_specs(
            root,
            combo_scenario_arguments,
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

    training_dll_path = root / "build-vs2026-x64" / "Release" / "fbneo_training.dll"
    validate_file(training_dll_path, "fbneo_training.dll")
    validate_file(root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll", "fbneo_libretro.dll")
    validate_file(root / "roms" / "fbneo" / "kof98.zip", "kof98.zip")
    try:
        contract_client = KofEnvClient(training_dll_path)
        runtime_contract = contract_client.contract_metadata()
        contract_client.close()
    except (OSError, RuntimeError) as error:
        print(f"fbneo_training contract check failed: {error}", file=sys.stderr)
        return 2
    combo_env_count = training_profiles.count(TrainingProfile.COMBO)
    fight_env_count = training_profiles.count(TrainingProfile.FIGHT)
    if (
        fight_env_count
        and effective_action_repeat != runtime_contract["p1_hold_chunk_frames"]
    ):
        print(
            "Fight training with action-set version "
            f"{runtime_contract['action_set_version']} requires action_repeat="
            f"{runtime_contract['p1_hold_chunk_frames']} so P1 hold chunks end "
            "on a decision boundary. Use --preset repeat4.",
            file=sys.stderr,
        )
        return 2
    if args.guided_fight_envs < 0 or args.guided_fight_envs > fight_env_count:
        print(
            "--guided-fight-envs must be between 0 and the number of Fight environments "
            f"({fight_env_count}).",
            file=sys.stderr,
        )
        return 2
    level_recipe_bank_path: Optional[Path] = None
    level_recipes: list[LevelRecipe] = []
    if args.level_recipe_bank is not None:
        if args.targeted_fight_curriculum or args.targeted_fight_envs is not None:
            print(
                "--level-recipe-bank replaces the legacy Targeted Fight "
                "curriculum flags; do not combine them.",
                file=sys.stderr,
            )
            return 2
        level_recipe_bank_path = (
            args.level_recipe_bank
            if args.level_recipe_bank.is_absolute()
            else root / args.level_recipe_bank
        ).resolve()
        validate_file(level_recipe_bank_path, "Level Recipe bank")
        try:
            level_recipes = load_level_recipes(level_recipe_bank_path)
        except (KeyError, TypeError, ValueError, OSError) as error:
            print(f"Invalid Level Recipe bank: {error}", file=sys.stderr)
            return 2
        if not level_recipes:
            print("Level Recipe bank contains no recipes.", file=sys.stderr)
            return 2
        if observation_version is ObservationVersion.V1:
            print(
                "--level-recipe-bank requires --observation-version v2 or v3 so the "
                "policy can observe measured strategy state.",
                file=sys.stderr,
            )
            return 2
        if args.fight_reward_version == "auto":
            fight_reward_version = FightRewardVersion.SYMMETRIC_TACTICAL_V3
            effective_fight_reward_version = fight_reward_version
        elif (
            fight_reward_version
            is not FightRewardVersion.SYMMETRIC_TACTICAL_V3
        ):
            print(
                "--level-recipe-bank requires --fight-reward-version "
                f"{FightRewardVersion.SYMMETRIC_TACTICAL_V3.value}.",
                file=sys.stderr,
            )
            return 2
        for recipe in level_recipes:
            validate_file(recipe.base_state, f"Level Recipe state for {recipe.name}")
        targeted_fight_curricula = tuple(
            dict.fromkeys(FightCurriculum(recipe.task.value) for recipe in level_recipes)
        )
        available_recipe_workers = max(0, fight_env_count - args.guided_fight_envs)
        targeted_fight_env_count = (
            min(len(level_recipes), available_recipe_workers)
            if args.level_recipe_envs is None
            else args.level_recipe_envs
        )
        fight_rotation_enabled = True
    else:
        if args.level_recipe_envs is not None:
            print("--level-recipe-envs requires --level-recipe-bank.", file=sys.stderr)
            return 2
        targeted_fight_curricula = tuple(
            FightCurriculum(value)
            for value in (args.targeted_fight_curriculum or [])
        )
        targeted_fight_env_count = (
            len(targeted_fight_curricula)
            if args.targeted_fight_envs is None
            else args.targeted_fight_envs
        )
        fight_rotation_enabled = args.targeted_fight_envs is not None
    if targeted_fight_env_count < 0:
        print("--targeted-fight-envs must be 0 or greater.", file=sys.stderr)
        return 2
    if level_recipes and targeted_fight_env_count == 0:
        print(
            "--level-recipe-bank requires at least one available Fight worker.",
            file=sys.stderr,
        )
        return 2
    if args.oracle_teacher_weight < 0.0:
        print("--oracle-teacher-weight must be non-negative.", file=sys.stderr)
        return 2
    if args.oracle_teacher_weight > 0.0 and not level_recipes:
        print(
            "--oracle-teacher-weight requires --level-recipe-bank.",
            file=sys.stderr,
        )
        return 2
    if targeted_fight_env_count and not targeted_fight_curricula:
        print(
            "Targeted Fight workers require either a Level Recipe bank or at "
            "least one --targeted-fight-curriculum.",
            file=sys.stderr,
        )
        return 2
    if args.guided_fight_envs + targeted_fight_env_count > fight_env_count:
        print(
            "Guided and targeted Fight environments exceed the number of Fight "
            f"environments ({fight_env_count}).",
            file=sys.stderr,
        )
        return 2
    guided_fight_scenarios = tuple(
        args.guided_fight_scenario or DEFAULT_GUIDED_FIGHT_SCENARIOS
    )
    p2_styles = tuple(
        P2Style(style_name)
        for style_name in (
            args.p2_style
            or [style.value for style in P2Style]
        )
    )
    targeted_style = {
        FightCurriculum.DEFENSE: P2Style.POKE,
        FightCurriculum.ANTI_AIR: P2Style.JUMP_IN,
        FightCurriculum.APPROACH: P2Style.GUARD,
        FightCurriculum.HIT_CONFIRM: P2Style.POKE,
    }

    def targeted_worker_index(fight_index: int) -> Optional[int]:
        targeted_index = fight_index - args.guided_fight_envs
        if 0 <= targeted_index < targeted_fight_env_count:
            return targeted_index
        return None

    def physical_worker_index(fight_index: int) -> Optional[int]:
        physical_index = (
            fight_index
            - args.guided_fight_envs
            - targeted_fight_env_count
        )
        return physical_index if physical_index >= 0 else None

    def fight_curriculum_for_index(fight_index: int) -> FightCurriculum:
        if fight_index < args.guided_fight_envs:
            return FightCurriculum.COMBO_ROUTE
        targeted_index = targeted_worker_index(fight_index)
        if targeted_index is not None:
            if level_recipes:
                return FightCurriculum(
                    level_recipes[targeted_index % len(level_recipes)].task.value
                )
            return targeted_fight_curricula[
                targeted_index % len(targeted_fight_curricula)
            ]
        return FightCurriculum.NONE

    def p2_style_for_index(fight_index: int) -> P2Style:
        curriculum = fight_curriculum_for_index(fight_index)
        physical_index = physical_worker_index(fight_index)
        style_index = (
            physical_index
            if fight_rotation_enabled and physical_index is not None
            else fight_index
        )
        return targeted_style.get(
            curriculum,
            p2_styles[style_index % len(p2_styles)],
        )

    targeted_fight_rotation = (
        []
        if level_recipes
        else [
            (curriculum, targeted_style[curriculum])
            for curriculum in targeted_fight_curricula
        ]
    )
    physical_fight_rotation = [
        (FightCurriculum.NONE, style)
        for style in p2_styles
    ]

    def fight_rotation_for_index(
        fight_index: int,
    ) -> Optional[list[tuple[FightCurriculum, P2Style]]]:
        if not fight_rotation_enabled:
            return None
        if targeted_worker_index(fight_index) is not None:
            if level_recipes:
                return None
            return targeted_fight_rotation
        if physical_worker_index(fight_index) is not None:
            return physical_fight_rotation
        return None

    def fight_rotation_offset_for_index(fight_index: int) -> int:
        targeted_index = targeted_worker_index(fight_index)
        if targeted_index is not None:
            return targeted_index
        physical_index = physical_worker_index(fight_index)
        return physical_index if physical_index is not None else 0

    def level_recipe_rotation_for_index(
        fight_index: int,
    ) -> Optional[list[LevelRecipe]]:
        if level_recipes and targeted_worker_index(fight_index) is not None:
            return level_recipes
        return None

    def level_recipe_rotation_offset_for_index(fight_index: int) -> int:
        targeted_index = targeted_worker_index(fight_index)
        return targeted_index if targeted_index is not None else 0

    p2_style_assignments = []
    for fight_index in range(fight_env_count):
        fight_rotation = fight_rotation_for_index(fight_index)
        p2_style_assignments.append({
            "fight_index": fight_index,
            "mode": (
                "guided"
                if fight_index < args.guided_fight_envs
                else (
                    "targeted"
                    if fight_curriculum_for_index(fight_index)
                    is not FightCurriculum.NONE
                    else "physical"
                )
            ),
            "curriculum": fight_curriculum_for_index(fight_index).value,
            "style": p2_style_for_index(fight_index).value,
            "guided_scenario": (
                guided_fight_scenarios[
                    fight_index % len(guided_fight_scenarios)
                ]
                if fight_curriculum_for_index(fight_index) in (
                    FightCurriculum.COMBO_ROUTE,
                    FightCurriculum.HIT_CONFIRM,
                )
                else None
            ),
            "rotation": [
                {
                    "curriculum": curriculum.value,
                    "style": style.value,
                }
                for curriculum, style in (fight_rotation or [])
            ],
            "level_recipes": [
                recipe.name
                for recipe in (
                    level_recipe_rotation_for_index(fight_index) or []
                )
            ],
        })

    metrics_fight_assignments = list(p2_style_assignments)
    if level_recipes and targeted_fight_env_count:
        metrics_fight_assignments.extend(
            {
                "fight_index": -1,
                "mode": "targeted",
                "curriculum": curriculum.value,
                "style": targeted_style[curriculum].value,
                "guided_scenario": None,
            }
            for curriculum in targeted_fight_curricula
        )
    elif fight_rotation_enabled and targeted_fight_env_count:
        metrics_fight_assignments.extend(
            {
                "fight_index": -1,
                "mode": "targeted",
                "curriculum": curriculum.value,
                "style": targeted_style[curriculum].value,
                "guided_scenario": None,
            }
            for curriculum in targeted_fight_curricula
        )
    physical_fight_env_count = (
        fight_env_count
        - args.guided_fight_envs
        - targeted_fight_env_count
    )
    if fight_rotation_enabled and physical_fight_env_count:
        metrics_fight_assignments.extend(
            {
                "fight_index": -1,
                "mode": "physical",
                "curriculum": FightCurriculum.NONE.value,
                "style": style.value,
                "guided_scenario": None,
            }
            for style in p2_styles
        )
    if combo_env_count:
        for scenario_name, scenario_state_path in combo_scenario_specs:
            validate_file(scenario_state_path, f"Combo state for {scenario_name}")
            print(f"Combo scenario: {scenario_name} => {scenario_state_path}")
    if fight_env_count:
        validate_file(fight_state_path, "Fight save state")
        print(f"Fight profile state: {fight_state_path}")
    print(
        f"Training environments: Combo={combo_env_count}, "
        f"Guided Fight={args.guided_fight_envs}, "
        f"Targeted Fight={targeted_fight_env_count}, "
        f"Physical Fight={physical_fight_env_count}"
    )
    if fight_env_count:
        print(
            "P2 styles: "
            + ", ".join(style.value for style in p2_styles)
        )
    if fight_rotation_enabled and targeted_fight_env_count:
        if level_recipes:
            print(
                "Level Recipe rotation: "
                + ", ".join(recipe.name for recipe in level_recipes)
            )
        else:
            print(
                "Targeted curriculum rotation: "
                + ", ".join(
                    curriculum.value for curriculum in targeted_fight_curricula
                )
            )
    if fight_rotation_enabled and physical_fight_env_count:
        print(
            "Physical P2 style rotation: "
            + ", ".join(style.value for style in p2_styles)
        )

    if args.save_name is None:
        args.save_name = f"kof98_{args.profile}_ppo"
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_run_name = (
        args.tensorboard_run_name
        or f"{args.save_name}_{run_timestamp}"
    )

    log_dir = (args.log_dir or root / "ai_logs").resolve()
    save_dir = (args.save_dir or root / "trained_models").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    resume_path: Optional[Path] = None
    if args.resume:
        resume_path = args.resume if args.resume.is_absolute() else root / args.resume
        validate_file(resume_path, "Resume model")
    migrate_path: Optional[Path] = None
    if args.migrate_from:
        migrate_path = (
            args.migrate_from
            if args.migrate_from.is_absolute()
            else root / args.migrate_from
        )
        validate_file(migrate_path, "V1 migration source model")
    teacher_path: Optional[Path] = None
    if args.teacher_model:
        teacher_path = (
            args.teacher_model
            if args.teacher_model.is_absolute()
            else root / args.teacher_model
        )
        validate_file(teacher_path, "Teacher model")

    manifest_files: dict[str, Path] = {
        "fbneo_training_dll": root / "build-vs2026-x64" / "Release" / "fbneo_training.dll",
        "fbneo_libretro_core": root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll",
        "rom": root / "roms" / "fbneo" / "kof98.zip",
    }
    if combo_env_count:
        for scenario_name, scenario_state_path in combo_scenario_specs:
            manifest_files[f"combo_state_{scenario_name}"] = scenario_state_path
    if fight_env_count:
        manifest_files["fight_state"] = fight_state_path
    if level_recipe_bank_path is not None:
        manifest_files["level_recipe_bank"] = level_recipe_bank_path
        for recipe in level_recipes:
            manifest_files[
                f"level_state_{recipe.name}"
            ] = recipe.base_state
    if resume_path is not None:
        manifest_files["resume_model"] = resume_path
    if migrate_path is not None:
        manifest_files["migration_source_model"] = migrate_path
    if teacher_path is not None:
        manifest_files["teacher_model"] = teacher_path
    manifest_path = log_dir / (
        f"run_manifest_{args.save_name}_"
        f"{run_timestamp}.json"
    )
    write_run_manifest(
        manifest_path,
        args,
        root,
        effective={
            "preset": args.preset,
            "action_repeat": effective_action_repeat,
            "gamma": effective_gamma,
            "gae_lambda": effective_gae_lambda,
            "n_steps": effective_n_steps,
            "timesteps": args.timesteps,
            "checkpoint_every": args.checkpoint_every,
            "relative_checkpoints": args.relative_checkpoints,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "profile": args.profile,
            "combo_ratio": args.combo_ratio,
            "guided_fight_envs": args.guided_fight_envs,
            "guided_fight_scenarios": guided_fight_scenarios,
            "targeted_fight_envs": targeted_fight_env_count,
            "level_recipe_bank": (
                str(level_recipe_bank_path)
                if level_recipe_bank_path is not None
                else None
            ),
            "level_recipe_envs": (
                targeted_fight_env_count if level_recipes else 0
            ),
            "level_recipes": [
                recipe.to_dict() for recipe in level_recipes
            ],
            "fight_rotation_enabled": fight_rotation_enabled,
            "targeted_fight_curricula": [
                curriculum.value for curriculum in targeted_fight_curricula
            ],
            "targeted_fight_rotation": [
                {
                    "curriculum": curriculum.value,
                    "style": style.value,
                }
                for curriculum, style in (
                    targeted_fight_rotation if fight_rotation_enabled else []
                )
            ],
            "physical_p2_style_rotation": [
                style.value
                for _, style in (
                    physical_fight_rotation if fight_rotation_enabled else []
                )
            ],
            "p2_styles": tuple(style.value for style in p2_styles),
            "p2_style_assignments": p2_style_assignments,
            "mask_level": args.mask_level,
            "p2_training_ai": args.p2_training_ai,
            "fight_reward_version": effective_fight_reward_version.value,
            "observation_version": observation_version.value,
            "observation_schema_id": observation_schema_id(observation_version),
            "observation_size": observation_size(observation_version),
            "observation_event_features": (
                not args.disable_observation_event_features
            ),
            "runtime_contract": runtime_contract,
            "allow_action_set_migration": args.allow_action_set_migration,
            "migration_source_model": str(migrate_path) if migrate_path else None,
            "teacher_model": str(teacher_path) if teacher_path else None,
            "teacher_weight": args.teacher_weight,
            "teacher_decay_steps": args.teacher_decay_steps,
            "teacher_batch_size": args.teacher_batch_size,
            "oracle_teacher_weight": args.oracle_teacher_weight,
            "oracle_teacher_decay_steps": args.oracle_teacher_decay_steps,
            "oracle_teacher_batch_size": args.oracle_teacher_batch_size,
            "oracle_teacher_updates": args.oracle_teacher_updates,
            "tensorboard_run_name": tensorboard_run_name,
        },
        manifest_files=manifest_files,
    )
    print(f"Run manifest: {manifest_path}")
    print(f"TensorBoard run: {tensorboard_run_name}")

    if args.tensorboard:
        start_tensorboard(log_dir, args.tensorboard_port)

    # 環境編組:Combo env 在每次 reset 輪替全部 scenario;Targeted Fight
    # 輪替完整 curriculum pool;Physical Fight 輪替完整 P2 style pool。
    environment_specs: list[
        tuple[
            TrainingProfile,
            str,
            Path,
            FightCurriculum,
            P2Style,
            int,
            Optional[list[tuple[FightCurriculum, P2Style]]],
            int,
            int,
            Optional[list[LevelRecipe]],
            int,
        ]
    ] = []
    combo_spec_index = 0
    fight_spec_index = 0
    for training_profile in training_profiles:
        if training_profile is TrainingProfile.COMBO:
            scenario_name, scenario_state_path = combo_scenario_specs[
                combo_spec_index % len(combo_scenario_specs)
            ]
            combo_rotation_offset = combo_spec_index
            combo_spec_index += 1
            fight_curriculum = FightCurriculum.NONE
            p2_style = P2Style.ONIYAKI
            fight_rotation = None
            fight_rotation_offset = 0
            fight_rotation_stride = 1
            level_recipe_rotation = None
            level_recipe_rotation_offset = 0
        else:
            combo_rotation_offset = 0
            fight_curriculum = fight_curriculum_for_index(fight_spec_index)
            p2_style = p2_style_for_index(fight_spec_index)
            fight_rotation = fight_rotation_for_index(fight_spec_index)
            fight_rotation_offset = fight_rotation_offset_for_index(
                fight_spec_index
            )
            fight_rotation_stride = 1
            level_recipe_rotation = level_recipe_rotation_for_index(
                fight_spec_index
            )
            level_recipe_rotation_offset = (
                level_recipe_rotation_offset_for_index(fight_spec_index)
            )
            needs_guided_scenario = fight_curriculum in (
                FightCurriculum.COMBO_ROUTE,
                FightCurriculum.HIT_CONFIRM,
            ) or any(
                curriculum in (
                    FightCurriculum.COMBO_ROUTE,
                    FightCurriculum.HIT_CONFIRM,
                )
                for curriculum, _ in (fight_rotation or [])
            )
            scenario_name = (
                guided_fight_scenarios[
                    fight_spec_index % len(guided_fight_scenarios)
                ]
                if needs_guided_scenario
                else combo_scenario_specs[0][0]
            )
            scenario_state_path = combo_scenario_specs[0][1]
            fight_spec_index += 1
        environment_specs.append(
            (
                training_profile,
                scenario_name,
                scenario_state_path,
                fight_curriculum,
                p2_style,
                combo_rotation_offset,
                fight_rotation,
                fight_rotation_offset,
                fight_rotation_stride,
                level_recipe_rotation,
                level_recipe_rotation_offset,
            )
        )

    def monitored_env(
        seed: int,
        training_profile: TrainingProfile,
        combo_scenario: str,
        scenario_state_path: Path,
        fight_curriculum: FightCurriculum,
        p2_style: P2Style,
        combo_rotation_offset: int,
        fight_rotation: Optional[list[tuple[FightCurriculum, P2Style]]],
        fight_rotation_offset: int,
        fight_rotation_stride: int,
        level_recipe_rotation: Optional[list[LevelRecipe]],
        level_recipe_rotation_offset: int,
    ) -> Callable:
        def _init():
            return MaskableMonitor(make_env(
                root=root,
                combo_state_path=scenario_state_path,
                fight_state_path=fight_state_path,
                training_profile=training_profile,
                combo_scenario=combo_scenario,
                combo_scenario_rotation=(
                    combo_scenario_specs
                    if training_profile is TrainingProfile.COMBO
                    else None
                ),
                combo_rotation_offset=combo_rotation_offset,
                combo_rotation_stride=max(1, combo_env_count),
                action_mask_level=ActionMaskLevel(args.mask_level),
                action_repeat=effective_action_repeat,
                seed=seed,
                p2_training_ai=args.p2_training_ai,
                p2_style=p2_style,
                fight_guided=fight_curriculum is FightCurriculum.COMBO_ROUTE,
                fight_curriculum=fight_curriculum,
                fight_rotation=fight_rotation,
                fight_rotation_offset=fight_rotation_offset,
                fight_rotation_stride=fight_rotation_stride,
                level_recipe_rotation=level_recipe_rotation,
                level_recipe_rotation_offset=level_recipe_rotation_offset,
                level_recipe_rotation_stride=1,
                hitbox_reward=not args.no_hitbox_reward,
                viewer=args.viewer,
                viewer_scale=args.viewer_scale,
                viewer_fps=args.viewer_fps,
                viewer_speed=args.viewer_speed,
                viewer_hitboxes=args.viewer_hitboxes,
                viewer_terminal_tail_frames=args.viewer_terminal_tail_frames,
                observation_version=observation_version,
                observation_event_features=(
                    not args.disable_observation_event_features
                ),
                fight_reward_version=fight_reward_version,
            )())

        return _init

    env_fns = [
        monitored_env(
            seed,
            training_profile,
            combo_scenario,
            scenario_state_path,
            fight_curriculum,
            p2_style,
            combo_rotation_offset,
            fight_rotation,
            fight_rotation_offset,
            fight_rotation_stride,
            level_recipe_rotation,
            level_recipe_rotation_offset,
        )
        for seed, (
            training_profile,
            combo_scenario,
            scenario_state_path,
            fight_curriculum,
            p2_style,
            combo_rotation_offset,
            fight_rotation,
            fight_rotation_offset,
            fight_rotation_stride,
            level_recipe_rotation,
            level_recipe_rotation_offset,
        )
        in enumerate(environment_specs)
    ]
    if len(env_fns) == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = MaskableSubprocVecEnv(env_fns, start_method="spawn")

    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be greater than zero")
    if args.relative_checkpoints:
        checkpoint_callback = RelativeCheckpointCallback(
            interval_timesteps=args.checkpoint_every,
            total_timesteps=args.timesteps,
            save_path=save_dir,
            name_prefix=args.save_name,
        )
    else:
        checkpoint_callback = CheckpointCallback(
            save_freq=max(
                1,
                args.checkpoint_every // max(1, len(env_fns)),
            ),
            save_path=str(save_dir),
            name_prefix=args.save_name,
        )
    callbacks = [
        checkpoint_callback,
        TrainingMetricsCallback(metrics_fight_assignments),
    ]
    if teacher_path is not None:
        callbacks.append(
            KickstartTeacherCallback(
                teacher_path,
                legacy_observation_size=OBSERVATION_V1_SIZE,
                initial_weight=args.teacher_weight,
                decay_timesteps=args.teacher_decay_steps,
                batch_size=args.teacher_batch_size,
            )
        )
    if level_recipes and args.oracle_teacher_weight > 0.0:
        callbacks.append(
            OracleCurriculumCallback(
                initial_weight=args.oracle_teacher_weight,
                decay_timesteps=args.oracle_teacher_decay_steps,
                batch_size=args.oracle_teacher_batch_size,
                updates_per_rollout=args.oracle_teacher_updates,
            )
        )
    callback = CallbackList(callbacks)

    def create_new_model():
        return MaskablePPO(
            "MlpPolicy",
            env,
            device=args.device,
            verbose=1,
            n_steps=effective_n_steps,
            batch_size=128,
            n_epochs=6,
            gamma=effective_gamma,
            gae_lambda=effective_gae_lambda,
            learning_rate=2.5e-4,
            ent_coef=0.01,
            clip_range=0.2,
            target_kl=0.03,
            tensorboard_log=str(log_dir),
            seed=args.seed,
        )

    expected_model_contract = {
        "kof_action_set_version": runtime_contract["action_set_version"],
        "kof_action_count": runtime_contract["action_count"],
        "kof_observation_schema_id": observation_schema_id(observation_version),
    }

    def stamp_model_contract(target_model) -> None:
        for name, value in expected_model_contract.items():
            setattr(target_model, name, value)

    def validate_resume_contract(target_model) -> None:
        mismatches = []
        for name, expected_value in expected_model_contract.items():
            actual_value = getattr(target_model, name, None)
            if actual_value != expected_value:
                mismatches.append(
                    f"{name}: checkpoint={actual_value!r}, runtime={expected_value!r}"
                )
        if not mismatches:
            return
        details = "; ".join(mismatches)
        if not args.allow_action_set_migration:
            raise ValueError(
                "Checkpoint contract does not match this training runtime: "
                f"{details}. Add --allow-action-set-migration only when this "
                "semantic change is intentional."
            )
        print(f"Explicit action-set/schema migration accepted: {details}")

    if resume_path is not None:
        model = MaskablePPO.load(
            str(resume_path),
            env=env,
            device=args.device,
            tensorboard_log=str(log_dir),
            gamma=effective_gamma,
            gae_lambda=effective_gae_lambda,
            n_steps=effective_n_steps,
        )
        validate_resume_contract(model)
        stamp_model_contract(model)
        print(
            f"Resumed with gamma={effective_gamma}, "
            f"gae_lambda={effective_gae_lambda}, n_steps={effective_n_steps}"
        )
    elif migrate_path is not None:
        source_model = MaskablePPO.load(str(migrate_path), device="cpu")
        source_shape = tuple(source_model.observation_space.shape)
        if int(source_model.action_space.n) != ACTION_COUNT:
            raise ValueError(
                f"Migration source has {source_model.action_space.n} actions; "
                f"expected {ACTION_COUNT}"
            )
        model = create_new_model()
        stamp_model_contract(model)
        if observation_version is ObservationVersion.V2:
            source_schema = getattr(
                source_model,
                "kof_observation_schema_id",
                None,
            )
            if (
                source_shape == (OBSERVATION_V2_SIZE,)
                and source_schema == OBSERVATION_SCHEMA_V2_ID
            ):
                model.policy.load_state_dict(
                    source_model.policy.state_dict(),
                    strict=True,
                )
                migration_message = (
                    "Cloned V2 policy exactly with a fresh optimizer for "
                    "controlled A/B comparison."
                )
            elif source_shape != (OBSERVATION_V1_SIZE,):
                raise ValueError(
                    "V2 migration source must be either a stamped V2 "
                    "140-value checkpoint or a V1 26-value checkpoint; "
                    f"got shape={source_shape}, schema={source_schema!r}"
                )
            else:
                report = transplant_policy_observation_inputs(
                    source_model,
                    model,
                    legacy_observation_size=OBSERVATION_V1_SIZE,
                )
                assert_legacy_policy_equivalence(
                    source_model,
                    model,
                    legacy_observation_size=OBSERVATION_V1_SIZE,
                    target_observation_size=observation_size(observation_version),
                )
                migration_message = (
                    "Migrated V1 policy without changing legacy outputs: "
                    f"{len(report.copied_tensors)} tensors copied, "
                    f"{len(report.expanded_tensors)} input tensors widened."
                )
        else:
            source_schema = getattr(
                source_model,
                "kof_observation_schema_id",
                None,
            )
            if source_shape != (OBSERVATION_V2_SIZE,):
                raise ValueError(
                    "V2->V3 migration source must use the 140-value "
                    f"observation, got {source_shape}"
                )
            if source_schema != OBSERVATION_SCHEMA_V2_ID:
                raise ValueError(
                    "V2->V3 migration requires a checkpoint stamped with "
                    f"{OBSERVATION_SCHEMA_V2_ID!r}, got {source_schema!r}"
                )
            report = transplant_v2_policy_to_v3(source_model, model)
            assert_v2_v3_policy_equivalence(source_model, model)
            migration_message = (
                "Migrated V2 policy to V3 with deterministic-equivalent "
                "neutral outputs: "
                f"{len(report.copied_tensors)} tensors copied, "
                f"{len(report.transformed_tensors)} first-layer tensors "
                "bias-folded/zeroed; 10,000 float32 samples passed."
            )
        model.num_timesteps = int(source_model.num_timesteps)
        model._n_updates = int(getattr(source_model, "_n_updates", 0))
        print(migration_message)
    else:
        model = create_new_model()
        stamp_model_contract(model)

    # MaskablePPO.load() restores the checkpoint's original seed. Explicitly
    # reseed both policy sampling and vector environments so paired B/C runs
    # share a seed while the three pilot pairs remain independent.
    model.set_random_seed(args.seed)
    print(f"PPO/vector RNG seed: {args.seed}", flush=True)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            reset_num_timesteps=resume_path is None and migrate_path is None,
            tb_log_name=tensorboard_run_name,
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
