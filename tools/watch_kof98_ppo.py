from __future__ import annotations

import argparse
import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from kof98_env import Kof98Env


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch KOF98 PPO output in an OpenGL window.")
    parser.add_argument("--root", type=Path, default=default_project_root(), help="qneogeo project root.")
    parser.add_argument("--model", type=Path, default=None, help="Stable-Baselines3 PPO .zip model.")
    parser.add_argument("--state", type=Path, default=None, help="Initial save state.")
    parser.add_argument("--device", default="cuda", help="PPO inference device.")
    parser.add_argument("--action-repeat", type=int, default=6, help="Frames to run per action.")
    parser.add_argument("--scale", type=int, default=3, help="Initial window scale.")
    parser.add_argument("--fps", type=int, default=60, help="Viewer frame limit.")
    parser.add_argument("--frames", type=int, default=0, help="Auto-close after this many viewer frames. 0 means forever.")
    parser.add_argument("--random", action="store_true", help="Use random actions when no model is supplied.")
    parser.add_argument("--fixed-action", type=int, default=None, help="Always send this action id, ignoring model/random.")
    parser.add_argument("--fixed-action-once", action="store_true", help="Send --fixed-action once after reset, then idle.")
    parser.add_argument("--fixed-action-ignore-ready", action="store_true", help="Send --fixed-action even while P1 is not in normal object state.")
    parser.add_argument(
        "--p2-training-ai",
        dest="p2_training_ai",
        action="store_true",
        default=True,
        help="Make P2 repeatedly run the built-in training action. Enabled by default for the viewer.",
    )
    parser.add_argument(
        "--no-p2-training-ai",
        dest="p2_training_ai",
        action="store_false",
        help="Disable the built-in P2 training action.",
    )
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic model actions.")
    parser.add_argument("--show-paths", action="store_true", help="Print Python/OpenGL/runtime paths before launching.")
    parser.add_argument("--hitboxes", action="store_true", help="Draw KOF98 hitboxes from FBNeo system RAM.")
    return parser.parse_args()


@dataclass
class Frame:
    pixels: bytes
    width: int
    height: int


@dataclass
class HitboxRect:
    left: int
    top: int
    width: int
    height: int
    fill: tuple[float, float, float, float]
    outline: tuple[float, float, float, float]


@dataclass
class HitboxAxis:
    x: int
    y: int
    color: tuple[float, float, float, float]


@dataclass
class HitboxOverlay:
    boxes: list[HitboxRect]
    axes: list[HitboxAxis]


class FrameSink:
    def __init__(self):
        self.frame: Optional[Frame] = None

    def receive(self, data: int, width: int, height: int, pitch: int) -> None:
        if not data or width <= 0 or height <= 0 or pitch <= 0:
            return

        row_bytes = width * 2
        raw = ctypes.string_at(data, pitch * height)
        if pitch == row_bytes:
            pixels = raw
        else:
            view = memoryview(raw)
            pixels = b"".join(view[row * pitch:row * pitch + row_bytes] for row in range(height))

        self.frame = Frame(pixels=pixels, width=width, height=height)


class OpenGlViewer:
    def __init__(self, width: int, height: int, scale: int):
        import pygame
        from OpenGL.GL import (
            GL_CLAMP_TO_EDGE,
            GL_COLOR_BUFFER_BIT,
            GL_LINEAR,
            GL_LINE_LOOP,
            GL_LINES,
            GL_NEAREST,
            GL_ONE_MINUS_SRC_ALPHA,
            GL_QUADS,
            GL_RGB,
            GL_SRC_ALPHA,
            GL_TEXTURE_2D,
            GL_TEXTURE_MAG_FILTER,
            GL_TEXTURE_MIN_FILTER,
            GL_TEXTURE_WRAP_S,
            GL_TEXTURE_WRAP_T,
            GL_TRIANGLE_STRIP,
            GL_UNPACK_ALIGNMENT,
            GL_UNSIGNED_SHORT_5_6_5,
            glBlendFunc,
            glBegin,
            glBindTexture,
            glClear,
            glClearColor,
            glColor4f,
            glDisable,
            glEnable,
            glEnd,
            glGenTextures,
            glLineWidth,
            glPixelStorei,
            glTexCoord2f,
            glTexImage2D,
            glTexParameteri,
            glVertex2f,
            glViewport,
        )

        self.pygame = pygame
        self.gl = {
            "GL_COLOR_BUFFER_BIT": GL_COLOR_BUFFER_BIT,
            "GL_LINEAR": GL_LINEAR,
            "GL_LINE_LOOP": GL_LINE_LOOP,
            "GL_LINES": GL_LINES,
            "GL_NEAREST": GL_NEAREST,
            "GL_ONE_MINUS_SRC_ALPHA": GL_ONE_MINUS_SRC_ALPHA,
            "GL_QUADS": GL_QUADS,
            "GL_RGB": GL_RGB,
            "GL_SRC_ALPHA": GL_SRC_ALPHA,
            "GL_TEXTURE_2D": GL_TEXTURE_2D,
            "GL_TEXTURE_MAG_FILTER": GL_TEXTURE_MAG_FILTER,
            "GL_TEXTURE_MIN_FILTER": GL_TEXTURE_MIN_FILTER,
            "GL_TEXTURE_WRAP_S": GL_TEXTURE_WRAP_S,
            "GL_TEXTURE_WRAP_T": GL_TEXTURE_WRAP_T,
            "GL_CLAMP_TO_EDGE": GL_CLAMP_TO_EDGE,
            "GL_TRIANGLE_STRIP": GL_TRIANGLE_STRIP,
            "GL_UNPACK_ALIGNMENT": GL_UNPACK_ALIGNMENT,
            "GL_UNSIGNED_SHORT_5_6_5": GL_UNSIGNED_SHORT_5_6_5,
            "glBlendFunc": glBlendFunc,
            "glBegin": glBegin,
            "glBindTexture": glBindTexture,
            "glClear": glClear,
            "glClearColor": glClearColor,
            "glColor4f": glColor4f,
            "glDisable": glDisable,
            "glEnable": glEnable,
            "glEnd": glEnd,
            "glGenTextures": glGenTextures,
            "glLineWidth": glLineWidth,
            "glPixelStorei": glPixelStorei,
            "glTexCoord2f": glTexCoord2f,
            "glTexImage2D": glTexImage2D,
            "glTexParameteri": glTexParameteri,
            "glVertex2f": glVertex2f,
            "glViewport": glViewport,
        }

        pygame.init()
        pygame.display.set_caption("KOF98 PPO OpenGL Viewer")
        pygame.display.set_mode((width * scale, height * scale), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)

        glDisable(0x0B71)  # GL_DEPTH_TEST
        glEnable(GL_TEXTURE_2D)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def draw(self, frame: Frame, overlay: Optional[HitboxOverlay] = None) -> None:
        gl = self.gl
        window_width, window_height = self.pygame.display.get_surface().get_size()
        gl["glViewport"](0, 0, window_width, window_height)
        gl["glClear"](gl["GL_COLOR_BUFFER_BIT"])

        gl["glBindTexture"](gl["GL_TEXTURE_2D"], self.texture)
        gl["glTexImage2D"](
            gl["GL_TEXTURE_2D"],
            0,
            gl["GL_RGB"],
            frame.width,
            frame.height,
            0,
            gl["GL_RGB"],
            gl["GL_UNSIGNED_SHORT_5_6_5"],
            frame.pixels,
        )

        gl["glBegin"](gl["GL_TRIANGLE_STRIP"])
        gl["glTexCoord2f"](0.0, 1.0)
        gl["glVertex2f"](-1.0, -1.0)
        gl["glTexCoord2f"](1.0, 1.0)
        gl["glVertex2f"](1.0, -1.0)
        gl["glTexCoord2f"](0.0, 0.0)
        gl["glVertex2f"](-1.0, 1.0)
        gl["glTexCoord2f"](1.0, 0.0)
        gl["glVertex2f"](1.0, 1.0)
        gl["glEnd"]()

        if overlay is not None:
            self.draw_overlay(frame, overlay)

        self.pygame.display.flip()

    def draw_overlay(self, frame: Frame, overlay: HitboxOverlay) -> None:
        gl = self.gl

        def px(x: float) -> float:
            return x / max(1, frame.width) * 2.0 - 1.0

        def py(y: float) -> float:
            return 1.0 - y / max(1, frame.height) * 2.0

        gl["glDisable"](gl["GL_TEXTURE_2D"])
        gl["glEnable"](0x0BE2)  # GL_BLEND
        gl["glBlendFunc"](gl["GL_SRC_ALPHA"], gl["GL_ONE_MINUS_SRC_ALPHA"])

        for box in overlay.boxes:
            left = px(box.left)
            right = px(box.left + box.width)
            top = py(box.top)
            bottom = py(box.top + box.height)

            gl["glColor4f"](*box.fill)
            gl["glBegin"](gl["GL_QUADS"])
            gl["glVertex2f"](left, top)
            gl["glVertex2f"](right, top)
            gl["glVertex2f"](right, bottom)
            gl["glVertex2f"](left, bottom)
            gl["glEnd"]()

            gl["glLineWidth"](2.5)
            gl["glColor4f"](*box.outline)
            gl["glBegin"](gl["GL_LINE_LOOP"])
            gl["glVertex2f"](left, top)
            gl["glVertex2f"](right, top)
            gl["glVertex2f"](right, bottom)
            gl["glVertex2f"](left, bottom)
            gl["glEnd"]()

        gl["glLineWidth"](2.0)
        for axis in overlay.axes:
            x = px(axis.x)
            y = py(axis.y)
            dx = 8.0 / max(1, frame.width) * 2.0
            dy = 8.0 / max(1, frame.height) * 2.0
            gl["glColor4f"](*axis.color)
            gl["glBegin"](gl["GL_LINES"])
            gl["glVertex2f"](x - dx, y)
            gl["glVertex2f"](x + dx, y)
            gl["glVertex2f"](x, y - dy)
            gl["glVertex2f"](x, y + dy)
            gl["glEnd"]()

        gl["glDisable"](0x0BE2)  # GL_BLEND
        gl["glEnable"](gl["GL_TEXTURE_2D"])


HITBOX_ATTACK = 1
HITBOX_VULNERABILITY = 2
HITBOX_PROJECTILE_VULNERABILITY = 3
HITBOX_PROJECTILE_ATTACK = 4
HITBOX_PUSH = 5
HITBOX_GUARD = 6


def _palette(hitbox_type: int) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    if hitbox_type == HITBOX_ATTACK:
        return (1.0, 0.0, 0.0, 0.25), (1.0, 0.0, 0.0, 1.0)
    if hitbox_type == HITBOX_PROJECTILE_ATTACK:
        return (1.0, 0.5, 0.0, 0.25), (1.0, 0.5, 0.0, 1.0)
    if hitbox_type == HITBOX_VULNERABILITY:
        return (0.0, 0.25, 1.0, 0.2), (0.0, 0.25, 1.0, 1.0)
    if hitbox_type == HITBOX_PROJECTILE_VULNERABILITY:
        return (0.0, 0.85, 1.0, 0.2), (0.0, 0.85, 1.0, 1.0)
    if hitbox_type == HITBOX_PUSH:
        return (0.0, 1.0, 0.0, 0.16), (0.0, 1.0, 0.0, 1.0)
    if hitbox_type == HITBOX_GUARD:
        return (1.0, 1.0, 0.0, 0.2), (1.0, 1.0, 0.0, 1.0)
    return (0.62, 0.62, 0.62, 0.16), (0.62, 0.62, 0.62, 1.0)


def build_hitbox_overlay_from_client(client, source_width: int, source_height: int) -> HitboxOverlay:
    rects, axes = client.get_hitbox_overlay(source_width, source_height)
    boxes: list[HitboxRect] = []
    for rect in rects:
        fill, outline = _palette(rect.type)
        boxes.append(HitboxRect(rect.left, rect.top, rect.width, rect.height, fill, outline))

    return HitboxOverlay(
        boxes=boxes,
        axes=[HitboxAxis(axis.x, axis.y, (1.0, 1.0, 1.0, 1.0)) for axis in axes],
    )


def p1_is_ready_for_fixed_action(env: Kof98Env) -> bool:
    return env.client.p1_ready_for_action()


def resolve_state_path(root: Path, state: Optional[Path]) -> Optional[Path]:
    if state is None:
        candidate = root / "saves" / "states" / "kof98.slot1.state"
        return candidate if candidate.exists() else None

    return state if state.is_absolute() else root / state


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    state_path = resolve_state_path(root, args.state)
    model_path = args.model if args.model is None or args.model.is_absolute() else root / args.model

    if args.show_paths:
        import OpenGL
        import pygame

        print("KOF98 OpenGL viewer paths:", flush=True)
        print(f"  python: {sys.executable}", flush=True)
        print(f"  script: {Path(__file__).resolve()}", flush=True)
        print(f"  project root: {root}", flush=True)
        print(f"  PyOpenGL: {Path(OpenGL.__file__).resolve()}", flush=True)
        print(f"  pygame: {Path(pygame.__file__).resolve()}", flush=True)
        print(f"  fbneo_training.dll: {root / 'build-vs2026-x64' / 'Release' / 'fbneo_training.dll'}", flush=True)
        print(f"  fbneo_libretro.dll: {root / 'downloads' / 'fbneo_libretro' / 'fbneo_libretro.dll'}", flush=True)
        print(f"  game: {root / 'roms' / 'fbneo' / 'kof98.zip'}", flush=True)
        print(f"  state: {state_path if state_path else '(none)'}", flush=True)
        print(f"  model: {model_path if model_path else '(none)'}", flush=True)

    sink = FrameSink()
    startup_frame = Frame(pixels=bytes(320 * 224 * 2), width=320, height=224)
    viewer = OpenGlViewer(startup_frame.width, startup_frame.height, max(1, args.scale))
    viewer.draw(startup_frame)
    pygame = viewer.pygame
    clock = pygame.time.Clock()

    env = Kof98Env(
        dll_path=root / "build-vs2026-x64" / "Release" / "fbneo_training.dll",
        core_path=root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll",
        game_path=root / "roms" / "fbneo" / "kof98.zip",
        system_dir=root / "system",
        save_dir=root / "saves",
        state_path=state_path,
        action_repeat=args.action_repeat,
        p2_training_ai=args.p2_training_ai,
    )
    env.client.set_video_refresh_callback(sink.receive)

    model = None
    if args.model:
        from stable_baselines3 import PPO

        model = PPO.load(str(model_path), device=args.device)

    obs, _ = env.reset()
    env.step(0)
    if sink.frame is None:
        print("Warning: no frame received from fbneo_training video callback yet.", flush=True)

    paused = False
    running = True
    rendered_frames = 0
    deterministic = not args.stochastic
    step_fps = max(1.0, float(args.fps) / max(1, args.action_repeat))
    fixed_action_sent = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    fixed_action_sent = False

        if not paused:
            if args.fixed_action is not None:
                if not args.fixed_action_ignore_ready and not p1_is_ready_for_fixed_action(env):
                    action_id = 0
                elif args.fixed_action_once and fixed_action_sent:
                    action_id = 0
                else:
                    action_id = int(args.fixed_action)
                    fixed_action_sent = True
            elif model is not None:
                action, _ = model.predict(obs, deterministic=deterministic)
                action_id = int(np.asarray(action).item())
            elif args.random:
                action_id = int(env.action_space.sample())
            else:
                action_id = 0

            obs, _reward, terminated, truncated, _info = env.step(action_id)
            if terminated or truncated:
                obs, _ = env.reset()

        frame = sink.frame if sink.frame is not None else startup_frame
        overlay = None
        if args.hitboxes:
            overlay = build_hitbox_overlay_from_client(env.client, frame.width, frame.height)
        viewer.draw(frame, overlay)
        rendered_frames += 1
        if args.frames > 0 and rendered_frames >= args.frames:
            running = False

        clock.tick(step_fps)

    env.close()
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
