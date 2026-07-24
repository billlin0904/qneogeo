"""OpenGL 觀賽器:目視檢查模型/腳本的實際行為(pygame + PyOpenGL)。

三種驅動模式(擇一):
    --model:         載入 MaskablePPO 模型自主決策(預設 deterministic,
                      --stochastic 改為抽樣)。mask 由 env 依 profile 提供。
    --fixed-actions:  照劇本播放動作序列(驗證 C++ 腳本用,不代表模型會)。
    --random / 無:    隨機或純待機。

Profile 差異(重要,常見誤解):
    --profile combo(預設):強制 action_repeat=1、載 combo state、
        依 --combo-scenario 的 mask 引導 —— 模型只能演出該 scenario。
        看模型「自由發揮」要配 --mask-level physical。
    --profile fight:實戰模式,配 --p2-training-ai 開啟 P2 對手、
        --action-repeat 4 對齊訓練條件。

快捷鍵:SPACE 暫停、R 重置、ESC 離開。--hitboxes 疊加判定框。
"""
from __future__ import annotations

import argparse
import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from kof98_env import (
    ActionMaskLevel,
    COMBO_SCENARIOS,
    DEFAULT_COMBO_SCENARIO_NAME,
    Kof98Env,
    P2Style,
    TrainingProfile,
)
from kof98_observation import (
    OBSERVATION_V1_SIZE,
    OBSERVATION_V2_SIZE,
    OBSERVATION_SCHEMA_V2_ID,
    OBSERVATION_SCHEMA_V3_ID,
    ObservationVersion,
)


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_fixed_actions(value: str) -> tuple[int, ...]:
    try:
        actions = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Fixed actions must be comma-separated integers.") from error

    if not actions:
        raise argparse.ArgumentTypeError("Fixed actions cannot be empty.")
    return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch KOF98 PPO output in an OpenGL window.")
    parser.add_argument("--root", type=Path, default=default_project_root(), help="qneogeo project root.")
    parser.add_argument("--model", type=Path, default=None, help="Stable-Baselines3 PPO .zip model.")
    parser.add_argument("--state", type=Path, default=None, help="Initial save state.")
    parser.add_argument(
        "--profile",
        choices=tuple(profile.value for profile in TrainingProfile),
        default=TrainingProfile.COMBO.value,
        help="Environment profile to watch.",
    )
    parser.add_argument(
        "--p2-training-ai",
        action="store_true",
        help="Enable the DLL's P2 training opponent. Intended for the fight profile.",
    )
    parser.add_argument(
        "--p2-style",
        choices=tuple(style.value for style in P2Style),
        default=P2Style.ONIYAKI.value,
        help="Fixed P2 behavior style used when the training opponent is enabled.",
    )
    parser.add_argument(
        "--combo-scenario",
        choices=tuple(sorted(COMBO_SCENARIOS)),
        default=DEFAULT_COMBO_SCENARIO_NAME,
        help="Combo scenario whose mask drives the model. Defaults to the dokugami chain.",
    )
    parser.add_argument(
        "--mask-level",
        choices=tuple(level.value for level in ActionMaskLevel),
        default=ActionMaskLevel.STRICT.value,
        help="strict locks the model to the scenario's next move; physical opens every action.",
    )
    parser.add_argument("--device", default="cuda", help="PPO inference device.")
    parser.add_argument("--action-repeat", type=int, default=6, help="Frames to run per action.")
    parser.add_argument("--scale", type=int, default=3, help="Initial window scale.")
    parser.add_argument("--fps", type=int, default=60, help="Viewer frame limit.")
    parser.add_argument("--frames", type=int, default=0, help="Auto-close after this many viewer frames. 0 means forever.")
    parser.add_argument(
        "--terminal-tail-frames",
        type=int,
        default=90,
        help="Continue emulation for this many frames after an episode ends before resetting.",
    )
    parser.add_argument("--random", action="store_true", help="Use random actions when no model is supplied.")
    parser.add_argument("--fixed-action", type=int, default=None, help="Always send this action id, ignoring model/random.")
    parser.add_argument(
        "--fixed-actions",
        type=parse_fixed_actions,
        default=None,
        help="Send a comma-separated action sequence once, e.g. 23,24,25.",
    )
    parser.add_argument(
        "--fixed-actions-loop",
        action="store_true",
        help="Reload the initial state and repeat --fixed-actions.",
    )
    parser.add_argument(
        "--fixed-actions-loop-delay",
        type=int,
        default=180,
        help="Frames to keep running after the final fixed action starts before reloading the state.",
    )
    parser.add_argument("--fixed-action-once", action="store_true", help="Send --fixed-action once after reset, then idle.")
    parser.add_argument("--fixed-action-ignore-ready", action="store_true", help="Send --fixed-action even while P1 is not in normal object state.")
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


@dataclass(frozen=True)
class InputDisplayState:
    up: bool
    down: bool
    left: bool
    right: bool
    a: bool
    b: bool
    c: bool
    d: bool

    @classmethod
    def from_joypad(cls, state) -> "InputDisplayState":
        return cls(
            bool(state.up),
            bool(state.down),
            bool(state.left),
            bool(state.right),
            bool(state.a),
            bool(state.b),
            bool(state.c),
            bool(state.d),
        )

    def active(self) -> bool:
        return any((self.up, self.down, self.left, self.right, self.a, self.b, self.c, self.d))


@dataclass
class InputHistoryEntry:
    state: InputDisplayState
    first_frame: int
    last_frame: int


class InputHistory:
    def __init__(self, capacity: int = 10):
        self.capacity = max(1, capacity)
        self.entries: list[InputHistoryEntry] = []

    def clear(self) -> None:
        self.entries.clear()

    def push(self, state, frame_number: int) -> None:
        if frame_number <= 0:
            return

        display_state = InputDisplayState.from_joypad(state)
        if self.entries and frame_number <= self.entries[-1].last_frame:
            self.clear()

        if (
            self.entries
            and self.entries[-1].state == display_state
            and frame_number == self.entries[-1].last_frame + 1
        ):
            self.entries[-1].last_frame = frame_number
            return

        self.entries.append(InputHistoryEntry(display_state, frame_number, frame_number))
        if len(self.entries) > self.capacity:
            del self.entries[:-self.capacity]


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
            GL_RGBA,
            GL_SRC_ALPHA,
            GL_TEXTURE_2D,
            GL_TEXTURE_MAG_FILTER,
            GL_TEXTURE_MIN_FILTER,
            GL_TEXTURE_WRAP_S,
            GL_TEXTURE_WRAP_T,
            GL_TRIANGLE_STRIP,
            GL_UNPACK_ALIGNMENT,
            GL_UNSIGNED_BYTE,
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
            "GL_RGBA": GL_RGBA,
            "GL_SRC_ALPHA": GL_SRC_ALPHA,
            "GL_TEXTURE_2D": GL_TEXTURE_2D,
            "GL_TEXTURE_MAG_FILTER": GL_TEXTURE_MAG_FILTER,
            "GL_TEXTURE_MIN_FILTER": GL_TEXTURE_MIN_FILTER,
            "GL_TEXTURE_WRAP_S": GL_TEXTURE_WRAP_S,
            "GL_TEXTURE_WRAP_T": GL_TEXTURE_WRAP_T,
            "GL_CLAMP_TO_EDGE": GL_CLAMP_TO_EDGE,
            "GL_TRIANGLE_STRIP": GL_TRIANGLE_STRIP,
            "GL_UNPACK_ALIGNMENT": GL_UNPACK_ALIGNMENT,
            "GL_UNSIGNED_BYTE": GL_UNSIGNED_BYTE,
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

        self.input_font_path = pygame.font.match_font("SF Mono", bold=True)
        input_font = pygame.font.Font(self.input_font_path, 10)
        self.frame_font = pygame.font.Font(self.input_font_path, 7)
        self.frame_textures = []
        for _ in range(10):
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            self.frame_textures.append(texture)

        self.input_label_textures = {}
        for label in "ABCD":
            surface = input_font.render(label, True, (13, 13, 13))
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA,
                surface.get_width(),
                surface.get_height(),
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                pygame.image.tobytes(surface, "RGBA", True),
            )
            self.input_label_textures[label] = (texture, surface.get_width(), surface.get_height())

    def draw(
        self,
        frame: Frame,
        overlay: Optional[HitboxOverlay] = None,
        input_history: Optional[list[InputHistoryEntry]] = None,
    ) -> None:
        gl = self.gl
        window_width, window_height = self.pygame.display.get_surface().get_size()
        gl["glViewport"](0, 0, window_width, window_height)
        gl["glClear"](gl["GL_COLOR_BUFFER_BIT"])

        gl["glColor4f"](1.0, 1.0, 1.0, 1.0)
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
        if input_history:
            self.draw_input_history(frame, input_history)

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
        gl["glColor4f"](1.0, 1.0, 1.0, 1.0)
        gl["glEnable"](gl["GL_TEXTURE_2D"])

    def draw_input_history(self, frame: Frame, history: list[InputHistoryEntry]) -> None:
        gl = self.gl

        def px(x: float) -> float:
            return x / max(1, frame.width) * 2.0 - 1.0

        def py(y: float) -> float:
            return 1.0 - y / max(1, frame.height) * 2.0

        def vertex(x: float, y: float) -> None:
            gl["glVertex2f"](px(x), py(y))

        visible = history[-10:]
        if not visible:
            return

        gl["glDisable"](gl["GL_TEXTURE_2D"])
        gl["glEnable"](0x0BE2)  # GL_BLEND
        gl["glBlendFunc"](gl["GL_SRC_ALPHA"], gl["GL_ONE_MINUS_SRC_ALPHA"])

        panel_top = 38.0
        panel_bottom = 216.0
        gl["glColor4f"](0.0, 0.0, 0.0, 0.42)
        gl["glBegin"](gl["GL_QUADS"])
        vertex(5.0, panel_top)
        vertex(160.0, panel_top)
        vertex(160.0, panel_bottom)
        vertex(5.0, panel_bottom)
        gl["glEnd"]()

        first_y = 198.0 - (len(visible) - 1) * 16.0
        button_colors = (
            (0.95, 0.12, 0.10, 0.95),
            (1.0, 0.78, 0.08, 0.95),
            (0.12, 0.78, 0.26, 0.95),
            (0.12, 0.42, 1.0, 0.95),
        )
        for row, entry in enumerate(visible):
            state = entry.state
            center_y = first_y + row * 16.0
            dx = int(state.right) - int(state.left)
            dy = int(state.down) - int(state.up)

            if entry.first_frame == entry.last_frame:
                frame_text = f"F{entry.first_frame:06d}"
            else:
                frame_text = f"F{entry.first_frame:06d}-{entry.last_frame:06d}"

            frame_surface = self.frame_font.render(frame_text, True, (235, 235, 235))
            frame_texture = self.frame_textures[row]
            frame_width = frame_surface.get_width()
            frame_height = frame_surface.get_height()
            gl["glEnable"](gl["GL_TEXTURE_2D"])
            gl["glBindTexture"](gl["GL_TEXTURE_2D"], frame_texture)
            gl["glTexImage2D"](
                gl["GL_TEXTURE_2D"],
                0,
                gl["GL_RGBA"],
                frame_width,
                frame_height,
                0,
                gl["GL_RGBA"],
                gl["GL_UNSIGNED_BYTE"],
                self.pygame.image.tobytes(frame_surface, "RGBA", True),
            )
            gl["glColor4f"](1.0, 1.0, 1.0, 1.0)
            gl["glBegin"](gl["GL_QUADS"])
            gl["glTexCoord2f"](0.0, 1.0)
            vertex(8.0, center_y - frame_height * 0.5)
            gl["glTexCoord2f"](1.0, 1.0)
            vertex(8.0 + frame_width, center_y - frame_height * 0.5)
            gl["glTexCoord2f"](1.0, 0.0)
            vertex(8.0 + frame_width, center_y + frame_height * 0.5)
            gl["glTexCoord2f"](0.0, 0.0)
            vertex(8.0, center_y + frame_height * 0.5)
            gl["glEnd"]()
            gl["glDisable"](gl["GL_TEXTURE_2D"])

            gl["glLineWidth"](2.5)
            gl["glColor4f"](1.0, 1.0, 1.0, 1.0)
            if dx or dy:
                length = max(1.0, (dx * dx + dy * dy) ** 0.5)
                ux = dx / length
                uy = dy / length
                tip_x = 92.0 + ux * 7.0
                tip_y = center_y + uy * 7.0
                tail_x = 92.0 - ux * 7.0
                tail_y = center_y - uy * 7.0
                perp_x = -uy
                perp_y = ux
                head_x = tip_x - ux * 4.0
                head_y = tip_y - uy * 4.0
                gl["glBegin"](gl["GL_LINES"])
                vertex(tail_x, tail_y)
                vertex(tip_x, tip_y)
                vertex(tip_x, tip_y)
                vertex(head_x + perp_x * 3.0, head_y + perp_y * 3.0)
                vertex(tip_x, tip_y)
                vertex(head_x - perp_x * 3.0, head_y - perp_y * 3.0)
                gl["glEnd"]()
            else:
                gl["glBegin"](gl["GL_QUADS"])
                vertex(90.5, center_y - 1.5)
                vertex(93.5, center_y - 1.5)
                vertex(93.5, center_y + 1.5)
                vertex(90.5, center_y + 1.5)
                gl["glEnd"]()

            for index, (active, label) in enumerate(zip((state.a, state.b, state.c, state.d), "ABCD")):
                if not active:
                    continue

                center_x = 109.0 + index * 13.0
                gl["glColor4f"](*button_colors[index])
                gl["glBegin"](gl["GL_QUADS"])
                vertex(center_x - 5.0, center_y - 5.0)
                vertex(center_x + 5.0, center_y - 5.0)
                vertex(center_x + 5.0, center_y + 5.0)
                vertex(center_x - 5.0, center_y + 5.0)
                gl["glEnd"]()

                texture, label_width, label_height = self.input_label_textures[label]
                label_left = center_x - label_width * 0.5
                label_right = center_x + label_width * 0.5
                label_top = center_y - label_height * 0.5
                label_bottom = center_y + label_height * 0.5
                gl["glEnable"](gl["GL_TEXTURE_2D"])
                gl["glBindTexture"](gl["GL_TEXTURE_2D"], texture)
                gl["glColor4f"](1.0, 1.0, 1.0, 1.0)
                gl["glBegin"](gl["GL_QUADS"])
                gl["glTexCoord2f"](0.0, 1.0)
                vertex(label_left, label_top)
                gl["glTexCoord2f"](1.0, 1.0)
                vertex(label_right, label_top)
                gl["glTexCoord2f"](1.0, 0.0)
                vertex(label_right, label_bottom)
                gl["glTexCoord2f"](0.0, 0.0)
                vertex(label_left, label_bottom)
                gl["glEnd"]()
                gl["glDisable"](gl["GL_TEXTURE_2D"])

        gl["glDisable"](0x0BE2)  # GL_BLEND
        gl["glColor4f"](1.0, 1.0, 1.0, 1.0)
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
    if args.fixed_actions_loop and args.fixed_actions is None:
        raise ValueError("--fixed-actions-loop requires --fixed-actions.")
    if args.fixed_actions_loop_delay < 0:
        raise ValueError("--fixed-actions-loop-delay cannot be negative.")

    root = args.root.resolve()
    state_path = resolve_state_path(root, args.state)
    model_path = args.model if args.model is None or args.model.is_absolute() else root / args.model
    model = None
    model_action_count = 29
    observation_version = ObservationVersion.V1
    if args.model:
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(str(model_path), device=args.device)
        model_action_count = int(model.action_space.n)
        model_observation_size = int(model.observation_space.shape[0])
        if model_observation_size == OBSERVATION_V1_SIZE:
            observation_version = ObservationVersion.V1
        elif model_observation_size == OBSERVATION_V2_SIZE:
            model_schema = getattr(
                model,
                "kof_observation_schema_id",
                OBSERVATION_SCHEMA_V2_ID,
            )
            if model_schema == OBSERVATION_SCHEMA_V3_ID:
                observation_version = ObservationVersion.V3
            elif model_schema == OBSERVATION_SCHEMA_V2_ID:
                observation_version = ObservationVersion.V2
            else:
                raise ValueError(
                    "A 140-value model must declare a supported observation "
                    f"schema, got {model_schema!r}"
                )
        else:
            raise ValueError(
                f"Unsupported model observation size: {model_observation_size}"
            )

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
        print(f"  observation: {observation_version.value}", flush=True)
        print(f"  profile: {args.profile}", flush=True)
        print(f"  P2 training AI: {'on' if args.p2_training_ai else 'off'}", flush=True)
        print(f"  P2 style: {args.p2_style}", flush=True)
        if args.fixed_actions is not None:
            print(f"  fixed actions: {','.join(str(action) for action in args.fixed_actions)}", flush=True)
            if args.fixed_actions_loop:
                print(f"  fixed actions loop delay: {args.fixed_actions_loop_delay} frames", flush=True)

    sink = FrameSink()
    input_history = InputHistory()
    startup_frame = Frame(pixels=bytes(320 * 224 * 2), width=320, height=224)
    viewer = OpenGlViewer(startup_frame.width, startup_frame.height, max(1, args.scale))
    viewer.draw(startup_frame)
    pygame = viewer.pygame
    clock = pygame.time.Clock()

    training_profile = TrainingProfile(args.profile)
    env = Kof98Env(
        dll_path=root / "build-vs2026-x64" / "Release" / "fbneo_training.dll",
        core_path=root / "downloads" / "fbneo_libretro" / "fbneo_libretro.dll",
        game_path=root / "roms" / "fbneo" / "kof98.zip",
        system_dir=root / "system",
        save_dir=root / "saves",
        combo_state_path=state_path if training_profile is TrainingProfile.COMBO else None,
        fight_state_path=state_path if training_profile is TrainingProfile.FIGHT else None,
        training_profile=training_profile,
        action_repeat=args.action_repeat,
        p2_training_ai=args.p2_training_ai,
        p2_style=P2Style(args.p2_style),
        combo_scenario=args.combo_scenario,
        action_mask_level=ActionMaskLevel(args.mask_level),
        observation_version=observation_version,
    )
    env.client.set_video_refresh_callback(sink.receive)

    if args.fixed_actions is not None:
        invalid_actions = [action for action in args.fixed_actions if not env.action_space.contains(action)]
        if invalid_actions:
            raise ValueError(f"Fixed action ids are out of range: {invalid_actions}")

    if model is not None:
        if model_action_count != env.action_space.n:
            print(
                f"Warning: model has {model_action_count} actions, environment has "
                f"{env.action_space.n}; viewer will trim the mask for compatibility.",
                flush=True,
            )

    obs, _ = env.reset()
    env.step(0)
    emulated_frame = env.action_repeat
    input_history.push(env.client.last_joypad(), emulated_frame)
    if sink.frame is None:
        print("Warning: no frame received from fbneo_training video callback yet.", flush=True)

    paused = False
    running = True
    rendered_frames = 0
    deterministic = not args.stochastic
    step_fps = max(1.0, float(args.fps) / max(1, env.action_repeat))
    fixed_action_sent = False
    fixed_action_index = 0
    fixed_sequence_last_action_started = False
    fixed_sequence_tail_remaining = 0
    terminal_tail_remaining = 0

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
                    input_history.clear()
                    emulated_frame = 0
                    fixed_action_sent = False
                    fixed_action_index = 0
                    fixed_sequence_last_action_started = False
                    fixed_sequence_tail_remaining = 0
                    terminal_tail_remaining = 0

        if not paused:
            if args.fixed_actions is not None:
                action_id = 0
                if fixed_action_index < len(args.fixed_actions):
                    next_action_id = args.fixed_actions[fixed_action_index]
                    can_send_action = (
                        env.client.input_ready()
                        or env.client.can_queue_action(next_action_id)
                    )
                    if can_send_action:
                        action_id = next_action_id
                        fixed_action_index += 1
                env.client.step(action_id, env.action_repeat)
                emulated_frame += env.action_repeat
                input_history.push(env.client.last_joypad(), emulated_frame)
                if (
                    args.fixed_actions_loop
                    and fixed_action_index >= len(args.fixed_actions)
                    and not fixed_sequence_last_action_started
                ):
                    action_status = env.client.action_status()
                    if action_status.last_started_action_id == args.fixed_actions[-1]:
                        fixed_sequence_last_action_started = True
                        fixed_sequence_tail_remaining = args.fixed_actions_loop_delay

                if fixed_sequence_last_action_started:
                    fixed_sequence_tail_remaining -= env.action_repeat
                    if fixed_sequence_tail_remaining <= 0 and env.client.input_ready():
                        obs, _ = env.reset()
                        input_history.clear()
                        emulated_frame = 0
                        fixed_action_index = 0
                        fixed_sequence_last_action_started = False
                        fixed_sequence_tail_remaining = 0
            elif terminal_tail_remaining > 0:
                env.client.step(0, env.action_repeat)
                emulated_frame += env.action_repeat
                input_history.push(env.client.last_joypad(), emulated_frame)
                terminal_tail_remaining -= env.action_repeat
                if terminal_tail_remaining <= 0:
                    obs, _ = env.reset()
                    input_history.clear()
                    emulated_frame = 0
                    fixed_action_sent = False
                    fixed_action_index = 0
            else:
                if args.fixed_action is not None:
                    if not args.fixed_action_ignore_ready and not p1_is_ready_for_fixed_action(env):
                        action_id = 0
                    elif args.fixed_action_once and fixed_action_sent:
                        action_id = 0
                    else:
                        action_id = int(args.fixed_action)
                        fixed_action_sent = True
                elif model is not None:
                    action_mask = env.action_masks()[:model_action_count]
                    action, _ = model.predict(
                        obs,
                        deterministic=deterministic,
                        action_masks=action_mask,
                    )
                    action_id = int(np.asarray(action).item())
                elif args.random:
                    action_id = int(env.action_space.sample())
                else:
                    action_id = 0

                obs, _reward, terminated, truncated, _info = env.step(action_id)
                emulated_frame += env.action_repeat
                input_history.push(env.client.last_joypad(), emulated_frame)
                if terminated or truncated:
                    terminal_tail_remaining = max(0, args.terminal_tail_frames)
                    if terminal_tail_remaining == 0:
                        obs, _ = env.reset()
                        input_history.clear()
                        emulated_frame = 0
                        fixed_action_sent = False

        frame = sink.frame if sink.frame is not None else startup_frame
        overlay = None
        if args.hitboxes:
            overlay = build_hitbox_overlay_from_client(env.client, frame.width, frame.height)
        viewer.draw(frame, overlay, input_history.entries)
        rendered_frames += 1
        if args.frames > 0 and rendered_frames >= args.frames:
            running = False

        clock.tick(step_fps)

    env.close()
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
