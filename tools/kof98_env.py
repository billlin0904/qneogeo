"""KOF98 PPO 訓練環境:fbneo_training.dll 的 ctypes 橋接 + Gymnasium 環境。

整體架構(三層):
    PPO (stable-baselines3)
        ↓ 29 個離散高階 Action(Discrete(29))
    Kof98Env(本檔)—— 觀測向量化、reward 計算、action mask、連段課程
        ↓ ctypes
    fbneo_training.dll(C++)—— 逐幀輸入腳本、派生技 queue、RAM 讀取、P2 AI

兩種訓練 profile:
    COMBO:訓練房。固定 save state 起手、每幀決策(action_repeat=1)、
           按 ComboScenario 指定的連段路線逐 phase 給獎勵。
    FIGHT:實戰。每 4 幀決策(repeat4 preset)、對抗 P2 AI、
           reward 以傷害/連段/勝負為主。

三段 mask 課程(strict → guided → physical):
    strict:  只開放「當前 phase 的正確招 + Idle」,等於洩題。
    guided:  正確招 + 2 個物理合法的干擾招,學會抗干擾。
    physical:29 招全開(僅過濾物理不可能的),與實戰條件相同。
    注意:mask 本身就是「任務標籤」——physical 拿掉標籤後,觀測相同的
    多套 scenario 會混疊(policy 對同一畫面只能出一條線),詳見訓練紀錄。

關鍵時序知識(除錯前必讀):
    - 近C 的取消窗口在第 5 幀就把派生技發射,但近C 的命中要到第 7 幀
      才入帳 —— 中間的「盲區」內 queue 的招,追蹤邏輯必須往後多看
      好幾個 phase(_queueable_followup_phase 掃描全部剩餘 phase)。
    - 傷害歸因不能用「本 step 選的 action」:該幀造成傷害的可能是
      更早 queue 的招。C++ 端以 StepEvent 逐幀記錄真正命中的招。
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from kof98_observation import (
    CombatPhaseMachine,
    ConfirmPhase,
    DefensePhase,
    GenericCombatStateMachine,
    OBSERVATION_V3_REPURPOSED_INDICES,
    ObservationContext,
    ObservationVersion,
    base_observation_vector,
    encode_observation,
    observation_size,
)
from kof98_curriculum import (
    CurriculumTask,
    LevelRecipe,
    RewardMachinePhase,
    TacticalRewardMachine,
    prepare_level,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - kept for quick local smoke tests.
    gym = None
    spaces = None


# ---------------------------------------------------------------------------
# ctypes 結構體:欄位順序與型別必須和 fbneo_training.h 完全一致(ABI)。
# C++ 端改結構體時,這裡要同步改,否則會靜默讀到錯位的資料。
# ---------------------------------------------------------------------------


class JoypadState(ctypes.Structure):
    _fields_ = [
        ("up", ctypes.c_uint8),
        ("down", ctypes.c_uint8),
        ("left", ctypes.c_uint8),
        ("right", ctypes.c_uint8),
        ("a", ctypes.c_uint8),
        ("b", ctypes.c_uint8),
        ("c", ctypes.c_uint8),
        ("d", ctypes.c_uint8),
        ("start", ctypes.c_uint8),
        ("coin", ctypes.c_uint8),
    ]


class Kof98Observation(ctypes.Structure):
    _fields_ = [
        ("round_time", ctypes.c_int32),
        ("p1_health", ctypes.c_int32),
        ("p2_health", ctypes.c_int32),
        ("p1_power", ctypes.c_int32),
        ("p2_power", ctypes.c_int32),
        ("p1_power_state", ctypes.c_int32),
        ("p2_power_state", ctypes.c_int32),
        ("p1_advanced_power_value", ctypes.c_int32),
        ("p1_advanced_power_stocks", ctypes.c_int32),
        ("p2_advanced_power_value", ctypes.c_int32),
        ("p2_advanced_power_stocks", ctypes.c_int32),
        ("p1_stun", ctypes.c_int32),
        ("p2_stun", ctypes.c_int32),
        ("p1_combo_count", ctypes.c_int32),
        ("p2_combo_count", ctypes.c_int32),
        ("p1_x", ctypes.c_int32),
        ("p1_y", ctypes.c_int32),
        ("p2_x", ctypes.c_int32),
        ("p2_y", ctypes.c_int32),
        ("distance_x", ctypes.c_int32),
        ("distance_y", ctypes.c_int32),
        ("p1_has_position", ctypes.c_uint8),
        ("p2_has_position", ctypes.c_uint8),
    ]


class HitboxRectResult(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int32),
        ("owner", ctypes.c_int32),
        ("left", ctypes.c_int32),
        ("top", ctypes.c_int32),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
    ]


class HitboxAxisResult(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
    ]


class ActionStatus(ctypes.Structure):
    """C++ 端動作生命週期的快照。

    active_action_id:      正在播放輸入腳本的招(-1 = 空閒)。
    queued_action_id:      已被 canQueueAction 接受、等待觸發幀的派生技。
    last_started_action_id:最後一個實際發動的招。注意這個值是「黏的」——
                           招式結束後仍保留,判斷時要配合其他欄位。
    action_accepted:       本次 setAction 是否被接受(busy 時只有
                           Idle 或合法 queue 會被接受)。
    命中與傷害歸因由 StepEvents 負責。ActionStatus 只描述輸入腳本
    的生命週期，避免同一事件在兩套 ABI 中產生不同答案。
    """

    _fields_ = [
        ("active_action_id", ctypes.c_int32),
        ("queued_action_id", ctypes.c_int32),
        ("last_started_action_id", ctypes.c_int32),
        ("action_accepted", ctypes.c_uint8),
    ]


STRATEGY_STATE_VERSION_1 = 1
COMBAT_TIMING_STATE_VERSION_1 = 1


class StrategyStateV1(ctypes.Structure):
    """StrategyV2 使用的版本化唯讀快照。欄位與 C ABI 完全對齊。

    p1_status/p2_status 是既有 ABI 名稱，實際內容為 hitbox slot active
    mask，不能解讀成 blockstun 或角色 animation state。
    """

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("p1_status", ctypes.c_int32),
        ("p2_status", ctypes.c_int32),
        ("p1_active_action_id", ctypes.c_int32),
        ("p1_queued_action_id", ctypes.c_int32),
        ("p1_last_started_action_id", ctypes.c_int32),
        ("p1_action_elapsed_frames", ctypes.c_int32),
        ("p1_action_remaining_frames", ctypes.c_int32),
        ("p2_active_action_id", ctypes.c_int32),
        ("p2_queued_action_id", ctypes.c_int32),
        ("p2_action_elapsed_frames", ctypes.c_int32),
        ("p2_action_remaining_frames", ctypes.c_int32),
        ("p1_input_ready", ctypes.c_uint8),
        ("p1_ready", ctypes.c_uint8),
        ("p2_input_ready", ctypes.c_uint8),
        ("p2_ready", ctypes.c_uint8),
        ("p1_facing_left", ctypes.c_uint8),
        ("p2_facing_left", ctypes.c_uint8),
        ("p2_scripted", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),
    ]


class PlayerTimingStateV1(ctypes.Structure):
    """單一玩家的通用 timing state。

    input_script_* 只描述 DLL 輸入腳本；reaction_* 是經 RAM 保守追蹤
    的受擊/防禦反應。actionable/recovery 必須先檢查 valid。
    """

    _fields_ = [
        ("input_script_remaining", ctypes.c_int32),
        ("reaction_kind", ctypes.c_int32),
        ("reaction_remaining", ctypes.c_int32),
        ("recovery_remaining", ctypes.c_int32),
        ("input_script_ready", ctypes.c_uint8),
        ("reaction_valid", ctypes.c_uint8),
        ("actionable_valid", ctypes.c_uint8),
        ("actionable", ctypes.c_uint8),
        ("recovery_valid", ctypes.c_uint8),
        ("reaction_remaining_valid", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 2),
    ]


class CombatTimingStateV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("engine_frame", ctypes.c_uint64),
        ("event_epoch", ctypes.c_uint32),
        ("frame_advantage", ctypes.c_int32),
        ("frame_advantage_valid", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
        ("p1", PlayerTimingStateV1),
        ("p2", PlayerTimingStateV1),
    ]


# ---------------------------------------------------------------------------
# StepEvents:C++ 端逐幀記錄的事件流,一個 step(可能跨多幀)內發生了
# 什麼、由哪個招造成。這是傷害歸因的權威來源 —— 一個 step 內可能同時有
# 「舊招命中 + 新招發動」,只看 step 前後的 HP/combo 差值無法分辨是誰打的。
#   ACTION_STARTED:某招在該幀實際發動(帶 action_serial 流水號,
#                   區分同一招的多次施放)。
#   COMBO_HIT:     P1 combo 計數上升的幀(真連段命中;被擋不會觸發)。
#   DAMAGE_ONLY:   P2 掉血但 combo 未增加(chip 傷害/削血)。
# ---------------------------------------------------------------------------
STEP_EVENTS_VERSION_1 = 1
STEP_EVENT_CAPACITY_V1 = 16
STEP_EVENT_ACTION_STARTED = 1
STEP_EVENT_COMBO_HIT = 2
STEP_EVENT_DAMAGE_ONLY = 3
STEP_EVENT_BLOCK_CONTACT = 4
STEP_EVENT_P1_DAMAGE = 5
STEP_EVENT_CHIP_DAMAGE = 6
STEP_EVENT_CLEAN_HIT = 7
STEP_EVENT_MANUAL_BLOCK_SUCCESS = 8
STEP_EVENT_AUTO_GUARD = 9
STEP_EVENT_BLOCKSTUN_STARTED = 10
STEP_EVENT_BLOCKSTUN_ENDED = 11
STEP_EVENTS_VERSION_2 = 2
STEP_EVENT_CAPACITY_V2 = 16
STEP_EVENTS_VERSION_3 = 3
STEP_EVENT_CAPACITY_V3 = 16
STEP_EVENTS_VERSION_4 = 4
STEP_EVENT_CAPACITY_V4 = 16
STEP_EVENTS_VERSION_5 = 5
STEP_EVENT_CAPACITY_V5 = 32
MOVE_DATA_VERSION_1 = 1


class StepEventV1(ctypes.Structure):
    _fields_ = [
        ("frame_offset", ctypes.c_int32),
        ("event_type", ctypes.c_int32),
        ("action_id", ctypes.c_int32),
        ("action_serial", ctypes.c_uint32),
        ("combo_before", ctypes.c_int32),
        ("combo_after", ctypes.c_int32),
        ("p2_hp_delta", ctypes.c_int32),
    ]


class StepEventsV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("event_count", ctypes.c_uint32),
        ("dropped_event_count", ctypes.c_uint32),
        ("events", StepEventV1 * STEP_EVENT_CAPACITY_V1),
    ]


class StepEventV2(ctypes.Structure):
    _fields_ = [
        ("frame_offset", ctypes.c_int32),
        ("event_type", ctypes.c_int32),
        ("action_id", ctypes.c_int32),
        ("action_serial", ctypes.c_uint32),
        ("combo_before", ctypes.c_int32),
        ("combo_after", ctypes.c_int32),
        ("p1_hp_delta", ctypes.c_int32),
        ("p2_hp_delta", ctypes.c_int32),
        ("target_y_at_event", ctypes.c_int32),
        ("target_airborne_at_event", ctypes.c_uint8),
        ("hit_contact", ctypes.c_uint8),
        ("block_contact", ctypes.c_uint8),
        ("target_airborne_after_event", ctypes.c_uint8),
    ]


class StepEventsV2(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("event_count", ctypes.c_uint32),
        ("dropped_event_count", ctypes.c_uint32),
        ("events", StepEventV2 * STEP_EVENT_CAPACITY_V2),
    ]


class StepEventV3(ctypes.Structure):
    _fields_ = [
        ("frame_offset", ctypes.c_int32),
        ("event_type", ctypes.c_int32),
        ("action_id", ctypes.c_int32),
        ("action_serial", ctypes.c_uint32),
        ("combo_before", ctypes.c_int32),
        ("combo_after", ctypes.c_int32),
        ("p1_hp_delta", ctypes.c_int32),
        ("p2_hp_delta", ctypes.c_int32),
        ("target_y_at_event", ctypes.c_int32),
        ("target_airborne_at_event", ctypes.c_uint8),
        ("hit_contact", ctypes.c_uint8),
        ("block_contact", ctypes.c_uint8),
        ("target_airborne_after_event", ctypes.c_uint8),
        ("p1_hit_guard_stop_before", ctypes.c_int32),
        ("p1_hit_guard_stop_after", ctypes.c_int32),
        ("p2_hit_guard_stop_before", ctypes.c_int32),
        ("p2_hit_guard_stop_after", ctypes.c_int32),
        ("action_elapsed_frames_at_event", ctypes.c_int32),
        ("expected_blockstun_frames", ctypes.c_int32),
        ("expected_blockstun_source", ctypes.c_int32),
    ]


class StepEventsV3(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("event_count", ctypes.c_uint32),
        ("dropped_event_count", ctypes.c_uint32),
        ("events", StepEventV3 * STEP_EVENT_CAPACITY_V3),
    ]


class StepEventV4(ctypes.Structure):
    _fields_ = StepEventV3._fields_ + [
        ("event_epoch", ctypes.c_uint32),
        ("guard_reaction_serial", ctypes.c_uint32),
    ]


class StepEventsV4(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("event_count", ctypes.c_uint32),
        ("dropped_event_count", ctypes.c_uint32),
        ("events", StepEventV4 * STEP_EVENT_CAPACITY_V4),
    ]


class StepEventV5(ctypes.Structure):
    _fields_ = StepEventV4._fields_ + [
        ("source_player", ctypes.c_int32),
        ("target_player", ctypes.c_int32),
        ("absolute_engine_frame", ctypes.c_uint64),
    ]


class StepEventsV5(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("event_count", ctypes.c_uint32),
        ("dropped_event_count", ctypes.c_uint32),
        ("batch_event_epoch", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("events", StepEventV5 * STEP_EVENT_CAPACITY_V5),
    ]


class KyoMoveDataV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("action_id", ctypes.c_int32),
        ("variant", ctypes.c_int32),
        ("move_class", ctypes.c_int32),
        ("startup_frames", ctypes.c_int32),
        ("active_frames", ctypes.c_int32),
        ("recovery_frames", ctypes.c_int32),
        ("reach_front", ctypes.c_int32),
        ("reach_back", ctypes.c_int32),
        ("movement_forward", ctypes.c_int32),
        ("attack_y_min", ctypes.c_int32),
        ("attack_y_max", ctypes.c_int32),
        ("anti_ground_small_jump_y", ctypes.c_int32),
        ("anti_ground_normal_jump_y", ctypes.c_int32),
        ("ground_blockstun_frames", ctypes.c_int32),
        ("air_blockstun_frames", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("source", ctypes.c_int32),
    ]


VideoRefreshCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_size_t,
    ctypes.c_void_p,
)

# ---------------------------------------------------------------------------
# Fight reward 常數。調整前先讀:每一項都有明確目的,量級之間互相制衡。
# 歷史教訓:HP 獎勵太大會讓「散打騙傷害」勝過連段;per-step 的 overlap
# 獎勵會與 HP 重複計分;cancel 獎勵沒有 combo 門檻會被二段戳刷分。
# ---------------------------------------------------------------------------
HITBOX_ATTACK = 1
HITBOX_VULNERABILITY = 2
HITBOX_PROJECTILE_VULNERABILITY = 3
HITBOX_PROJECTILE_ATTACK = 4

HITBOX_OWNER_P1 = 1
HITBOX_OWNER_P2 = 2
HITBOX_REWARD_SOURCE_WIDTH = 320
HITBOX_REWARD_SOURCE_HEIGHT = 224
# Edge-triggered (False -> True transition only): per-step overlap payouts
# double-count with the hp reward once the attack actually connects.
P1_ATTACK_OVERLAP_REWARD = 0.5
P2_ATTACK_OVERLAP_PENALTY = 0.5
# 距離塑形:鼓勵停留在有效攻擊距離(35~85 px),距離 >110 開始線性
# 罰款、180 達罰滿。量級刻意極小(0.02/step),只當探索方向盤不當主收入。
EFFECTIVE_DISTANCE_MIN = 35
EFFECTIVE_DISTANCE_MAX = 85
DISTANCE_TOO_FAR_START = 110
DISTANCE_MAX_PENALTY_AT = 180
DISTANCE_IN_RANGE_REWARD = 0.02
DISTANCE_FAR_PENALTY = 0.04
ACTION_COUNT = 29
EXPECTED_DLL_API_VERSION = 3
ACTION_SET_VERSION = 2
P1_HOLD_CHUNK_FRAMES = 4
GUARD_ACTION_IDS = {2, 3, 4}
P1_HOLD_ACTION_IDS = {1, 2, 3, 4}
DEFENSE_PRESSURE_MARGIN = 28
DEFENSE_GUARD_REWARD = 0.08
DEFENSE_UNGUARDED_PRESSURE_PENALTY = 0.04
DEFENSE_BAD_GUARD_PENALTY = 0.08


class FightRewardVersion(str, Enum):
    LEGACY_COMBO4 = "combo4_milestone_v3"
    SYMMETRIC_V2 = "symmetric_hp_outcome_v2"
    SYMMETRIC_TACTICAL_V3 = "symmetric_tactical_rm_v3"


SYMMETRIC_DAMAGE_ROUND_VALUE = 10.0
SYMMETRIC_OUTCOME_REWARD = 10.0
TACTICAL_REWARD_MACHINE_SUCCESS = 3.0
TACTICAL_REWARD_MACHINE_FAILURE = -0.5
TACTICAL_REWARD_MACHINE_TIMEOUT = -0.5
# Escalating per-hit combo rewards: hit 1 is paid through hp damage, later
# hits pay increasingly more so continuing a chain beats resetting to neutral
# even under KOF98 damage scaling. Hits past 5 pay the cap.
FIGHT_COMBO_HIT_REWARDS = {2: 1.0, 3: 2.0, 4: 3.5, 5: 5.0}
FIGHT_COMBO_HIT_REWARD_CAP = 6.0
# 每回合首次達成 4 hit 的一次性里程碑獎金(每 episode 只發一次,
# 防止用多段技反覆刷)。
FIGHT_COMBO_4PLUS_MILESTONE_HITS = 4
FIGHT_COMBO_4PLUS_MILESTONE_REWARD = 8.0
SUPER_COMBO_BONUS = 3.0
SUPER_ACTION_IDS = {18, 19}
SUPER_POWER_STOCKS_REQUIRED = 1
SUPER_NO_STOCK_PENALTY = 0.5
ONIYAKI_ACTION_ID = 16
P2_AIRBORNE_Y_THRESHOLD = 185
ONIYAKI_ANTI_AIR_BONUS = 1.0
# 攻擊安全性:出招(特殊技以上,id>=14)後開一個 24 幀的觀察窗 ——
# 窗內被打 = 被確反(重罰 4.0);沒被打且距離安全 = 小獎。
# 教「不要亂揮大招」。窗長以幀計,和 action_repeat 解耦。
ATTACK_RISK_ACTION_IDS = set(range(14, ACTION_COUNT))
ATTACK_RISK_WINDOW_FRAMES = 24
ATTACK_RISK_CLOSE_DISTANCE = 55
ATTACK_RISK_PUNISH_PENALTY = 4.0
ATTACK_RISK_UNSAFE_CLOSE_PENALTY = 0.12
ATTACK_RISK_SAFE_REWARD = 0.05
# KO 時依剩餘時間比例給的快勝獎金(回合級信號)。
FAST_WIN_BONUS_MAX = 15.0
# Per-step shaping terms were tuned at action_repeat=6; scale them by
# action_repeat / 6 so reward accrued per emulated second stays constant.
FIGHT_SHAPING_BASELINE_FRAMES = 6.0
FIGHT_FOLLOWUP_TRACK_WINDOW_FRAMES = 180
DEFENSE_COUNTER_WINDOW_FRAMES = 36
DEFENSE_BLOCKSTUN_PENDING_TIMEOUT_FRAMES = 240
TACTICAL_HIT_CONFIRM_WINDOW_FRAMES = 180
TACTICAL_OPENER_CONFIRM_WINDOW_FRAMES = 32
TACTICAL_DEFENSE_DISTANCE = 110
TACTICAL_ANTI_AIR_DISTANCE = 130
TACTICAL_EFFECTIVE_DISTANCE = 90
TACTICAL_APPROACH_MIN_P1_PROGRESS = 12.0

# ---------------------------------------------------------------------------
# Action id 對照(高階動作,C++ 端負責展開成逐幀輸入):
#   0 Idle | 1 前進 | 2-4 防禦 | 5-13 普攻(站/蹲 A~D)
#   14 荒咬 | 15 琴月陽 | 16 鬼燒 | 17 R.E.D. Kick | 18 大蛇薙(超)
#   19 無式(超) | 22 前B(75式) | 23 毒咬 | 24 罪詠 | 25 罰詠
#   26 七十五式改 | 27 八錆 | 28 砌穿
# ---------------------------------------------------------------------------
IDLE_ACTION_ID = 0
FORWARD_ACTION_ID = 1
CLOSE_C_ACTION_ID = 8
CROUCH_A_ACTION_ID = 10
CROUCH_B_ACTION_ID = 11
ARAGAMI_ACTION_ID = 14
KOTOTSUKI_YOU_ACTION_ID = 15
RED_KICK_ACTION_ID = 17
OROCHINAGI_ACTION_ID = 18
MUSHIKI_ACTION_ID = 19
FORWARD_B_ACTION_ID = 22
POISON_BITE_ACTION_ID = 23
TSUMI_YOMI_ACTION_ID = 24
BATSU_YOMI_ACTION_ID = 25
SEVENTY_FIVE_SHIKI_KAI_ACTION_ID = 26
YANO_SABI_ACTION_ID = 27
MIGIRI_UGACHI_ACTION_ID = 28

COMBO_CLOSE_DISTANCE = 45
COMBO_PHASE_AGE_SCALE_FRAMES = 30.0
COMBO_WRONG_ACTION_PENALTY = 0.1
GUIDED_DISTRACTOR_COUNT = 2

SEVENTY_FIVE_KAI_FINISHER_ACTION_IDS = (
    ARAGAMI_ACTION_ID,
    KOTOTSUKI_YOU_ACTION_ID,
    RED_KICK_ACTION_ID,
    OROCHINAGI_ACTION_ID,
)
SEVENTY_FIVE_KAI_FINISHER_HIT_TIMEOUTS = {
    ARAGAMI_ACTION_ID: 130,
    KOTOTSUKI_YOU_ACTION_ID: 150,
    RED_KICK_ACTION_ID: 120,
    OROCHINAGI_ACTION_ID: 160,
}
SEVENTY_FIVE_KAI_ALTERNATE_REWARD = 8.0


def seventy_five_kai_alternates(designated_action_id: int) -> tuple[tuple[int, float], ...]:
    return tuple(
        (action_id, SEVENTY_FIVE_KAI_ALTERNATE_REWARD)
        for action_id in SEVENTY_FIVE_KAI_FINISHER_ACTION_IDS
        if action_id != designated_action_id
    )


@dataclass(frozen=True)
class ComboPhase:
    """連段課程的一個階段(一招)。

    action_id:            這個 phase 要求的招。
    required_combo:       命中後 combo 計數至少要到多少才算此 phase 完成
                          (多段技會一次跳好幾)。
    reward:               phase 完成的獎勵,越後段越高(引導走完全程)。
    wait_after_hit_frames:命中後強制等待幀數(給演出時間,期間視為未 ready)。
    queue_during_previous:此招必須在前一招動畫中 queue(取消/派生),
                          而不是等前一招收招後再按。
    require_combo_increment:要求 combo 計數「嚴格遞增」才算(防止殘留的
                          舊 combo 數字誤判)。
    require_power_stock_spent:超必用 —— 要求氣條 stock 實際減少(證明
                          超必真的發動,而不是輸入失敗跑出普通版)。
    require_damage:       False = 允許命中先入帳、傷害延遲(琴月陽的
                          延遲爆炸:第二段先加 combo,HP 之後才扣)。
    hit_timeout_frames:   queue 之後多少幀內必須命中,逾時作廢
                          (預設用 COMBO_PROFILE.action_hit_timeout)。
    allow_started_followup_on_hit:命中判定時放寬「必須是 active/剛發動」
                          的條件(某些派生鏈的時序特例)。
    """

    action_id: int
    required_combo: int
    reward: float
    wait_after_hit_frames: int = 0
    queue_during_previous: bool = False
    require_combo_increment: bool = False
    require_power_stock_spent: bool = False
    require_damage: bool = True
    hit_timeout_frames: Optional[int] = None
    allow_started_followup_on_hit: bool = False
    # Sibling finishers that legitimately combo from the same opener. Landing
    # one ends the episode with its (smaller) reward instead of a wrong-action
    # penalty, so scenarios sharing an opener stop emitting contradictory
    # gradients against real combos.
    alternate_actions: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True)
class ComboScenario:
    """一套完整連段 = 依序的 ComboPhase + 全套完成的結算獎金。

    注意:多套 scenario 共用同一個 save state 時,模型「看不出」自己在
    哪套 scenario(觀測相同)—— strict/guided 靠 mask 洩題,physical 下
    會混疊。需要區分時用不同 state(如零氣 slot3)讓觀測可分辨。
    """

    name: str
    phases: tuple[ComboPhase, ...]
    complete_reward: float = 25.0


KYO_CORNER_DOKUGAMI_SCENARIO = ComboScenario(
    name="kyo_corner_dokugami",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(FORWARD_B_ACTION_ID, 2, 3.0),
        ComboPhase(
            POISON_BITE_ACTION_ID,
            4,
            10.0,
            queue_during_previous=True,
            require_combo_increment=True,
        ),
        ComboPhase(TSUMI_YOMI_ACTION_ID, 5, 20.0),
        ComboPhase(
            BATSU_YOMI_ACTION_ID,
            6,
            30.0,
            queue_during_previous=True,
            require_combo_increment=True,
        ),
    ),
)

KYO_FORWARD_B_KOTOTSUKI_SCENARIO = ComboScenario(
    name="kyo_forward_b_kototsuki",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(FORWARD_B_ACTION_ID, 2, 3.0),
        ComboPhase(
            KOTOTSUKI_YOU_ACTION_ID,
            5,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
            # The second hit increments the combo before its delayed explosion updates HP.
            require_damage=False,
            hit_timeout_frames=120,
        ),
    ),
)

KYO_FORWARD_B_OROCHINAGI_SCENARIO = ComboScenario(
    name="kyo_forward_b_orochinagi",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(FORWARD_B_ACTION_ID, 2, 3.0),
        ComboPhase(
            OROCHINAGI_ACTION_ID,
            4,
            30.0,
            queue_during_previous=True,
            require_combo_increment=True,
            require_power_stock_spent=True,
            hit_timeout_frames=140,
        ),
    ),
)

KYO_FORWARD_B_RED_KICK_SCENARIO = ComboScenario(
    name="kyo_forward_b_red_kick",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(FORWARD_B_ACTION_ID, 2, 3.0),
        ComboPhase(
            RED_KICK_ACTION_ID,
            4,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
        ),
    ),
)

KYO_FORWARD_B_ARAGAMI_SCENARIO = ComboScenario(
    name="kyo_forward_b_aragami",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(FORWARD_B_ACTION_ID, 2, 3.0),
        ComboPhase(
            ARAGAMI_ACTION_ID,
            4,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
        ),
    ),
)

KYO_CLOSE_C_SEVENTY_FIVE_SHIKI_KAI_OROCHINAGI_SCENARIO = ComboScenario(
    name="kyo_close_c_seventy_five_shiki_kai_orochinagi",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(
            SEVENTY_FIVE_SHIKI_KAI_ACTION_ID,
            3,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
        ),
        ComboPhase(
            OROCHINAGI_ACTION_ID,
            4,
            30.0,
            queue_during_previous=True,
            require_combo_increment=True,
            require_power_stock_spent=True,
            hit_timeout_frames=160,
            alternate_actions=seventy_five_kai_alternates(OROCHINAGI_ACTION_ID),
        ),
    ),
)

KYO_CLOSE_C_SEVENTY_FIVE_SHIKI_KAI_RED_KICK_SCENARIO = ComboScenario(
    name="kyo_close_c_seventy_five_shiki_kai_red_kick",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(
            SEVENTY_FIVE_SHIKI_KAI_ACTION_ID,
            3,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
        ),
        ComboPhase(
            RED_KICK_ACTION_ID,
            4,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
            hit_timeout_frames=120,
            alternate_actions=seventy_five_kai_alternates(RED_KICK_ACTION_ID),
        ),
    ),
)

KYO_CLOSE_C_SEVENTY_FIVE_SHIKI_KAI_KOTOTSUKI_SCENARIO = ComboScenario(
    name="kyo_close_c_seventy_five_shiki_kai_kototsuki",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(
            SEVENTY_FIVE_SHIKI_KAI_ACTION_ID,
            3,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
        ),
        ComboPhase(
            KOTOTSUKI_YOU_ACTION_ID,
            5,
            20.0,
            queue_during_previous=True,
            require_combo_increment=True,
            require_damage=False,
            hit_timeout_frames=150,
            alternate_actions=seventy_five_kai_alternates(KOTOTSUKI_YOU_ACTION_ID),
        ),
    ),
)

KYO_CLOSE_C_SEVENTY_FIVE_SHIKI_KAI_ARAGAMI_SCENARIO = ComboScenario(
    name="kyo_close_c_seventy_five_shiki_kai_aragami",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(
            SEVENTY_FIVE_SHIKI_KAI_ACTION_ID,
            3,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
        ),
        ComboPhase(
            ARAGAMI_ACTION_ID,
            4,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
            hit_timeout_frames=130,
            alternate_actions=seventy_five_kai_alternates(ARAGAMI_ACTION_ID),
        ),
    ),
)

KYO_CORNER_SEVENTY_FIVE_SHIKI_KAI_ARAGAMI_CHAIN_SCENARIO = ComboScenario(
    name="kyo_corner_seventy_five_shiki_kai_aragami_chain",
    phases=(
        ComboPhase(CLOSE_C_ACTION_ID, 1, 1.0),
        ComboPhase(
            SEVENTY_FIVE_SHIKI_KAI_ACTION_ID,
            3,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
        ),
        ComboPhase(
            ARAGAMI_ACTION_ID,
            4,
            15.0,
            queue_during_previous=True,
            require_combo_increment=True,
            hit_timeout_frames=130,
            allow_started_followup_on_hit=True,
        ),
        ComboPhase(
            YANO_SABI_ACTION_ID,
            5,
            20.0,
            queue_during_previous=True,
            require_combo_increment=True,
            hit_timeout_frames=180,
            allow_started_followup_on_hit=True,
        ),
        ComboPhase(
            MIGIRI_UGACHI_ACTION_ID,
            6,
            30.0,
            queue_during_previous=True,
            require_combo_increment=True,
            hit_timeout_frames=180,
        ),
    ),
)

KYO_CROUCH_B_CROUCH_A_MUSHIKI_SCENARIO = ComboScenario(
    name="kyo_crouch_b_crouch_a_mushiki",
    phases=(
        ComboPhase(CROUCH_B_ACTION_ID, 1, 1.0),
        ComboPhase(
            CROUCH_A_ACTION_ID,
            2,
            8.0,
            queue_during_previous=True,
            require_combo_increment=True,
            hit_timeout_frames=80,
            allow_started_followup_on_hit=True,
        ),
        ComboPhase(
            MUSHIKI_ACTION_ID,
            7,
            40.0,
            queue_during_previous=True,
            require_combo_increment=True,
            require_power_stock_spent=True,
            hit_timeout_frames=180,
        ),
    ),
    complete_reward=35.0,
)

COMBO_SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        KYO_CORNER_DOKUGAMI_SCENARIO,
        KYO_FORWARD_B_KOTOTSUKI_SCENARIO,
        KYO_FORWARD_B_OROCHINAGI_SCENARIO,
        KYO_FORWARD_B_RED_KICK_SCENARIO,
        KYO_FORWARD_B_ARAGAMI_SCENARIO,
        KYO_CLOSE_C_SEVENTY_FIVE_SHIKI_KAI_OROCHINAGI_SCENARIO,
        KYO_CLOSE_C_SEVENTY_FIVE_SHIKI_KAI_RED_KICK_SCENARIO,
        KYO_CLOSE_C_SEVENTY_FIVE_SHIKI_KAI_KOTOTSUKI_SCENARIO,
        KYO_CLOSE_C_SEVENTY_FIVE_SHIKI_KAI_ARAGAMI_SCENARIO,
        KYO_CORNER_SEVENTY_FIVE_SHIKI_KAI_ARAGAMI_CHAIN_SCENARIO,
        KYO_CROUCH_B_CROUCH_A_MUSHIKI_SCENARIO,
    )
}
DEFAULT_COMBO_SCENARIO_NAME = KYO_CORNER_DOKUGAMI_SCENARIO.name


class TrainingProfile(str, Enum):
    COMBO = "combo"
    FIGHT = "fight"


class P2Style(str, Enum):
    """P2 訓練對手的行為風格(C++ 端實作,每個環境固定一種)。

    ONIYAKI:鬼燒反擊型(最早的唯一對手,近身就被升龍)。
    GUARD:  防守型 —— 對它連段會被擋,考驗壓制與破防時機。
    JUMP_IN:跳入型 —— 考驗對空(鬼燒)。
    POKE:   普攻戳型 —— 有週期性破綻,適合練近C 起手。
    設計原則:每種風格都必須留「可學習的破綻」,不可打贏的對手
    給不出任何梯度(v1 的 guard/jump_in 勝率整段 0% 就是教訓)。
    """

    ONIYAKI = "oniyaki"
    GUARD = "guard"
    JUMP_IN = "jump_in"
    POKE = "poke"


class FightCurriculum(str, Enum):
    NONE = "none"
    COMBO_ROUTE = "combo_route"
    DEFENSE = "defense"
    ANTI_AIR = "anti_air"
    APPROACH = "approach"
    HIT_CONFIRM = "hit_confirm"


CURRICULUM_TASK_TO_FIGHT = {
    CurriculumTask.DEFENSE: FightCurriculum.DEFENSE,
    CurriculumTask.ANTI_AIR: FightCurriculum.ANTI_AIR,
    CurriculumTask.APPROACH: FightCurriculum.APPROACH,
    CurriculumTask.HIT_CONFIRM: FightCurriculum.HIT_CONFIRM,
}

CURRICULUM_TASK_TO_P2_STYLE = {
    CurriculumTask.DEFENSE: P2Style.POKE,
    CurriculumTask.ANTI_AIR: P2Style.JUMP_IN,
    CurriculumTask.APPROACH: P2Style.GUARD,
    CurriculumTask.HIT_CONFIRM: P2Style.POKE,
}


P2_STYLE_IDS = {
    P2Style.ONIYAKI: 0,
    P2Style.GUARD: 1,
    P2Style.JUMP_IN: 2,
    P2Style.POKE: 3,
}


class ActionMaskLevel(str, Enum):
    STRICT = "strict"
    GUIDED = "guided"
    PHYSICAL = "physical"


@dataclass(frozen=True)
class ComboProfileConfig:
    """Combo 訓練房的固定參數。action_repeat 恆為 1(逐幀決策)——
    連段輸入窗最窄只有 5 幀,更粗的決策粒度會物理性錯過。"""

    action_repeat: int = 1
    max_episode_steps: int = 480
    damage_reward: float = 0.01
    damage_reward_cap: float = 0.25
    action_hit_timeout: int = 75
    chain_timeout: int = 120
    phase_reset_penalty: float = 1.0
    episode_timeout_penalty: float = 1.0
    ko_without_combo_penalty: float = 3.0
    time_penalty: float = 0.001


COMBO_PROFILE = ComboProfileConfig()


def can_use_super(observation: Optional["Kof98Observation"]) -> bool:
    if observation is None:
        return False

    return observation.p1_advanced_power_stocks >= SUPER_POWER_STOCKS_REQUIRED


def is_p2_airborne(observation: Optional["Kof98Observation"]) -> bool:
    if observation is None or not observation.p2_has_position:
        return False

    return 0 <= observation.p2_y < P2_AIRBORNE_Y_THRESHOLD


class KofEnvClient:
    """fbneo_training.dll 的薄封裝:純轉發,不含訓練邏輯。

    生命週期:load_core → load_game → (load_state | reset) → step 迴圈。
    step(action_id, frames) 的語義:在開頭 setAction 一次,然後連跑
    frames 幀 —— 中間無法插入新決策,這就是 action_repeat 的由來。
    """

    def __init__(self, dll_path: str | Path):
        self.dll_path = Path(dll_path)
        self.dll = ctypes.CDLL(str(self.dll_path))
        self._configure_api()
        self.api_version = int(self.dll.kof_env_api_version())
        self.action_count = int(self.dll.kof_env_public_action_count())
        self.action_set_version = int(self.dll.kof_env_action_set_version())
        self.p1_hold_chunk_frames = int(self.dll.kof_env_p1_hold_chunk_frames())
        self._validate_contract()
        self.handle = self.dll.kof_env_create()
        self._video_callback_ref: Optional[VideoRefreshCallback] = None
        self.last_step_event_batch_epoch = 0
        self.last_step_event_dropped_count = 0
        if not self.handle:
            raise RuntimeError("kof_env_create failed")

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.dll.kof_env_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    def load_core(self, core_path: str | Path) -> None:
        self._check(self.dll.kof_env_load_core(self.handle, str(core_path)))

    def load_game(self, game_path: str | Path, system_dir: str | Path, save_dir: str | Path) -> None:
        self._check(
            self.dll.kof_env_load_game(
                self.handle,
                str(game_path),
                str(system_dir),
                str(save_dir),
            )
        )

    def reset(self) -> None:
        self._check(self.dll.kof_env_reset(self.handle))

    def load_state(self, state_path: str | Path) -> None:
        self._check(self.dll.kof_env_load_state(self.handle, str(state_path)))

    def save_state(self, state_path: str | Path) -> None:
        self._check(self.dll.kof_env_save_state(self.handle, str(state_path)))

    def snapshot_safe(self) -> bool:
        return bool(self.dll.kof_env_snapshot_safe(self.handle))

    def set_video_refresh_callback(
        self,
        callback: Optional[Callable[[int, int, int, int], None]],
    ) -> None:
        if callback is None:
            self._video_callback_ref = None
            self.dll.kof_env_set_video_refresh(self.handle, None, None)
            return

        def _callback(data, width, height, pitch, _user_data):
            callback(data, int(width), int(height), int(pitch))

        self._video_callback_ref = VideoRefreshCallback(_callback)
        self.dll.kof_env_set_video_refresh(
            self.handle,
            ctypes.cast(self._video_callback_ref, ctypes.c_void_p),
            None,
        )

    def set_joypad_for_port(self, port: int, state: JoypadState) -> None:
        self.dll.kof_env_set_joypad_for_port(
            self.handle,
            int(port),
            ctypes.byref(state),
        )

    def set_p2_training_ai(self, enabled: bool) -> None:
        self.dll.kof_env_set_p2_random_ai(self.handle, 1 if enabled else 0)

    def set_p2_style(self, style: P2Style) -> None:
        self._check(
            self.dll.kof_env_set_p2_style(
                self.handle,
                P2_STYLE_IDS[style],
            )
        )

    def set_p2_action_ai(self, enabled: bool) -> None:
        self.dll.kof_env_set_p2_action_ai(self.handle, 1 if enabled else 0)

    def set_p2_action(self, action_id: int) -> None:
        self._check(self.dll.kof_env_set_p2_action(self.handle, int(action_id)))

    def p2_input_ready(self) -> bool:
        return bool(self.dll.kof_env_p2_input_ready(self.handle))

    def p2_ready_for_action(self) -> bool:
        return bool(self.dll.kof_env_p2_ready_for_action(self.handle))

    def can_queue_p2_action(self, action_id: int) -> bool:
        return bool(
            self.dll.kof_env_can_queue_p2_action(
                self.handle,
                int(action_id),
            )
        )

    def step(self, action_id: int, frames: int = 6) -> Kof98Observation:
        observation = Kof98Observation()
        self._check(self.dll.kof_env_step(self.handle, action_id, frames, ctypes.byref(observation)))
        return observation

    def run_frames(self, frames: int) -> None:
        self._check(self.dll.kof_env_run_frames(self.handle, int(frames)))

    def observation(self) -> Kof98Observation:
        observation = Kof98Observation()
        self._check(self.dll.kof_env_get_observation(self.handle, ctypes.byref(observation)))
        return observation

    def p1_ready_for_action(self) -> bool:
        return bool(self.dll.kof_env_p1_ready_for_action(self.handle))

    def input_ready(self) -> bool:
        return bool(self.dll.kof_env_input_ready(self.handle))

    def can_queue_action(self, action_id: int) -> bool:
        return bool(self.dll.kof_env_can_queue_action(self.handle, int(action_id)))

    def action_status(self) -> ActionStatus:
        status = ActionStatus()
        self._check(self.dll.kof_env_get_action_status(self.handle, ctypes.byref(status)))
        return status

    def strategy_state(self) -> StrategyStateV1:
        state = StrategyStateV1()
        state.struct_size = ctypes.sizeof(StrategyStateV1)
        state.version = STRATEGY_STATE_VERSION_1
        self._check(
            self.dll.kof_env_get_strategy_state_v1(
                self.handle,
                ctypes.byref(state),
            )
        )
        return state

    def combat_timing_state(self) -> CombatTimingStateV1:
        state = CombatTimingStateV1()
        state.struct_size = ctypes.sizeof(CombatTimingStateV1)
        state.version = COMBAT_TIMING_STATE_VERSION_1
        self._check(
            self.dll.kof_env_get_combat_timing_state_v1(
                self.handle,
                ctypes.byref(state),
            )
        )
        return state

    def step_events(self) -> list[StepEventV5]:
        result = StepEventsV5()
        result.struct_size = ctypes.sizeof(StepEventsV5)
        result.version = STEP_EVENTS_VERSION_5
        self._check(self.dll.kof_env_get_step_events_v5(self.handle, ctypes.byref(result)))
        self.last_step_event_batch_epoch = int(result.batch_event_epoch)
        self.last_step_event_dropped_count = int(result.dropped_event_count)
        if result.dropped_event_count:
            raise RuntimeError(
                "Step event buffer overflowed: "
                f"{result.dropped_event_count} event(s) were dropped"
            )

        events: list[StepEventV5] = []
        for index in range(result.event_count):
            source = result.events[index]
            events.append(
                StepEventV5(
                    source.frame_offset,
                    source.event_type,
                    source.action_id,
                    source.action_serial,
                    source.combo_before,
                    source.combo_after,
                    source.p1_hp_delta,
                    source.p2_hp_delta,
                    source.target_y_at_event,
                    source.target_airborne_at_event,
                    source.hit_contact,
                    source.block_contact,
                    source.target_airborne_after_event,
                    source.p1_hit_guard_stop_before,
                    source.p1_hit_guard_stop_after,
                    source.p2_hit_guard_stop_before,
                    source.p2_hit_guard_stop_after,
                    source.action_elapsed_frames_at_event,
                    source.expected_blockstun_frames,
                    source.expected_blockstun_source,
                    source.event_epoch,
                    source.guard_reaction_serial,
                    source.source_player,
                    source.target_player,
                    source.absolute_engine_frame,
                )
            )
        return events

    def kyo_move_data(self, action_id: int, variant: int = 0) -> KyoMoveDataV1 | None:
        result = KyoMoveDataV1()
        result.struct_size = ctypes.sizeof(KyoMoveDataV1)
        result.version = MOVE_DATA_VERSION_1
        if not self.dll.kof_env_get_kyo_move_data_v1(
            int(action_id),
            int(variant),
            ctypes.byref(result),
        ):
            return None
        return result

    def last_joypad(self, port: int = 0) -> JoypadState:
        state = JoypadState()
        self._check(
            self.dll.kof_env_get_last_joypad_for_port(
                self.handle,
                port,
                ctypes.byref(state),
            )
        )
        return state

    def copy_system_ram(self) -> bytes:
        size = self.dll.kof_env_system_ram_size(self.handle)
        if size <= 0:
            return b""

        buffer = ctypes.create_string_buffer(size)
        self._check(self.dll.kof_env_copy_system_ram(self.handle, buffer, size))
        return buffer.raw

    def get_hitbox_overlay(
        self,
        source_width: int,
        source_height: int,
        rect_capacity: int = 128,
        axis_capacity: int = 16,
    ) -> tuple[list[HitboxRectResult], list[HitboxAxisResult]]:
        rect_array_type = HitboxRectResult * rect_capacity
        axis_array_type = HitboxAxisResult * axis_capacity
        rects = rect_array_type()
        axes = axis_array_type()
        rect_count = ctypes.c_uint32()
        axis_count = ctypes.c_uint32()

        self._check(
            self.dll.kof_env_get_hitbox_overlay(
                self.handle,
                int(source_width),
                int(source_height),
                rects,
                rect_capacity,
                ctypes.byref(rect_count),
                axes,
                axis_capacity,
                ctypes.byref(axis_count),
            )
        )

        return (
            [rects[index] for index in range(min(rect_count.value, rect_capacity))],
            [axes[index] for index in range(min(axis_count.value, axis_capacity))],
        )

    def contract_metadata(self) -> dict[str, int]:
        return {
            "api_version": self.api_version,
            "action_count": self.action_count,
            "action_set_version": self.action_set_version,
            "p1_hold_chunk_frames": self.p1_hold_chunk_frames,
        }

    def _validate_contract(self) -> None:
        expected = {
            "api_version": EXPECTED_DLL_API_VERSION,
            "action_count": ACTION_COUNT,
            "action_set_version": ACTION_SET_VERSION,
            "p1_hold_chunk_frames": P1_HOLD_CHUNK_FRAMES,
        }
        actual = self.contract_metadata()
        mismatches = [
            f"{name}: DLL={actual[name]}, Python={expected_value}"
            for name, expected_value in expected.items()
            if actual[name] != expected_value
        ]
        if mismatches:
            raise RuntimeError(
                "fbneo_training contract mismatch; rebuild the DLL and use a "
                "matching Python environment (" + ", ".join(mismatches) + ")"
            )

    def _configure_api(self) -> None:
        self.dll.kof_env_api_version.restype = ctypes.c_uint32
        self.dll.kof_env_public_action_count.restype = ctypes.c_uint32
        self.dll.kof_env_action_set_version.restype = ctypes.c_uint32
        self.dll.kof_env_p1_hold_chunk_frames.restype = ctypes.c_uint32

        self.dll.kof_env_create.restype = ctypes.c_void_p
        self.dll.kof_env_destroy.argtypes = [ctypes.c_void_p]

        self.dll.kof_env_load_core.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        self.dll.kof_env_load_core.restype = ctypes.c_int

        self.dll.kof_env_load_game.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
        ]
        self.dll.kof_env_load_game.restype = ctypes.c_int

        self.dll.kof_env_reset.argtypes = [ctypes.c_void_p]
        self.dll.kof_env_reset.restype = ctypes.c_int

        self.dll.kof_env_load_state.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        self.dll.kof_env_load_state.restype = ctypes.c_int

        self.dll.kof_env_save_state.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        self.dll.kof_env_save_state.restype = ctypes.c_int

        self.dll.kof_env_snapshot_safe.argtypes = [ctypes.c_void_p]
        self.dll.kof_env_snapshot_safe.restype = ctypes.c_int

        self.dll.kof_env_set_video_refresh.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]

        self.dll.kof_env_set_joypad_for_port.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.POINTER(JoypadState),
        ]
        self.dll.kof_env_set_joypad_for_port.restype = None

        self.dll.kof_env_set_p2_random_ai.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.dll.kof_env_set_p2_style.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        self.dll.kof_env_set_p2_style.restype = ctypes.c_int
        self.dll.kof_env_set_p2_action_ai.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.dll.kof_env_set_p2_action.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
        ]
        self.dll.kof_env_set_p2_action.restype = ctypes.c_int
        self.dll.kof_env_can_queue_p2_action.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
        ]
        self.dll.kof_env_can_queue_p2_action.restype = ctypes.c_int
        self.dll.kof_env_p2_input_ready.argtypes = [ctypes.c_void_p]
        self.dll.kof_env_p2_input_ready.restype = ctypes.c_int
        self.dll.kof_env_p2_ready_for_action.argtypes = [ctypes.c_void_p]
        self.dll.kof_env_p2_ready_for_action.restype = ctypes.c_int

        self.dll.kof_env_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(Kof98Observation),
        ]
        self.dll.kof_env_step.restype = ctypes.c_int

        self.dll.kof_env_run_frames.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        self.dll.kof_env_run_frames.restype = ctypes.c_int

        self.dll.kof_env_get_observation.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(Kof98Observation),
        ]
        self.dll.kof_env_get_observation.restype = ctypes.c_int

        self.dll.kof_env_input_ready.argtypes = [ctypes.c_void_p]
        self.dll.kof_env_input_ready.restype = ctypes.c_int

        self.dll.kof_env_can_queue_action.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        self.dll.kof_env_can_queue_action.restype = ctypes.c_int

        self.dll.kof_env_get_action_status.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ActionStatus),
        ]
        self.dll.kof_env_get_action_status.restype = ctypes.c_int

        self.dll.kof_env_get_strategy_state_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(StrategyStateV1),
        ]
        self.dll.kof_env_get_strategy_state_v1.restype = ctypes.c_int

        self.dll.kof_env_get_combat_timing_state_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(CombatTimingStateV1),
        ]
        self.dll.kof_env_get_combat_timing_state_v1.restype = ctypes.c_int

        self.dll.kof_env_get_step_events_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(StepEventsV1),
        ]
        self.dll.kof_env_get_step_events_v1.restype = ctypes.c_int

        self.dll.kof_env_get_step_events_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(StepEventsV2),
        ]
        self.dll.kof_env_get_step_events_v2.restype = ctypes.c_int

        self.dll.kof_env_get_step_events_v3.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(StepEventsV3),
        ]
        self.dll.kof_env_get_step_events_v3.restype = ctypes.c_int

        self.dll.kof_env_get_step_events_v4.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(StepEventsV4),
        ]
        self.dll.kof_env_get_step_events_v4.restype = ctypes.c_int

        self.dll.kof_env_get_step_events_v5.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(StepEventsV5),
        ]
        self.dll.kof_env_get_step_events_v5.restype = ctypes.c_int

        self.dll.kof_env_get_kyo_move_data_v1.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(KyoMoveDataV1),
        ]
        self.dll.kof_env_get_kyo_move_data_v1.restype = ctypes.c_int

        self.dll.kof_env_p1_ready_for_action.argtypes = [ctypes.c_void_p]
        self.dll.kof_env_p1_ready_for_action.restype = ctypes.c_int

        self.dll.kof_env_get_last_joypad_for_port.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.POINTER(JoypadState),
        ]
        self.dll.kof_env_get_last_joypad_for_port.restype = ctypes.c_int

        self.dll.kof_env_system_ram_size.argtypes = [ctypes.c_void_p]
        self.dll.kof_env_system_ram_size.restype = ctypes.c_uint32

        self.dll.kof_env_copy_system_ram.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self.dll.kof_env_copy_system_ram.restype = ctypes.c_int

        self.dll.kof_env_get_hitbox_overlay.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(HitboxRectResult),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(HitboxAxisResult),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.dll.kof_env_get_hitbox_overlay.restype = ctypes.c_int

        self.dll.kof_env_last_error.argtypes = [ctypes.c_void_p]
        self.dll.kof_env_last_error.restype = ctypes.c_char_p

    def _check(self, result: int) -> None:
        if result:
            return

        message = self.dll.kof_env_last_error(self.handle)
        raise RuntimeError(message.decode("utf-8", errors="replace") if message else "fbneo_training failed")


def observation_to_vector(observation: Kof98Observation) -> np.ndarray:
    """RAM 原始值 → 23 維正規化向量(除以各欄位的遊戲上限,壓到 ~[0,1])。

    負值(RAM 讀取無效)一律夾成 0,並靠 p*_has_position 旗標讓模型
    知道座標是否可信。改動維度 = 改觀測形狀 = 舊模型全部作廢,慎重。
    """
    return base_observation_vector(observation)


def rects_overlap(a: HitboxRectResult, b: HitboxRectResult, margin: int = 0) -> bool:
    return (
        a.left - margin < b.left + b.width
        and a.left + a.width + margin > b.left
        and a.top - margin < b.top + b.height
        and a.top + a.height + margin > b.top
    )


def has_attack_hurtbox_overlap(
    rects: list[HitboxRectResult],
    attacker_owner: int,
    defender_owner: int,
    margin: int = 0,
) -> bool:
    attack_types = {HITBOX_ATTACK, HITBOX_PROJECTILE_ATTACK}
    hurtbox_types = {HITBOX_VULNERABILITY, HITBOX_PROJECTILE_VULNERABILITY}
    attacks = [
        rect for rect in rects
        if rect.owner == attacker_owner and rect.type in attack_types
    ]
    hurtboxes = [
        rect for rect in rects
        if rect.owner == defender_owner and rect.type in hurtbox_types
    ]

    return any(rects_overlap(attack, hurtbox, margin) for attack in attacks for hurtbox in hurtboxes)


class Kof98Env(gym.Env if gym else object):
    """Gymnasium 環境本體。觀測依 ABI 為 V1 26 維或 V2 140 維，
    動作固定為 Discrete(29)。

    - COMBO profile:_step_combo,逐 phase 課程給獎。
    - FIGHT profile:_step_fight,實戰 reward 由 fight_reward_version 選擇。
    - fight_curriculum 可掛 combo route、hit confirm、防禦、對空或接近
      的局部 action-mask teacher；完整 Fight reward 仍只看傷害與勝負。
    - Combo env 可在每次 reset 輪替 scenario，避免環境數少於課表時
      永遠遺漏尾端連段。
    - Fight env 也可在每次 reset 輪替 curriculum/P2 style，以較少的
      FBNeo process 覆蓋完整戰術與對手分佈。
    - MaskablePPO 透過 action_masks() 取得合法動作;沒有 mask 支援的
      推論端也可以直接用 policy(但可能選到物理不可能的招被 DLL 忽略)。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dll_path: str | Path,
        core_path: str | Path,
        game_path: str | Path,
        system_dir: str | Path,
        save_dir: str | Path,
        state_path: Optional[str | Path] = None,
        action_repeat: int = 6,
        hitbox_reward: bool = True,
        p2_training_ai: bool = False,
        p2_style: P2Style | str = P2Style.ONIYAKI,
        fight_guided: bool = False,
        fight_curriculum: FightCurriculum | str = FightCurriculum.NONE,
        fight_rotation: Optional[
            list[tuple[FightCurriculum | str, P2Style | str]]
        ] = None,
        fight_rotation_offset: int = 0,
        fight_rotation_stride: int = 1,
        training_profile: TrainingProfile | str = TrainingProfile.COMBO,
        combo_state_path: Optional[str | Path] = None,
        fight_state_path: Optional[str | Path] = None,
        combo_scenario: ComboScenario | str = DEFAULT_COMBO_SCENARIO_NAME,
        combo_scenario_rotation: Optional[
            list[tuple[ComboScenario | str, str | Path]]
        ] = None,
        combo_rotation_offset: int = 0,
        combo_rotation_stride: int = 1,
        action_mask_level: ActionMaskLevel | str = ActionMaskLevel.STRICT,
        observation_version: ObservationVersion | str = ObservationVersion.V1,
        observation_event_features: bool = True,
        fight_reward_version: FightRewardVersion | str | None = None,
        level_recipe: LevelRecipe | None = None,
        level_recipe_rotation: Optional[list[LevelRecipe]] = None,
        level_recipe_rotation_offset: int = 0,
        level_recipe_rotation_stride: int = 1,
    ):
        if gym is None or spaces is None:
            raise RuntimeError("Install gymnasium before using Kof98Env")

        super().__init__()
        self.client = KofEnvClient(dll_path)
        self.client.load_core(core_path)
        self.client.load_game(game_path, system_dir, save_dir)
        self.state_path = Path(state_path) if state_path else None
        self.combo_state_path = Path(combo_state_path) if combo_state_path else self.state_path
        self.fight_state_path = Path(fight_state_path) if fight_state_path else self.state_path
        self.training_profile = TrainingProfile(training_profile)
        self.action_mask_level = ActionMaskLevel(action_mask_level)
        self.observation_version = ObservationVersion(observation_version)
        self.observation_event_features = bool(observation_event_features)
        self.fight_reward_version = (
            FightRewardVersion(fight_reward_version)
            if fight_reward_version is not None
            else (
                FightRewardVersion.SYMMETRIC_V2
                if self.observation_version is not ObservationVersion.V1
                else FightRewardVersion.LEGACY_COMBO4
            )
        )
        self.level_recipe = level_recipe
        self.level_recipe_rotation = list(level_recipe_rotation or [])
        if self.level_recipe is not None and self.level_recipe_rotation:
            raise ValueError(
                "Use either level_recipe or level_recipe_rotation, not both"
            )
        if self.level_recipe is not None:
            if self.training_profile is not TrainingProfile.FIGHT:
                raise ValueError("Level recipes are only supported by the Fight profile")
            self.level_recipe.validate(self.client.action_count)
        if self.level_recipe_rotation:
            if self.training_profile is not TrainingProfile.FIGHT:
                raise ValueError("Level recipe rotation requires the Fight profile")
            for rotating_recipe in self.level_recipe_rotation:
                rotating_recipe.validate(self.client.action_count)
        self.level_recipe_rotation_offset = max(
            0,
            int(level_recipe_rotation_offset),
        )
        self.level_recipe_rotation_stride = max(
            1,
            int(level_recipe_rotation_stride),
        )
        self.level_recipe_rotation_episode = 0
        self.level_recipe_p2_start_delay_remaining = 0
        self.level_recipe_p2_started = False
        self.level_oracle_action_index = 0
        self.level_oracle_wait_remaining_frames = 0
        self.level_oracle_trajectory_valid = True
        self.level_route_next_action_index = 0
        self.level_route_started: list[tuple[int, int]] = []
        self.level_route_hit_indices: set[int] = set()
        self.level_route_invalid = False
        self.level_guard_contact_seen = False
        self.combat_phase_machine = CombatPhaseMachine()
        self.generic_combat_state_machine = GenericCombatStateMachine()
        self.tactical_reward_machine = TacticalRewardMachine()
        self.last_step_events: list[StepEventV5] = []
        self.level_episode_frames = 0
        if isinstance(combo_scenario, ComboScenario):
            self.combo_scenario = combo_scenario
        else:
            try:
                self.combo_scenario = COMBO_SCENARIOS[combo_scenario]
            except KeyError as error:
                raise ValueError(f"Unknown combo scenario: {combo_scenario}") from error
        self.combo_scenario_rotation: list[tuple[ComboScenario, Path]] = []
        for rotating_scenario, rotating_state_path in combo_scenario_rotation or []:
            if isinstance(rotating_scenario, ComboScenario):
                resolved_scenario = rotating_scenario
            else:
                try:
                    resolved_scenario = COMBO_SCENARIOS[rotating_scenario]
                except KeyError as error:
                    raise ValueError(
                        f"Unknown rotating combo scenario: {rotating_scenario}"
                    ) from error
            self.combo_scenario_rotation.append(
                (resolved_scenario, Path(rotating_state_path)),
            )
        self.combo_rotation_offset = max(0, int(combo_rotation_offset))
        self.combo_rotation_stride = max(1, int(combo_rotation_stride))
        self.combo_rotation_episode = 0
        self.action_repeat = (
            COMBO_PROFILE.action_repeat
            if self.training_profile is TrainingProfile.COMBO
            else action_repeat
        )
        self.fight_frame_scale = float(self.action_repeat) / FIGHT_SHAPING_BASELINE_FRAMES
        self.hitbox_reward = (
            hitbox_reward
            and self.training_profile is TrainingProfile.FIGHT
            and self.fight_reward_version is FightRewardVersion.LEGACY_COMBO4
        )
        self.p2_training_ai = p2_training_ai
        self.p2_style = P2Style(p2_style)
        self.fight_curriculum = FightCurriculum(fight_curriculum)
        if fight_guided and self.fight_curriculum is FightCurriculum.NONE:
            self.fight_curriculum = FightCurriculum.COMBO_ROUTE
        if self.training_profile is not TrainingProfile.FIGHT:
            self.fight_curriculum = FightCurriculum.NONE
        if self.level_recipe is not None:
            self.fight_curriculum = CURRICULUM_TASK_TO_FIGHT[
                self.level_recipe.task
            ]
            self.p2_style = CURRICULUM_TASK_TO_P2_STYLE[
                self.level_recipe.task
            ]
        self.fight_rotation: list[tuple[FightCurriculum, P2Style]] = [
            (FightCurriculum(curriculum), P2Style(style))
            for curriculum, style in (fight_rotation or [])
        ]
        if self.training_profile is not TrainingProfile.FIGHT:
            self.fight_rotation = []
        self.fight_rotation_offset = max(0, int(fight_rotation_offset))
        self.fight_rotation_stride = max(1, int(fight_rotation_stride))
        self.fight_rotation_episode = 0
        self.fight_guided = self.fight_curriculum in (
            FightCurriculum.COMBO_ROUTE,
            FightCurriculum.HIT_CONFIRM,
        )
        if self.level_recipe is not None or self.level_recipe_rotation:
            self.fight_guided = False
        self.previous_observation: Optional[Kof98Observation] = None
        self.previous_strategy_state: Optional[StrategyStateV1] = None
        self.pending_attack_risk: Optional[dict[str, float]] = None
        self.fight_pending_followups: list[dict] = []
        self.fight_prev_p1_attack_overlap = False
        self.fight_prev_p2_attack_overlap = False
        self.fight_combo_4plus_rewarded = False
        self.fight_teacher_phase = 0
        self.fight_teacher_chain_seen = False
        self.fight_teacher_route_completed = False
        self.defense_counter_window_frames = 0
        self.defense_blockstun_pending = False
        self.defense_blockstun_pending_deadline_frame = -1
        self.defense_blockstun_pending_epoch = -1
        self.defense_blockstun_pending_reaction_serial = -1
        self.defense_counter_window_start_frame = -1
        self.defense_counter_window_deadline_frame = -1
        self.defense_counter_action_id = -1
        self.defense_counter_action_serial = -1
        self.defense_counter_hit_deadline_frame = -1
        self.fight_hit_confirm_window_frames = 0
        self.fight_pending_opener_action: Optional[int] = None
        self.fight_pending_opener_serial: Optional[int] = None
        self.fight_pending_opener_age = 0
        self.fight_approach_pending = False
        self.fight_approach_p1_progress = 0.0
        self.fight_approach_forward_frames = 0
        self.level_guard_started = False
        self.level_guard_hold_frames = 0
        self.level_guard_damage = 0.0
        self.level_guard_clean_hit = False
        self.episode_steps = 0
        self.episode_max_combo = 0
        self.combo_phase = 0
        self.pending_chain_action: Optional[int] = None
        self.pending_action_age = 0
        self.pending_action_power_stocks: Optional[int] = None
        self.pending_followups: dict[int, int] = {}
        self.pending_alternate_action: Optional[int] = None
        self.pending_alternate_reward = 0.0
        self.pending_alternate_age = 0
        self.pending_alternate_min_combo = 0
        self.pending_alternate_power_stocks: Optional[int] = None
        self.pending_alternate_hit_timeout = COMBO_PROFILE.action_hit_timeout
        self.frames_since_chain_hit = 0
        self.phase_wait_remaining = 0

        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(observation_size(self.observation_version),),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.level_recipe_rotation:
            recipe_index = (
                self.level_recipe_rotation_offset
                + self.level_recipe_rotation_episode
                * self.level_recipe_rotation_stride
            ) % len(self.level_recipe_rotation)
            self.level_recipe = self.level_recipe_rotation[recipe_index]
            self.level_recipe_rotation_episode += 1
            self.fight_curriculum = CURRICULUM_TASK_TO_FIGHT[
                self.level_recipe.task
            ]
            self.p2_style = CURRICULUM_TASK_TO_P2_STYLE[
                self.level_recipe.task
            ]
            self.fight_guided = False
        elif (
            self.training_profile is TrainingProfile.FIGHT
            and self.fight_rotation
        ):
            rotation_index = (
                self.fight_rotation_offset
                + self.fight_rotation_episode * self.fight_rotation_stride
            ) % len(self.fight_rotation)
            self.fight_curriculum, self.p2_style = self.fight_rotation[
                rotation_index
            ]
            self.fight_rotation_episode += 1
            self.fight_guided = self.fight_curriculum in (
                FightCurriculum.COMBO_ROUTE,
                FightCurriculum.HIT_CONFIRM,
            )

        if (
            self.training_profile is TrainingProfile.COMBO
            and self.combo_scenario_rotation
        ):
            rotation_index = (
                self.combo_rotation_offset
                + self.combo_rotation_episode * self.combo_rotation_stride
            ) % len(self.combo_scenario_rotation)
            self.combo_scenario, state_path = self.combo_scenario_rotation[rotation_index]
            self.combo_rotation_episode += 1
        else:
            state_path = (
                self.combo_state_path
                if self.training_profile is TrainingProfile.COMBO
                else self.fight_state_path
            )
        if self.level_recipe is not None:
            observation = prepare_level(self.client, self.level_recipe)
            self.level_recipe_p2_start_delay_remaining = (
                self.level_recipe.p2_start_delay_frames
            )
            self.level_recipe_p2_started = (
                self.level_recipe.p2_action_id is None
                or self.level_recipe.p2_start_delay_frames == 0
            )
        else:
            if state_path:
                self.client.load_state(state_path)
            else:
                self.client.reset()

            p2_training_ai = (
                self.p2_training_ai
                if self.training_profile is TrainingProfile.FIGHT
                else False
            )
            self.client.set_p2_style(self.p2_style)
            self.client.set_p2_training_ai(p2_training_ai)
            observation = self.client.observation()
            self.level_recipe_p2_start_delay_remaining = 0
            self.level_recipe_p2_started = False
        self.previous_observation = None
        self.previous_strategy_state = None
        self.combat_phase_machine.reset()
        self.generic_combat_state_machine.reset()
        self.tactical_reward_machine.reset()
        self.level_episode_frames = 0
        self.pending_attack_risk = None
        self.fight_pending_followups = []
        self.fight_prev_p1_attack_overlap = False
        self.fight_prev_p2_attack_overlap = False
        self.fight_combo_4plus_rewarded = False
        self.fight_teacher_phase = 0
        self.fight_teacher_chain_seen = False
        self.fight_teacher_route_completed = False
        self.defense_counter_window_frames = 0
        self.defense_blockstun_pending = False
        self.defense_blockstun_pending_deadline_frame = -1
        self.defense_blockstun_pending_epoch = -1
        self.defense_blockstun_pending_reaction_serial = -1
        self.defense_counter_window_start_frame = -1
        self.defense_counter_window_deadline_frame = -1
        self.defense_counter_action_id = -1
        self.defense_counter_action_serial = -1
        self.defense_counter_hit_deadline_frame = -1
        self.fight_hit_confirm_window_frames = 0
        self.fight_pending_opener_action = None
        self.fight_pending_opener_serial = None
        self.fight_pending_opener_age = 0
        self.fight_approach_pending = False
        self.fight_approach_p1_progress = 0.0
        self.fight_approach_forward_frames = 0
        self.level_guard_started = False
        self.level_guard_hold_frames = 0
        self.level_guard_damage = 0.0
        self.level_guard_clean_hit = False
        self.level_guard_contact_seen = False
        self.level_oracle_action_index = 0
        self.level_oracle_wait_remaining_frames = (
            self.level_recipe.oracle_actions[0].wait_before_frames
            if self.level_recipe is not None
            and self.level_recipe.oracle_actions
            else 0
        )
        self.level_oracle_trajectory_valid = True
        self.level_route_next_action_index = 0
        self.level_route_started = []
        self.level_route_hit_indices = set()
        self.level_route_invalid = False
        self.episode_steps = 0
        self.episode_max_combo = 0
        self.combo_phase = 0
        self.pending_chain_action = None
        self.pending_action_age = 0
        self.pending_action_power_stocks = None
        self.pending_followups = {}
        self.pending_alternate_action = None
        self.pending_alternate_reward = 0.0
        self.pending_alternate_age = 0
        self.pending_alternate_min_combo = 0
        self.pending_alternate_power_stocks = None
        self.pending_alternate_hit_timeout = COMBO_PROFILE.action_hit_timeout
        self.frames_since_chain_hit = 0
        self.phase_wait_remaining = 0
        encoded_observation = self._make_observation(
            observation,
            None,
            elapsed_frames=0,
        )
        self.previous_observation = observation
        return encoded_observation, {
            "raw": observation,
            "training_profile": self.training_profile.value,
            "combo_scenario": self.combo_scenario.name,
            "action_mask_level": self.action_mask_level.value,
            "observation_version": self.observation_version.value,
            "fight_reward_version": self.fight_reward_version.value,
            "fight_curriculum": self.fight_curriculum.value,
            "fight_guided": float(self.fight_guided),
            "p2_style": self.p2_style.value,
            "level_recipe": self.level_recipe.name if self.level_recipe else "",
            "curriculum_level": float(self.level_recipe.level if self.level_recipe else -1),
            "curriculum_task": self.level_recipe.task.value if self.level_recipe else "",
            "level_p2_start_delay_remaining": float(
                self.level_recipe_p2_start_delay_remaining
            ),
            "combat_phase": float(self.combat_phase_machine.phase),
        }

    def step(self, action):
        action_id = int(action)
        if self.training_profile is TrainingProfile.COMBO:
            return self._step_combo(action_id)

        self._start_scheduled_level_p2_action()
        result = self._step_fight(action_id)
        if not self.level_recipe_p2_started:
            self.level_recipe_p2_start_delay_remaining = max(
                0,
                self.level_recipe_p2_start_delay_remaining - self.action_repeat,
            )
        return result

    def _start_scheduled_level_p2_action(self) -> None:
        if (
            self.level_recipe is None
            or self.level_recipe_p2_started
            or self.level_recipe_p2_start_delay_remaining > 0
            or self.level_recipe.p2_action_id is None
        ):
            return

        self.client.set_p2_action(self.level_recipe.p2_action_id)
        self.level_recipe_p2_started = True

    def _level_oracle_action(self) -> int:
        if (
            self.level_recipe is None
            or not self.level_oracle_trajectory_valid
            or self.level_oracle_action_index
            >= len(self.level_recipe.oracle_actions)
        ):
            return -1
        if self.level_oracle_wait_remaining_frames > 0:
            return IDLE_ACTION_ID
        desired_action = self.level_recipe.oracle_actions[
            self.level_oracle_action_index
        ].action_id
        if not self._physical_action_mask()[desired_action]:
            return IDLE_ACTION_ID
        return desired_action

    def _advance_level_oracle(
        self,
        expected_action: int,
        selected_action: int,
        action_accepted: bool,
    ) -> None:
        if self.level_recipe is None:
            return
        if not self.level_oracle_trajectory_valid or expected_action < 0:
            return
        if selected_action != expected_action or not action_accepted:
            self.level_oracle_trajectory_valid = False
            return
        if self.level_oracle_wait_remaining_frames > 0:
            self.level_oracle_wait_remaining_frames = max(
                0,
                self.level_oracle_wait_remaining_frames - self.action_repeat,
            )
            return

        desired_action = self.level_recipe.oracle_actions[
            self.level_oracle_action_index
        ].action_id
        if expected_action == IDLE_ACTION_ID and desired_action != IDLE_ACTION_ID:
            return

        completed = self.level_recipe.oracle_actions[
            self.level_oracle_action_index
        ]
        self.level_oracle_action_index += 1
        next_wait = completed.settle_after_frames
        if self.level_oracle_action_index < len(self.level_recipe.oracle_actions):
            next_wait += self.level_recipe.oracle_actions[
                self.level_oracle_action_index
            ].wait_before_frames
        self.level_oracle_wait_remaining_frames = next_wait

    def _level_recipe_lesson(self) -> str:
        if self.level_recipe is None:
            return ""
        return str(self.level_recipe.metadata.get("lesson", self.level_recipe.task.value))

    def _clear_defense_counter_window(self) -> None:
        self.defense_counter_window_frames = 0
        self.defense_counter_window_start_frame = -1
        self.defense_counter_window_deadline_frame = -1
        self.defense_counter_action_id = -1
        self.defense_counter_action_serial = -1
        self.defense_counter_hit_deadline_frame = -1

    def _post_block_response_hit_timeout(self, action_id: int) -> int:
        known_timeout = SEVENTY_FIVE_KAI_FINISHER_HIT_TIMEOUTS.get(action_id)
        if known_timeout is not None:
            return known_timeout
        client = getattr(self, "client", None)
        move_data = (
            client.kyo_move_data(action_id)
            if client is not None
            else None
        )
        if move_data is not None and move_data.startup_frames >= 0:
            active_frames = max(1, int(move_data.active_frames))
            return max(
                24,
                int(move_data.startup_frames) + active_frames + 12,
            )
        return COMBO_PROFILE.action_hit_timeout

    def _begin_defense_blockstun_wait(
        self,
        absolute_frame: int,
        event_epoch: int,
        guard_reaction_serial: int,
    ) -> None:
        self._clear_defense_counter_window()
        self.defense_blockstun_pending = True
        self.defense_blockstun_pending_deadline_frame = (
            int(absolute_frame) + DEFENSE_BLOCKSTUN_PENDING_TIMEOUT_FRAMES
        )
        self.defense_blockstun_pending_epoch = int(event_epoch)
        self.defense_blockstun_pending_reaction_serial = int(
            guard_reaction_serial
        )

    def _clear_defense_blockstun_wait(self) -> None:
        self.defense_blockstun_pending = False
        self.defense_blockstun_pending_deadline_frame = -1
        self.defense_blockstun_pending_epoch = -1
        self.defense_blockstun_pending_reaction_serial = -1

    def _open_defense_counter_window(self, blockstun_end_frame: int) -> None:
        # BLOCKSTUN_ENDED 所在幀剛完成；下一個 emulator frame 才是第一個
        # 可確反幀。有效範圍採半開區間 [start, start + 36)。
        self.defense_counter_window_start_frame = int(blockstun_end_frame) + 1
        self.defense_counter_window_deadline_frame = (
            self.defense_counter_window_start_frame
            + DEFENSE_COUNTER_WINDOW_FRAMES
        )
        self.defense_counter_window_frames = DEFENSE_COUNTER_WINDOW_FRAMES
        self.defense_counter_action_id = -1
        self.defense_counter_action_serial = -1
        self.defense_counter_hit_deadline_frame = -1
        self._clear_defense_blockstun_wait()

    def _defense_counter_window_contains(self, absolute_frame: int) -> bool:
        return (
            self.defense_counter_window_start_frame >= 0
            and self.defense_counter_window_start_frame <= int(absolute_frame)
            < self.defense_counter_window_deadline_frame
        )

    def _process_defense_counter_events(
        self,
        step_events: list[StepEventV5],
        step_base_frame: int,
    ) -> tuple[list[StepEventV5], bool, bool, bool]:
        """更新 block→counter 狀態，回傳命中、開窗、曾開窗、逾時。

        所有 deadline 都以 emulator absolute frame 計算；PPO 是否有自由
        決策、攻擊腳本是否仍在執行，都不會暫停這 36 幀。
        """
        window_was_open = self.defense_counter_window_deadline_frame >= 0
        window_opened_now = False
        counter_hit_events: list[StepEventV5] = []

        for event in sorted(step_events, key=lambda value: int(value.frame_offset)):
            absolute_frame = int(event.absolute_engine_frame)

            if (
                self.defense_blockstun_pending
                and self.defense_blockstun_pending_epoch >= 0
                and int(event.event_epoch)
                != self.defense_blockstun_pending_epoch
            ):
                # reset/load/換局後，舊 epoch 的 manual block 不得等待新
                # 回合的 END。Counter window 也同時失效。
                self._clear_defense_blockstun_wait()
                self._clear_defense_counter_window()

            if (
                event.event_type == STEP_EVENT_MANUAL_BLOCK_SUCCESS
                and bool(event.block_contact)
                and int(event.source_player) == 2
                and int(event.target_player) == 1
                and int(event.guard_reaction_serial) > 0
            ):
                self._begin_defense_blockstun_wait(
                    absolute_frame,
                    int(event.event_epoch),
                    int(event.guard_reaction_serial),
                )
                continue

            if (
                event.event_type == STEP_EVENT_CLEAN_HIT
                and int(event.source_player) == 2
                and int(event.target_player) == 1
                and event.p1_hp_delta > 0
            ):
                self._clear_defense_blockstun_wait()
                self._clear_defense_counter_window()
                continue

            if (
                event.event_type == STEP_EVENT_BLOCKSTUN_ENDED
                and int(event.source_player) == 2
                and int(event.target_player) == 1
                and self.defense_blockstun_pending
                and int(event.event_epoch)
                == self.defense_blockstun_pending_epoch
                and int(event.guard_reaction_serial)
                == self.defense_blockstun_pending_reaction_serial
            ):
                # C++ 只有在已確認 BLOCK_CONTACT、D2:D3 倒數 N→N-1，
                # 最後 0→-1 且 E3 0x20→0x00 時才發 END。START 只是
                # 晚一幀確認倒數的 telemetry，不能拿來開確反窗口。
                self._open_defense_counter_window(absolute_frame)
                window_opened_now = True
                continue

            if (
                event.event_type == STEP_EVENT_ACTION_STARTED
                and int(event.source_player) == 1
                and event.action_id >= 6
                and self.defense_counter_action_serial < 0
                and self._defense_counter_window_contains(absolute_frame)
            ):
                self.defense_counter_action_id = int(event.action_id)
                self.defense_counter_action_serial = int(event.action_serial)
                self.defense_counter_hit_deadline_frame = (
                    absolute_frame
                    + self._post_block_response_hit_timeout(
                        self.defense_counter_action_id
                    )
                )
                continue

            if (
                event.event_type == STEP_EVENT_COMBO_HIT
                and int(event.source_player) == 1
                and int(event.target_player) == 2
                and event.combo_after > event.combo_before
                and int(event.action_id) == self.defense_counter_action_id
                and int(event.action_serial) == self.defense_counter_action_serial
                and self.defense_counter_hit_deadline_frame >= 0
                and absolute_frame < self.defense_counter_hit_deadline_frame
            ):
                counter_hit_events.append(event)

        # Production uses the DLL's absolute engine clock. Lightweight unit
        # fixtures created with object.__new__ do not own a client; there the
        # explicit frame argument remains the deterministic clock source.
        if hasattr(self, "client"):
            step_end_frame = int(
                self.client.combat_timing_state().engine_frame
            )
        else:
            step_end_frame = int(step_base_frame)
        if (
            self.defense_blockstun_pending
            and self.defense_blockstun_pending_deadline_frame >= 0
            and step_end_frame >= self.defense_blockstun_pending_deadline_frame
        ):
            self._clear_defense_blockstun_wait()

        window_existed_this_step = (
            window_was_open
            or window_opened_now
            or self.defense_counter_window_deadline_frame >= 0
        )
        if self.defense_counter_action_serial >= 0:
            window_expired_now = (
                self.defense_counter_hit_deadline_frame >= 0
                and step_end_frame >= self.defense_counter_hit_deadline_frame
            )
        else:
            window_expired_now = (
                self.defense_counter_window_deadline_frame >= 0
                and step_end_frame >= self.defense_counter_window_deadline_frame
            )
        active_deadline = (
            self.defense_counter_hit_deadline_frame
            if self.defense_counter_action_serial >= 0
            else self.defense_counter_window_deadline_frame
        )
        if active_deadline >= 0:
            self.defense_counter_window_frames = max(
                0,
                active_deadline - step_end_frame,
            )
        else:
            self.defense_counter_window_frames = 0

        return (
            counter_hit_events,
            window_opened_now,
            window_existed_this_step,
            window_expired_now,
        )

    def _update_level_hit_confirm_route(
        self,
        step_events: list[StepEventV5],
        current_combo: int,
    ) -> tuple[bool, bool, bool, bool]:
        if (
            self.level_recipe is None
            or self.level_recipe.task is not CurriculumTask.HIT_CONFIRM
        ):
            return False, False, False, False

        route = tuple(action.action_id for action in self.level_recipe.oracle_actions)
        if not route:
            return False, False, False, True

        for event in step_events:
            if event.event_type == STEP_EVENT_ACTION_STARTED:
                if self.level_route_invalid:
                    continue
                if (
                    self.level_route_next_action_index >= len(route)
                    or int(event.action_id)
                    != route[self.level_route_next_action_index]
                ):
                    self.level_route_invalid = True
                    continue
                self.level_route_started.append(
                    (int(event.action_id), int(event.action_serial))
                )
                self.level_route_next_action_index += 1
            elif event.event_type == STEP_EVENT_COMBO_HIT:
                event_key = (int(event.action_id), int(event.action_serial))
                for index, started in enumerate(self.level_route_started):
                    if started == event_key:
                        self.level_route_hit_indices.add(index)
                        break

        required_combo = int(self.level_recipe.metadata.get("required_combo", 2))
        route_started = self.level_route_next_action_index > 0
        route_committed = self.level_route_next_action_index > 1
        all_actions_started = self.level_route_next_action_index == len(route)
        all_actions_hit = all_actions_started and all(
            index in self.level_route_hit_indices for index in range(len(route))
        )
        route_success = (
            not self.level_route_invalid
            and all_actions_hit
            and current_combo >= required_combo
        )
        return route_started, route_committed, route_success, self.level_route_invalid

    def _combo_phase_at(self, phase_index: int) -> Optional[ComboPhase]:
        if 0 <= phase_index < len(self.combo_scenario.phases):
            return self.combo_scenario.phases[phase_index]
        return None

    def _current_combo_phase(self) -> Optional[ComboPhase]:
        return self._combo_phase_at(self.combo_phase)

    def _queueable_followup_phase(self) -> Optional[ComboPhase]:
        # Scan every remaining phase: at repeat=1 a follow-up window can open
        # before the previous phase's hit registers (e.g. 75 Kai fires on
        # frame 5 while Close C's hit lands on frame 7), so the queueable
        # phase may sit more than one step past combo_phase.
        for phase_index in range(self.combo_phase, len(self.combo_scenario.phases)):
            phase = self._combo_phase_at(phase_index)
            if (
                phase is not None
                and phase.queue_during_previous
                and self.client.can_queue_action(phase.action_id)
            ):
                return phase
        return None

    def _step_combo(self, action_id: int):
        """Combo 訓練房的一步。流程:

        1. 捕捉 step 前狀態(input_ready、可 queue 的 phase)——
           判定必須用「按下當下」的狀態,step 後狀態已經變了。
        2. 執行一幀,拿 action_status。
        3. 追蹤三種 pending:
           - pending_chain(input ready 時按的當前 phase 招)
           - pending_followups(busy 中 queue 的派生技,dict 多槽 ——
             盲區內前後兩招會同時 pending,單槽會互相覆蓋)
           - pending_alternate(同起手的兄弟收尾,命中給小獎結束回合,
             避免「懲罰真實連段」的矛盾梯度)
        4. _combo_reward 判定 phase 推進/斷鏈/完成,組合 reward。
        """
        previous = self.previous_observation
        self.phase_wait_remaining = max(
            0,
            self.phase_wait_remaining - COMBO_PROFILE.action_repeat,
        )
        input_ready_before_step = self._combo_input_ready()
        queueable_followup_phase = self._queueable_followup_phase()
        expected_phase_before_step = queueable_followup_phase or self._current_combo_phase()
        observation = self.client.step(action_id, COMBO_PROFILE.action_repeat)
        action_status = self.client.action_status()
        step_events = self.client.step_events()
        self.last_step_events = list(step_events)
        action_accepted = bool(action_status.action_accepted)
        self._track_pending_chain_action(
            action_id,
            input_ready_before_step,
            action_accepted,
        )
        # Alternates from every remaining phase: an early queue can target a
        # phase more than one step ahead while combo_phase lags the DLL state.
        expected_alternates: dict[int, float] = {}
        alternate_min_combo: dict[int, int] = {}
        alternate_hit_timeouts: dict[int, int] = {}
        for phase_index in range(self.combo_phase, len(self.combo_scenario.phases)):
            future_phase = self.combo_scenario.phases[phase_index]
            previous_required = (
                self.combo_scenario.phases[phase_index - 1].required_combo
                if phase_index > 0
                else 0
            )
            for alternate_id, alternate_reward in future_phase.alternate_actions:
                if alternate_id not in expected_alternates:
                    expected_alternates[alternate_id] = alternate_reward
                    alternate_min_combo[alternate_id] = previous_required + 1
                    alternate_hit_timeouts[alternate_id] = (
                        SEVENTY_FIVE_KAI_FINISHER_HIT_TIMEOUTS.get(
                            alternate_id,
                            future_phase.hit_timeout_frames
                            or COMBO_PROFILE.action_hit_timeout,
                        )
                    )
        if (
            not input_ready_before_step
            and queueable_followup_phase is not None
            and action_id == queueable_followup_phase.action_id
            and action_accepted
            and (
                action_status.queued_action_id == action_id
                or action_status.active_action_id == action_id
                or action_status.last_started_action_id == action_id
            )
        ):
            self.pending_followups[action_id] = 0
            if previous is not None:
                self.pending_action_power_stocks = max(
                    0,
                    previous.p1_advanced_power_stocks,
                )
        elif (
            not input_ready_before_step
            and action_id in expected_alternates
            and action_accepted
            and (
                action_status.queued_action_id == action_id
                or action_status.active_action_id == action_id
                or action_status.last_started_action_id == action_id
            )
        ):
            self.pending_alternate_action = action_id
            self.pending_alternate_reward = expected_alternates[action_id]
            self.pending_alternate_age = 0
            self.pending_alternate_min_combo = alternate_min_combo[action_id]
            self.pending_alternate_power_stocks = (
                max(0, previous.p1_advanced_power_stocks)
                if previous is not None
                else None
            )
            self.pending_alternate_hit_timeout = alternate_hit_timeouts[action_id]
        self.episode_steps += 1

        if self.pending_chain_action is not None:
            self.pending_action_age += COMBO_PROFILE.action_repeat
        for pending_followup_action in self.pending_followups:
            self.pending_followups[pending_followup_action] += COMBO_PROFILE.action_repeat
        if self.pending_alternate_action is not None:
            self.pending_alternate_age += COMBO_PROFILE.action_repeat

        p1_damage = 0.0
        p2_damage = 0.0
        if previous is not None:
            if previous.p1_health >= 0 and observation.p1_health >= 0:
                p1_damage = float(max(0, previous.p1_health - observation.p1_health))
            if previous.p2_health >= 0 and observation.p2_health >= 0:
                p2_damage = float(max(0, previous.p2_health - observation.p2_health))

        phase_before_reward = self.combo_phase
        reward, reward_parts, combo_success, alternate_success = self._combo_reward(
            previous,
            observation,
            p1_damage,
            p2_damage,
            action_status,
        )
        wrong_action = (
            action_accepted
            and action_id != IDLE_ACTION_ID
            and action_id not in expected_alternates
            and (
                expected_phase_before_step is None
                or action_id != expected_phase_before_step.action_id
            )
        )
        reward_parts["wrong_action"] = 0.0
        if wrong_action:
            reward_parts["wrong_action"] -= COMBO_WRONG_ACTION_PENALTY
            reward -= COMBO_WRONG_ACTION_PENALTY
        phase_advanced = self.combo_phase > phase_before_reward
        advanced_phase = self._combo_phase_at(phase_before_reward)
        advanced_action = advanced_phase.action_id if advanced_phase is not None else None
        p1_ko = 0 <= observation.p1_health <= 0
        p2_ko = 0 <= observation.p2_health <= 0
        round_finished = observation.round_time == 0
        terminated = combo_success or alternate_success or p1_ko or p2_ko or round_finished
        truncated = not terminated and self.episode_steps >= COMBO_PROFILE.max_episode_steps

        encoded_observation = self._make_observation(
            observation,
            previous,
            p1_damage=p1_damage,
            step_events=step_events,
            last_action_accepted=action_accepted,
        )
        self.previous_observation = observation
        info = {
            "raw": observation,
            "training_profile": self.training_profile.value,
            "combo_scenario": self.combo_scenario.name,
            "action_mask_level": self.action_mask_level.value,
            "observation_version": self.observation_version.value,
            "combat_phase": float(self.combat_phase_machine.phase),
            "action": action_id,
            "frame_count": float(COMBO_PROFILE.action_repeat),
            "p1_health": observation.p1_health,
            "p2_health": observation.p2_health,
            "p1_damage": p1_damage,
            "p2_damage": p2_damage,
            "p1_combo_count": float(max(0, observation.p1_combo_count)),
            "p2_combo_count": float(max(0, observation.p2_combo_count)),
            "episode_max_combo": float(self.episode_max_combo),
            "combo_phase": float(self.combo_phase),
            "combo_success": float(combo_success),
            "combo_alternate_success": float(alternate_success),
            "combo_alternate_action": float(
                self.pending_alternate_action
                if self.pending_alternate_action is not None
                else -1
            ),
            "input_ready": float(self._combo_input_ready()),
            "input_accepted": float(action_accepted),
            "pending_chain_action": float(self.pending_chain_action if self.pending_chain_action is not None else -1),
            "pending_followup_count": float(len(self.pending_followups)),
            "phase_wait_remaining": float(self.phase_wait_remaining),
            "active_action_id": float(action_status.active_action_id),
            "queued_action_id": float(action_status.queued_action_id),
            "last_started_action_id": float(action_status.last_started_action_id),
            "action_14": float(action_id == ARAGAMI_ACTION_ID and action_accepted),
            "action_14_hit": float(phase_advanced and advanced_action == ARAGAMI_ACTION_ID),
            "action_15": float(action_id == KOTOTSUKI_YOU_ACTION_ID and action_accepted),
            "action_15_hit": float(phase_advanced and advanced_action == KOTOTSUKI_YOU_ACTION_ID),
            "action_17": float(action_id == RED_KICK_ACTION_ID and action_accepted),
            "action_17_hit": float(phase_advanced and advanced_action == RED_KICK_ACTION_ID),
            "action_18": float(action_id == OROCHINAGI_ACTION_ID and action_accepted),
            "action_18_hit": float(phase_advanced and advanced_action == OROCHINAGI_ACTION_ID),
            "action_19": float(action_id == MUSHIKI_ACTION_ID and action_accepted),
            "action_19_hit": float(phase_advanced and advanced_action == MUSHIKI_ACTION_ID),
            "action_22": float(action_id == FORWARD_B_ACTION_ID and action_accepted),
            "action_22_hit": float(phase_advanced and advanced_action == FORWARD_B_ACTION_ID),
            "action_23": float(action_id == POISON_BITE_ACTION_ID and action_accepted),
            "action_23_hit": float(phase_advanced and advanced_action == POISON_BITE_ACTION_ID),
            "action_24": float(action_id == TSUMI_YOMI_ACTION_ID and action_accepted),
            "action_24_hit": float(phase_advanced and advanced_action == TSUMI_YOMI_ACTION_ID),
            "action_25": float(action_id == BATSU_YOMI_ACTION_ID and action_accepted),
            "action_25_hit": float(phase_advanced and advanced_action == BATSU_YOMI_ACTION_ID),
            "action_26": float(action_id == SEVENTY_FIVE_SHIKI_KAI_ACTION_ID and action_accepted),
            "action_26_hit": float(phase_advanced and advanced_action == SEVENTY_FIVE_SHIKI_KAI_ACTION_ID),
            "action_27": float(action_id == YANO_SABI_ACTION_ID and action_accepted),
            "action_27_hit": float(phase_advanced and advanced_action == YANO_SABI_ACTION_ID),
            "action_28": float(action_id == MIGIRI_UGACHI_ACTION_ID and action_accepted),
            "action_28_hit": float(phase_advanced and advanced_action == MIGIRI_UGACHI_ACTION_ID),
            "reward_damage": reward_parts["damage"],
            "reward_phase": reward_parts["phase"],
            "reward_complete": reward_parts["complete"],
            "reward_alternate": reward_parts["alternate"],
            "reward_phase_reset": reward_parts["phase_reset"],
            "reward_timeout": reward_parts["timeout"],
            "reward_ko_without_combo": reward_parts["ko_without_combo"],
            "reward_time": reward_parts["time"],
            "reward_wrong_action": reward_parts["wrong_action"],
        }
        return encoded_observation, reward, terminated, truncated, info

    def _step_fight(self, action_id: int):
        """實戰的一步。與 combo 的關鍵差異:

        - 沒有指定路線,reward 來自傷害/連段長度/勝負(見 _reward)。
        - 傷害歸因用 StepEvents(逐幀事件)而非本 step 的 action_id ——
          該幀真正打中的可能是更早 queue 的招。
        - followup(cancel)追蹤:queue 被接受 → 發動(以 action_serial
          精確對應)→ COMBO_HIT 事件且 combo 遞增才算命中;
          被擋的 chip 傷害(DAMAGE_ONLY)不算,防止刷被擋超必。
        - 決策型態分類(free/queue/forced_idle)只做指標,不影響 reward。
        """
        previous = self.previous_observation
        previous_strategy_state = self.previous_strategy_state
        step_base_frame = self.level_episode_frames
        level_oracle_action_before = self._level_oracle_action()
        level_oracle_trajectory_valid_before = self.level_oracle_trajectory_valid
        dll_input_ready = bool(self.client.input_ready())
        free_decision = dll_input_ready and bool(self.client.p1_ready_for_action())
        action_availability = self._physical_action_mask()
        legal_action_count = int(action_availability.sum())
        queue_decision = not free_decision and legal_action_count > 1
        forced_idle = legal_action_count <= 1
        observation = self.client.step(action_id, self.action_repeat)
        action_status = self.client.action_status()
        step_events = self.client.step_events()
        self.last_step_events = list(step_events)
        timing_state_after_step = self.client.combat_timing_state()
        batch_epoch_mismatch = int(
            self.client.last_step_event_batch_epoch
            != int(timing_state_after_step.event_epoch)
        )

        # V5 的 chip 結果必須和 Hit 互斥。同幀、同攻擊方向若同時
        # 出現 Hit 與 Block，代表 provisional hit 沒有被正確移除。
        def event_contact_key(event: StepEventV5) -> tuple[int, int, int, int]:
            return (
                int(event.event_epoch),
                int(event.absolute_engine_frame),
                int(event.source_player),
                int(event.target_player),
            )

        hit_contact_keys = {
            event_contact_key(event)
            for event in step_events
            if event.event_type in (
                STEP_EVENT_COMBO_HIT,
                STEP_EVENT_DAMAGE_ONLY,
                STEP_EVENT_CLEAN_HIT,
            )
        }
        block_contact_keys = {
            event_contact_key(event)
            for event in step_events
            if event.event_type == STEP_EVENT_BLOCK_CONTACT
        }
        chip_hit_block_conflicts = len(hit_contact_keys & block_contact_keys)
        action_accepted = bool(action_status.action_accepted)
        self._advance_level_oracle(
            level_oracle_action_before,
            action_id,
            action_accepted,
        )
        hitbox_rects: list[HitboxRectResult] = []
        if self.hitbox_reward or self.level_recipe is not None:
            try:
                hitbox_rects, _axes = self.client.get_hitbox_overlay(
                    HITBOX_REWARD_SOURCE_WIDTH,
                    HITBOX_REWARD_SOURCE_HEIGHT,
                )
            except RuntimeError:
                hitbox_rects = []

        p1_attack_overlap = has_attack_hurtbox_overlap(hitbox_rects, HITBOX_OWNER_P1, HITBOX_OWNER_P2)
        p2_attack_overlap = has_attack_hurtbox_overlap(hitbox_rects, HITBOX_OWNER_P2, HITBOX_OWNER_P1)
        p2_attack_pressure = has_attack_hurtbox_overlap(
            hitbox_rects,
            HITBOX_OWNER_P2,
            HITBOX_OWNER_P1,
            DEFENSE_PRESSURE_MARGIN,
        )
        p1_attack_overlap_edge = p1_attack_overlap and not self.fight_prev_p1_attack_overlap
        p2_attack_overlap_edge = p2_attack_overlap and not self.fight_prev_p2_attack_overlap
        self.fight_prev_p1_attack_overlap = p1_attack_overlap
        self.fight_prev_p2_attack_overlap = p2_attack_overlap

        p2_damage = 0.0
        p1_damage = 0.0
        if previous is not None:
            if previous.p2_health >= 0 and observation.p2_health >= 0:
                p2_damage = float(max(0, previous.p2_health - observation.p2_health))
            if previous.p1_health >= 0 and observation.p1_health >= 0:
                p1_damage = float(max(0, previous.p1_health - observation.p1_health))

        block_contact_events = [
            event
            for event in step_events
            if event.event_type == STEP_EVENT_BLOCK_CONTACT
            and int(event.source_player) == 2
            and int(event.target_player) == 1
            and bool(event.block_contact)
        ]
        manual_block_events = [
            event
            for event in step_events
            if event.event_type == STEP_EVENT_MANUAL_BLOCK_SUCCESS
            and int(event.source_player) == 2
            and int(event.target_player) == 1
            and bool(event.block_contact)
        ]
        auto_guard_events = [
            event
            for event in step_events
            if event.event_type == STEP_EVENT_AUTO_GUARD
            and int(event.source_player) == 2
            and int(event.target_player) == 1
            and bool(event.block_contact)
        ]
        blockstun_started_events = [
            event
            for event in step_events
            if event.event_type == STEP_EVENT_BLOCKSTUN_STARTED
            and int(event.source_player) == 2
            and int(event.target_player) == 1
        ]
        blockstun_ended_events = [
            event
            for event in step_events
            if event.event_type == STEP_EVENT_BLOCKSTUN_ENDED
            and int(event.source_player) == 2
            and int(event.target_player) == 1
        ]
        p1_damage_events = [
            event
            for event in step_events
            if event.event_type in (
                STEP_EVENT_P1_DAMAGE,
                STEP_EVENT_CHIP_DAMAGE,
                STEP_EVENT_CLEAN_HIT,
            )
            and event.p1_hp_delta > 0
        ]
        if p1_damage_events:
            p1_damage = float(sum(int(event.p1_hp_delta) for event in p1_damage_events))
        clean_hit_events = [
            event
            for event in step_events
            if event.event_type == STEP_EVENT_CLEAN_HIT
            and event.p1_hp_delta > 0
        ]
        all_clean_hit_events = [
            event
            for event in step_events
            if event.event_type == STEP_EVENT_CLEAN_HIT
        ]

        previous_combo = max(0, previous.p1_combo_count) if previous is not None else 0
        current_combo = max(0, observation.p1_combo_count)
        combo_delta = max(0, current_combo - previous_combo)
        (
            recipe_route_started,
            recipe_route_committed,
            recipe_route_success,
            recipe_route_failure,
        ) = self._update_level_hit_confirm_route(step_events, current_combo)
        fight_teacher_completed_now = self._update_fight_teacher(
            action_id,
            action_accepted,
            action_status,
            previous_combo,
            current_combo,
            p1_damage,
        )

        # A follow-up is an action issued while the DLL runtime is busy that it
        # accepted into its queue (or that already fired within this step's
        # frames). Gate on DLL-level input_ready: setAction only accepts a
        # non-idle action while busy through canQueueAction, so acceptance
        # here proves a genuine queue rather than a raw start.
        queued_followup = (
            not dll_input_ready
            and action_accepted
            and action_id != IDLE_ACTION_ID
            and (
                action_status.queued_action_id == action_id
                or action_status.active_action_id == action_id
                or action_status.last_started_action_id == action_id
            )
        )
        queued_followup_actions: list[int] = []
        started_followup_actions: list[int] = []
        hit_followup_actions: list[int] = []
        if queued_followup:
            self.fight_pending_followups.append(
                {
                    "action": action_id,
                    "age": 0,
                    "started": False,
                    "serial": None,
                },
            )
            queued_followup_actions.append(action_id)

        for pending in self.fight_pending_followups:
            pending["age"] += self.action_repeat

        for event in step_events:
            if event.event_type == STEP_EVENT_ACTION_STARTED:
                for pending in self.fight_pending_followups:
                    if pending["started"] or pending["action"] != event.action_id:
                        continue

                    pending["started"] = True
                    pending["serial"] = int(event.action_serial)
                    started_followup_actions.append(pending["action"])
                    break
            elif event.event_type == STEP_EVENT_COMBO_HIT:
                for pending_index, pending in enumerate(self.fight_pending_followups):
                    if (
                        not pending["started"]
                        or pending["action"] != event.action_id
                        or pending["serial"] != int(event.action_serial)
                        or event.combo_after <= event.combo_before
                        or event.combo_after < 2
                    ):
                        continue

                    hit_followup_actions.append(pending["action"])
                    del self.fight_pending_followups[pending_index]
                    break

        self.fight_pending_followups = [
            pending
            for pending in self.fight_pending_followups
            if pending["age"] <= FIGHT_FOLLOWUP_TRACK_WINDOW_FRAMES
        ]

        hit_action_ids = {
            int(event.action_id)
            for event in step_events
            if event.event_type in (STEP_EVENT_COMBO_HIT, STEP_EVENT_DAMAGE_ONLY)
            and event.action_id >= 0
        }
        combo_hit_action_ids = {
            int(event.action_id)
            for event in step_events
            if event.event_type == STEP_EVENT_COMBO_HIT
            and event.action_id >= 0
        }
        damage_action_id = -1
        for event in step_events:
            if (
                event.event_type in (STEP_EVENT_COMBO_HIT, STEP_EVENT_DAMAGE_ONLY)
                and event.p2_hp_delta > 0
            ):
                damage_action_id = int(event.action_id)

        followup_action = -1
        for actions in (
            hit_followup_actions,
            started_followup_actions,
            queued_followup_actions,
        ):
            if actions:
                followup_action = actions[0]
                break
        if followup_action < 0 and self.fight_pending_followups:
            followup_action = self.fight_pending_followups[0]["action"]

        started_followup = bool(started_followup_actions)
        followup_hit = bool(hit_followup_actions)

        previous_distance = abs(previous.distance_x) if previous is not None else 0
        current_distance = abs(observation.distance_x)
        previous_p2_action = (
            previous_strategy_state.p2_active_action_id
            if previous_strategy_state is not None
            else -1
        )
        guard_action = action_id in GUARD_ACTION_IDS
        # Hitbox collection can be disabled for high-throughput StrategyV2
        # runs.  The fallback still uses the *actually active* P2 action from
        # the runtime, not the P2 style's intended script.
        tactical_guard_opportunity = p2_attack_pressure or (
            previous_p2_action >= 6
            and previous_distance <= TACTICAL_DEFENSE_DISTANCE
        )
        # BLOCK_CONTACT 是遊戲物理事件，可能來自 autoguard/counter。
        # Curriculum success 只接受同幀有 Action 2/3 後方向輸入的事件。
        tactical_guard_contact = bool(manual_block_events)
        guard_success = tactical_guard_contact and not clean_hit_events
        recipe_defense = (
            self.level_recipe is not None
            and self.level_recipe.task is CurriculumTask.DEFENSE
        )
        recipe_defense_opportunity = (
            recipe_defense
            and not self.level_guard_started
            and self.level_episode_frames
            <= max(16, int(self.level_recipe.trigger_frame or 0) + 8)
        )
        guard_held = (
            guard_action
            or action_status.active_action_id in GUARD_ACTION_IDS
            or (
                previous_strategy_state is not None
                and previous_strategy_state.p1_active_action_id
                in GUARD_ACTION_IDS
            )
        )
        recipe_guard_failure = False
        recipe_guard_success = False
        if recipe_defense:
            if (
                not self.level_guard_started
                and (tactical_guard_opportunity or recipe_defense_opportunity)
                and guard_action
                and action_accepted
            ):
                self.level_guard_started = True
                self.level_guard_hold_frames = self.action_repeat
                self.level_guard_damage = p1_damage
                self.level_guard_clean_hit = bool(clean_hit_events)
            elif self.level_guard_started:
                self.level_guard_damage += p1_damage
                self.level_guard_clean_hit = (
                    self.level_guard_clean_hit or bool(clean_hit_events)
                )
                if not self.level_guard_contact_seen:
                    if guard_held:
                        self.level_guard_hold_frames += self.action_repeat
                    elif free_decision:
                        recipe_guard_failure = True

            recipe_guard_failure = (
                recipe_guard_failure or self.level_guard_clean_hit
            )
            if tactical_guard_contact:
                self.level_guard_contact_seen = True
            recipe_guard_success = (
                tactical_guard_contact
                and not recipe_guard_failure
            )
        super_action = action_id in SUPER_ACTION_IDS
        super_available_before_action = can_use_super(previous)
        super_without_stock = super_action and not super_available_before_action
        p2_airborne_before_action = is_p2_airborne(previous)
        tactical_anti_air_opportunity = (
            p2_airborne_before_action
            and previous_distance <= TACTICAL_ANTI_AIR_DISTANCE
        )
        oniyaki_anti_air_hit = any(
            event.event_type == STEP_EVENT_COMBO_HIT
            and event.action_id == ONIYAKI_ACTION_ID
            and bool(event.hit_contact)
            and bool(event.target_airborne_at_event)
            for event in step_events
        )

        if self.fight_pending_opener_action is not None:
            self.fight_pending_opener_age += self.action_repeat
            if (
                self.fight_pending_opener_age
                > TACTICAL_OPENER_CONFIRM_WINDOW_FRAMES
            ):
                self.fight_pending_opener_action = None
                self.fight_pending_opener_serial = None
                self.fight_pending_opener_age = 0

        for event in step_events:
            if (
                event.event_type == STEP_EVENT_ACTION_STARTED
                and event.action_id in (CLOSE_C_ACTION_ID, CROUCH_B_ACTION_ID)
                and event.combo_before <= 0
            ):
                self.fight_pending_opener_action = int(event.action_id)
                self.fight_pending_opener_serial = int(event.action_serial)
                self.fight_pending_opener_age = 0

        opener_event = next((
            event
            for event in step_events
            if (
                event.event_type == STEP_EVENT_COMBO_HIT
                and event.combo_before <= 0
                and event.combo_after > event.combo_before
                and self.fight_pending_opener_action is not None
                and event.action_id == self.fight_pending_opener_action
                and int(event.action_serial) == self.fight_pending_opener_serial
            )
        ), None)
        combo_opened = opener_event is not None
        opener_confirmed = (
            combo_opened
            and self.fight_pending_opener_action is not None
        )
        confirmed_opener_action = (
            self.fight_pending_opener_action if opener_confirmed else -1
        )
        if opener_confirmed:
            self.fight_pending_opener_action = None
            self.fight_pending_opener_serial = None
            self.fight_pending_opener_age = 0
        confirm_window_active = self.fight_hit_confirm_window_frames > 0
        tactical_confirm_success = followup_hit and (
            confirm_window_active or opener_confirmed
        )
        tactical_confirm_opportunity = opener_confirmed
        if tactical_confirm_success:
            self.fight_hit_confirm_window_frames = 0
        elif opener_confirmed:
            self.fight_hit_confirm_window_frames = TACTICAL_HIT_CONFIRM_WINDOW_FRAMES
        else:
            self.fight_hit_confirm_window_frames = max(
                0,
                self.fight_hit_confirm_window_frames - self.action_repeat,
            )

        tactical_approach_opportunity = False
        tactical_approach_step_progress = 0.0
        tactical_approach_step_forward_frames = 0
        if (
            not self.fight_approach_pending
            and previous is not None
            and previous_distance > TACTICAL_EFFECTIVE_DISTANCE
        ):
            tactical_approach_opportunity = True
            self.fight_approach_pending = True
            self.fight_approach_p1_progress = 0.0
            self.fight_approach_forward_frames = 0
        if (
            self.fight_approach_pending
            and previous is not None
            and action_id == FORWARD_ACTION_ID
            and action_accepted
            and previous.p1_has_position
            and observation.p1_has_position
        ):
            p1_toward_delta = (
                observation.p1_x - previous.p1_x
                if previous.distance_x >= 0
                else previous.p1_x - observation.p1_x
            )
            tactical_approach_step_progress = float(max(0, p1_toward_delta))
            tactical_approach_step_forward_frames = self.action_repeat
            self.fight_approach_p1_progress += tactical_approach_step_progress
            self.fight_approach_forward_frames += tactical_approach_step_forward_frames
        tactical_approach_p1_progress = self.fight_approach_p1_progress
        tactical_approach_forward_frames = self.fight_approach_forward_frames
        tactical_approach_success = (
            self.fight_approach_pending
            and current_distance <= TACTICAL_EFFECTIVE_DISTANCE
            and p1_damage <= 0.0
            and tactical_approach_p1_progress >= TACTICAL_APPROACH_MIN_P1_PROGRESS
        )
        approach_window_closed = self.fight_approach_pending and (
            current_distance <= TACTICAL_EFFECTIVE_DISTANCE or p1_damage > 0.0
        )
        if approach_window_closed:
            self.fight_approach_pending = False
            self.fight_approach_p1_progress = 0.0
            self.fight_approach_forward_frames = 0

        curriculum_opportunity = False
        curriculum_success = False
        curriculum_guard_success = False
        curriculum_counter_opportunity = False
        curriculum_counter_success = False

        # BLOCKSTUN_ENDED 的下一個 emulator frame 才開啟 36F 確反窗。
        # 窗口採絕對幀 deadline，並把命中綁定到窗口內第一個實際啟動
        # 的攻擊 serial，避免舊招延遲判定或其他攻擊冒領確反獎勵。
        (
            counter_hit_events,
            _counter_window_opened_now,
            counter_window_existed_this_step,
            counter_window_expired_now,
        ) = self._process_defense_counter_events(
            step_events,
            step_base_frame,
        )
        counter_window_start_frame = self.defense_counter_window_start_frame
        counter_window_deadline_frame = self.defense_counter_window_deadline_frame
        counter_action_id = self.defense_counter_action_id
        counter_action_serial = self.defense_counter_action_serial

        if self.fight_curriculum is FightCurriculum.DEFENSE:
            defense_lesson = (
                self._level_recipe_lesson() if recipe_defense else "guard_only"
            )
            block_counter_lesson = defense_lesson == "block_counter"
            curriculum_opportunity = (
                tactical_guard_opportunity or recipe_defense_opportunity
            )
            curriculum_guard_success = (
                recipe_guard_success if recipe_defense else guard_success
            )
            curriculum_counter_opportunity = (
                block_counter_lesson
                and counter_window_existed_this_step
            )
            curriculum_counter_success = (
                curriculum_counter_opportunity
                and bool(counter_hit_events)
                and p1_damage <= 0.0
            )
            curriculum_success = (
                curriculum_counter_success
                if block_counter_lesson
                else curriculum_guard_success
            )
            if curriculum_counter_success:
                self._clear_defense_counter_window()
                self._clear_defense_blockstun_wait()
        elif self.fight_curriculum is FightCurriculum.ANTI_AIR:
            curriculum_opportunity = tactical_anti_air_opportunity
            curriculum_success = oniyaki_anti_air_hit
        elif self.fight_curriculum is FightCurriculum.APPROACH:
            curriculum_opportunity = tactical_approach_opportunity
            curriculum_success = tactical_approach_success
        elif self.fight_curriculum is FightCurriculum.HIT_CONFIRM:
            if self.level_recipe is not None:
                curriculum_opportunity = recipe_route_started
                curriculum_success = recipe_route_success
            else:
                curriculum_opportunity = tactical_confirm_opportunity
                curriculum_success = tactical_confirm_success
        elif self.fight_curriculum is FightCurriculum.COMBO_ROUTE:
            curriculum_opportunity = True
            curriculum_success = fight_teacher_completed_now

        curriculum_committed = False
        curriculum_failure = False
        machine_active = (
            self.tactical_reward_machine.phase
            is not RewardMachinePhase.WAITING
        )
        if self.fight_curriculum is FightCurriculum.DEFENSE:
            block_counter_lesson = (
                recipe_defense
                and self._level_recipe_lesson() == "block_counter"
            )
            curriculum_committed = (
                (
                    guard_action
                    and action_accepted
                    and (tactical_guard_opportunity or machine_active)
                )
                or (block_counter_lesson and self.level_guard_contact_seen)
            )
            curriculum_failure = (
                recipe_guard_failure
                if recipe_defense
                else machine_active and p1_damage > 0.0
            )
            if (
                block_counter_lesson
                and self.level_guard_contact_seen
                and counter_window_expired_now
                and not curriculum_counter_success
            ):
                curriculum_failure = True
        elif self.fight_curriculum is FightCurriculum.ANTI_AIR:
            curriculum_committed = (
                action_id == ONIYAKI_ACTION_ID
                and action_accepted
                and (tactical_anti_air_opportunity or machine_active)
            )
            curriculum_failure = (
                machine_active
                and not is_p2_airborne(observation)
                and not oniyaki_anti_air_hit
            )
        elif self.fight_curriculum is FightCurriculum.APPROACH:
            curriculum_committed = (
                action_id == FORWARD_ACTION_ID
                and action_accepted
                and tactical_approach_step_progress > 0.0
            )
            curriculum_failure = (
                machine_active
                and approach_window_closed
                and not tactical_approach_success
            )
        elif self.fight_curriculum is FightCurriculum.HIT_CONFIRM:
            if self.level_recipe is not None:
                curriculum_committed = recipe_route_committed
                curriculum_failure = recipe_route_failure
            else:
                curriculum_committed = (
                    (queued_followup or started_followup)
                    and (confirm_window_active or opener_confirmed or machine_active)
                )
                curriculum_failure = (
                    machine_active
                    and self.fight_hit_confirm_window_frames <= 0
                    and not tactical_confirm_success
                )

        reward_machine_transition = None
        if self.level_recipe is not None:
            reward_machine_transition = self.tactical_reward_machine.advance(
                opportunity=curriculum_opportunity,
                committed=curriculum_committed,
                success=curriculum_success,
                failure=curriculum_failure,
            )
        if counter_window_expired_now:
            self._clear_defense_counter_window()
        reward, reward_parts = self._reward(
            previous,
            observation,
            action_id,
            damage_action_id,
            combo_hit_action_ids,
            p1_damage,
            p1_attack_overlap_edge,
            p2_attack_overlap_edge,
            p2_attack_pressure,
            super_available_before_action,
            p2_airborne_before_action,
            self.fight_frame_scale,
        )
        curriculum_reward = 0.0
        if (
            reward_machine_transition is not None
            and self.fight_reward_version
            is FightRewardVersion.SYMMETRIC_TACTICAL_V3
        ):
            if reward_machine_transition.success:
                curriculum_reward = TACTICAL_REWARD_MACHINE_SUCCESS
            elif reward_machine_transition.failure:
                curriculum_reward = TACTICAL_REWARD_MACHINE_FAILURE
            reward += curriculum_reward
        reward_parts["curriculum"] = curriculum_reward
        safety_info = {
            "pending": 0.0,
            "punished": 0.0,
            "unsafe_close": 0.0,
            "safe": 0.0,
        }
        reward_parts["cancel"] = 0.0
        reward_parts["combo_4plus_milestone"] = 0.0
        if self.fight_reward_version is FightRewardVersion.LEGACY_COMBO4:
            safety_reward, safety_info = self._update_attack_safety(
                action_id,
                observation,
                p1_damage,
                p2_damage,
            )
            reward += safety_reward
            reward_parts["safety"] = safety_reward
            combo_4plus_milestone = (
                not self.fight_combo_4plus_rewarded
                and previous_combo < FIGHT_COMBO_4PLUS_MILESTONE_HITS
                and current_combo >= FIGHT_COMBO_4PLUS_MILESTONE_HITS
            )
            reward_parts["combo_4plus_milestone"] = (
                FIGHT_COMBO_4PLUS_MILESTONE_REWARD
                if combo_4plus_milestone
                else 0.0
            )
            if combo_4plus_milestone:
                self.fight_combo_4plus_rewarded = True
                reward += reward_parts["combo_4plus_milestone"]
        encoded_observation = self._make_observation(
            observation,
            previous,
            p1_damage=p1_damage,
            step_events=step_events,
            p2_attack_pressure=p2_attack_pressure,
            last_action_accepted=action_accepted,
        )
        generic_combat_state = self.generic_combat_state_machine.snapshot()
        p1_reaction_active = bool(timing_state_after_step.p1.reaction_valid)
        p2_reaction_active = bool(timing_state_after_step.p2.reaction_valid)
        p1_reaction_remaining_valid = bool(
            timing_state_after_step.p1.reaction_remaining_valid
        )
        p2_reaction_remaining_valid = bool(
            timing_state_after_step.p2.reaction_remaining_valid
        )
        defense_non_neutral = (
            generic_combat_state.defense_phase is not DefensePhase.NEUTRAL
        )
        confirm_non_neutral = (
            generic_combat_state.confirm_phase is not ConfirmPhase.NEUTRAL
        )
        observation_finite = bool(np.isfinite(encoded_observation).all())
        event_feature_nonzero_count = 0
        if self.observation_version is ObservationVersion.V3:
            event_feature_nonzero_count = int(np.count_nonzero(
                encoded_observation[
                    list(OBSERVATION_V3_REPURPOSED_INDICES)
                ],
            ))
        self.previous_observation = observation

        self.level_episode_frames += self.action_repeat
        level_success = (
            self.level_recipe is not None
            and self.tactical_reward_machine.phase
            is RewardMachinePhase.SUCCEEDED
        )
        level_failure = (
            self.level_recipe is not None
            and self.tactical_reward_machine.phase
            is RewardMachinePhase.FAILED
        )
        level_timeout = (
            self.level_recipe is not None
            and not level_success
            and not level_failure
            and self.level_episode_frames >= self.level_recipe.settle_frames
        )
        if (
            level_timeout
            and self.fight_reward_version
            is FightRewardVersion.SYMMETRIC_TACTICAL_V3
        ):
            reward += TACTICAL_REWARD_MACHINE_TIMEOUT
            reward_parts["curriculum"] += TACTICAL_REWARD_MACHINE_TIMEOUT

        terminated = (
            0 <= observation.p1_health <= 0
            or 0 <= observation.p2_health <= 0
            or observation.round_time == 0
            or level_success
            or level_failure
        )
        truncated = bool(level_timeout and not terminated)
        fight_outcome = ""
        if terminated:
            p1_ko = 0 <= observation.p1_health <= 0
            p2_ko = 0 <= observation.p2_health <= 0
            if p1_ko and p2_ko:
                fight_outcome = "draw_ko"
            elif p2_ko:
                fight_outcome = "win_ko"
            elif p1_ko:
                fight_outcome = "loss_ko"
            elif observation.p1_health > observation.p2_health:
                fight_outcome = "win_timeout"
            elif observation.p1_health < observation.p2_health:
                fight_outcome = "loss_timeout"
            else:
                fight_outcome = "draw_timeout"
        info = {
            "raw": observation,
            "training_profile": self.training_profile.value,
            "fight_guided": float(self.fight_guided),
            "fight_curriculum": self.fight_curriculum.value,
            "curriculum_opportunity": float(curriculum_opportunity),
            "curriculum_success": float(curriculum_success),
            "curriculum_committed": float(curriculum_committed),
            "curriculum_failure": float(curriculum_failure),
            "reward_machine_phase": float(self.tactical_reward_machine.phase),
            "reward_machine_success": float(level_success),
            "reward_machine_failure": float(level_failure),
            "reward_machine_timeout": float(level_timeout),
            "curriculum_guard_success": float(curriculum_guard_success),
            "curriculum_counter_opportunity": float(curriculum_counter_opportunity),
            "curriculum_counter_success": float(curriculum_counter_success),
            "p2_style": self.p2_style.value,
            "level_recipe": self.level_recipe.name if self.level_recipe else "",
            "curriculum_level": float(self.level_recipe.level if self.level_recipe else -1),
            "curriculum_task": self.level_recipe.task.value if self.level_recipe else "",
            "curriculum_lesson": self._level_recipe_lesson(),
            "curriculum_oracle_action_before": float(level_oracle_action_before),
            "curriculum_oracle_trajectory_valid_before": float(
                level_oracle_trajectory_valid_before
            ),
            "curriculum_oracle_trajectory_valid": float(
                self.level_oracle_trajectory_valid
            ),
            "curriculum_oracle_action_index": float(
                self.level_oracle_action_index
            ),
            "level_p2_start_delay_remaining": float(
                self.level_recipe_p2_start_delay_remaining
            ),
            "level_p2_started": float(self.level_recipe_p2_started),
            "level_episode_frames": float(self.level_episode_frames),
            "fight_reward_version": self.fight_reward_version.value,
            "observation_version": self.observation_version.value,
            "combat_phase": float(self.combat_phase_machine.phase),
            "fight_teacher_phase": float(self.fight_teacher_phase),
            "fight_teacher_complete": float(fight_teacher_completed_now),
            "action": action_id,
            "input_accepted": float(action_accepted),
            "active_action_id": float(action_status.active_action_id),
            "queued_action_id": float(action_status.queued_action_id),
            "last_started_action_id": float(action_status.last_started_action_id),
            "frame_count": float(self.action_repeat),
            "free_decision": float(free_decision),
            "queue_decision": float(queue_decision),
            "forced_idle": float(forced_idle),
            "legal_action_count": float(legal_action_count),
            "action_availability": action_availability.copy(),
            "queued_followup": float(queued_followup),
            "started_followup": float(started_followup),
            "followup_hit": float(followup_hit),
            "followup_action": float(followup_action),
            "queued_followup_actions": queued_followup_actions,
            "started_followup_actions": started_followup_actions,
            "hit_followup_actions": hit_followup_actions,
            "step_event_count": float(len(step_events)),
            "step_event_dropped": float(
                self.client.last_step_event_dropped_count
            ),
            "step_event_batch_epoch_mismatch": float(batch_epoch_mismatch),
            "step_event_chip_hit_block_conflict": float(
                chip_hit_block_conflicts
            ),
            "observation_finite": float(observation_finite),
            "observation_event_features_enabled": float(
                self.observation_version is ObservationVersion.V3
                and self.observation_event_features
            ),
            "observation_event_nonzero_count": float(
                event_feature_nonzero_count
            ),
            "p1_reaction_active": float(p1_reaction_active),
            "p1_reaction_remaining_valid": float(
                p1_reaction_remaining_valid
            ),
            "p2_reaction_active": float(p2_reaction_active),
            "p2_reaction_remaining_valid": float(
                p2_reaction_remaining_valid
            ),
            "defense_non_neutral": float(defense_non_neutral),
            "confirm_non_neutral": float(confirm_non_neutral),
            "step_block_contact_count": float(len(block_contact_events)),
            "step_clean_hit_count": float(len(all_clean_hit_events)),
            "step_manual_block_count": float(len(manual_block_events)),
            "step_auto_guard_count": float(len(auto_guard_events)),
            "step_blockstun_started_count": float(len(blockstun_started_events)),
            "step_blockstun_ended_count": float(len(blockstun_ended_events)),
            "step_starter_hit_count": float(
                self.generic_combat_state_machine.last_update_starter_hit_count
            ),
            "step_starter_blocked_count": float(
                self.generic_combat_state_machine.last_update_starter_blocked_count
            ),
            "defense_blockstun_pending": float(self.defense_blockstun_pending),
            "defense_counter_window_frames": float(self.defense_counter_window_frames),
            "defense_counter_window_start_frame": float(counter_window_start_frame),
            "defense_counter_window_deadline_frame": float(counter_window_deadline_frame),
            "defense_counter_action_id": float(counter_action_id),
            "defense_counter_action_serial": float(counter_action_serial),
            "step_p1_damage_event_count": float(len(p1_damage_events)),
            "damage_action_id": float(damage_action_id),
            "step_hit_action_ids": sorted(hit_action_ids),
            "step_combo_hit_action_ids": sorted(combo_hit_action_ids),
            "fight_outcome": fight_outcome,
            "p1_health": observation.p1_health,
            "p2_health": observation.p2_health,
            "p1_advanced_power_value": float(max(0, observation.p1_advanced_power_value)),
            "p1_advanced_power_stocks": float(max(0, observation.p1_advanced_power_stocks)),
            "p2_advanced_power_value": float(max(0, observation.p2_advanced_power_value)),
            "p2_advanced_power_stocks": float(max(0, observation.p2_advanced_power_stocks)),
            "p1_damage": p1_damage,
            "p2_damage": p2_damage,
            "p1_combo_count": float(max(0, observation.p1_combo_count)),
            "p2_combo_count": float(max(0, observation.p2_combo_count)),
            "p1_attack_overlap": float(p1_attack_overlap),
            "p2_attack_overlap": float(p2_attack_overlap),
            "p2_attack_pressure": float(p2_attack_pressure),
            "guard_action": float(guard_action),
            "guard_success": float(guard_success),
            "super_available": float(super_available_before_action),
            "super_without_stock": float(super_without_stock),
            "p2_airborne": float(p2_airborne_before_action),
            "oniyaki_anti_air_hit": float(oniyaki_anti_air_hit),
            "tactical_guard_opportunity": float(tactical_guard_opportunity),
            "tactical_guard_success": float(guard_success),
            "tactical_anti_air_opportunity": float(tactical_anti_air_opportunity),
            "tactical_anti_air_success": float(oniyaki_anti_air_hit),
            "tactical_approach_opportunity": float(tactical_approach_opportunity),
            "tactical_approach_success": float(tactical_approach_success),
            "tactical_approach_p1_progress": tactical_approach_p1_progress,
            "tactical_approach_forward_frames": float(tactical_approach_forward_frames),
            "tactical_approach_step_progress": tactical_approach_step_progress,
            "tactical_approach_step_forward_frames": float(
                tactical_approach_step_forward_frames
            ),
            "tactical_confirm_opportunity": float(tactical_confirm_opportunity),
            "tactical_confirm_success": float(tactical_confirm_success),
            "confirmed_opener_action": float(confirmed_opener_action),
            "pending_opener_action": float(
                self.fight_pending_opener_action
                if self.fight_pending_opener_action is not None
                else -1
            ),
            "pending_opener_serial": float(
                self.fight_pending_opener_serial
                if self.fight_pending_opener_serial is not None
                else 0
            ),
            "recipe_route_started_actions": float(
                self.level_route_next_action_index
            ),
            "recipe_route_hit_actions": float(len(self.level_route_hit_indices)),
            "recipe_route_invalid": float(self.level_route_invalid),
            "recipe_guard_hold_frames": float(self.level_guard_hold_frames),
            "attack_safety_pending": safety_info["pending"],
            "attack_safety_punished": safety_info["punished"],
            "attack_safety_unsafe_close": safety_info["unsafe_close"],
            "attack_safety_safe": safety_info["safe"],
            "action_14": float(action_id == 14),
            "action_14_hit": float(14 in hit_action_ids),
            "action_15": float(action_id == 15),
            "action_15_hit": float(15 in hit_action_ids),
            "action_16": float(action_id == 16),
            "action_16_hit": float(16 in hit_action_ids),
            "action_17": float(action_id == 17),
            "action_17_hit": float(17 in hit_action_ids),
            "action_18": float(action_id == 18),
            "action_18_hit": float(18 in hit_action_ids),
            "action_19": float(action_id == 19),
            "action_19_hit": float(19 in hit_action_ids),
            "action_20": float(action_id == 20),
            "action_20_hit": float(20 in hit_action_ids),
            "action_21": float(action_id == 21),
            "action_21_hit": float(21 in hit_action_ids),
            "action_22": float(action_id == 22),
            "action_22_hit": float(22 in hit_action_ids),
            "action_23": float(action_id == 23),
            "action_23_hit": float(23 in hit_action_ids),
            "action_24": float(action_id == 24),
            "action_24_hit": float(24 in hit_action_ids),
            "action_25": float(action_id == 25),
            "action_25_hit": float(25 in hit_action_ids),
            "action_26": float(action_id == SEVENTY_FIVE_SHIKI_KAI_ACTION_ID),
            "action_26_hit": float(SEVENTY_FIVE_SHIKI_KAI_ACTION_ID in hit_action_ids),
            "action_27": float(action_id == YANO_SABI_ACTION_ID),
            "action_27_hit": float(YANO_SABI_ACTION_ID in hit_action_ids),
            "action_28": float(action_id == MIGIRI_UGACHI_ACTION_ID),
            "action_28_hit": float(MIGIRI_UGACHI_ACTION_ID in hit_action_ids),
            "distance_x_abs": float(abs(observation.distance_x)),
            "reward_hp": reward_parts["hp"],
            "reward_hitbox": reward_parts["hitbox"],
            "reward_distance": reward_parts["distance"],
            "reward_defense": reward_parts["defense"],
            "reward_combo": reward_parts["combo"],
            "reward_fight_combo_4plus_milestone": reward_parts["combo_4plus_milestone"],
            "reward_super": reward_parts["super"],
            "reward_anti_air": reward_parts["anti_air"],
            "reward_safety": reward_parts["safety"],
            "reward_cancel": reward_parts["cancel"],
            "reward_fast_win": reward_parts["fast_win"],
            "reward_outcome": reward_parts["outcome"],
            "reward_time": reward_parts["time"],
            "reward_curriculum": reward_parts["curriculum"],
        }
        return encoded_observation, reward, terminated, truncated, info

    def _combo_reward(
        self,
        previous: Optional[Kof98Observation],
        current: Kof98Observation,
        p1_damage: float,
        p2_damage: float,
        action_status: ActionStatus,
    ) -> tuple[float, dict[str, float], bool, bool]:
        """Combo 課程的 reward 與 phase 狀態機。回傳
        (reward, 分項明細, 指定路線完成?, 替代收尾完成?)。

        Phase 推進的三重驗證(缺一不可,都是為了堵誤判):
          1. pending_action_hit:pending 的招在時限內且確實 active/發動過
          2. continuity_valid:combo 計數達標且(可選)嚴格遞增
          3. require_power_stock_spent:超必需驗證氣真的花掉
        斷鏈(chain_expired)條件:被打、超過 chain_timeout 沒新命中、
        combo 計數歸零 —— 觸發 phase_reset 懲罰並清空所有 pending。
        """
        reward_parts = {
            "damage": 0.0,
            "phase": 0.0,
            "complete": 0.0,
            "alternate": 0.0,
            "phase_reset": 0.0,
            "timeout": 0.0,
            "ko_without_combo": 0.0,
            "time": -COMBO_PROFILE.time_penalty,
        }
        if previous is None:
            return 0.0, reward_parts, False, False

        if p2_damage > 0.0:
            reward_parts["damage"] += min(
                p2_damage * COMBO_PROFILE.damage_reward,
                COMBO_PROFILE.damage_reward_cap,
            )

        previous_combo = max(0, previous.p1_combo_count)
        current_combo = max(0, current.p1_combo_count)
        self.episode_max_combo = max(self.episode_max_combo, current_combo)

        phase = self._current_combo_phase()
        expected_action = phase.action_id if phase is not None else None
        phase_hit_timeout = (
            phase.hit_timeout_frames
            if phase is not None and phase.hit_timeout_frames is not None
            else COMBO_PROFILE.action_hit_timeout
        )
        pending_action_hit = (
            phase is not None
            and (p2_damage > 0.0 or not phase.require_damage)
            and self.pending_chain_action == expected_action
            and self.pending_action_age <= phase_hit_timeout
            and (
                not phase.queue_during_previous
                or action_status.active_action_id == expected_action
                or action_status.last_started_action_id == expected_action
                or phase.allow_started_followup_on_hit
            )
        )

        if phase is not None and phase.require_combo_increment:
            continuity_valid = (
                previous_combo >= max(0, phase.required_combo - 1)
                and current_combo >= phase.required_combo
                and current_combo > previous_combo
            )
        else:
            continuity_valid = (
                phase is not None
                and current_combo >= phase.required_combo
            )

        phase_advanced = pending_action_hit and continuity_valid
        if phase is not None and phase.require_power_stock_spent:
            phase_advanced = (
                phase_advanced
                and self.pending_action_power_stocks is not None
                and 0 <= current.p1_advanced_power_stocks
                < self.pending_action_power_stocks
            )
        if phase_advanced:
            reward_parts["phase"] += phase.reward
            self.combo_phase += 1
            self.phase_wait_remaining = phase.wait_after_hit_frames
            next_phase = self._current_combo_phase()
            followup_age = (
                self.pending_followups.get(next_phase.action_id)
                if next_phase is not None
                else None
            )
            followup_is_queued_or_started = (
                next_phase is not None
                and next_phase.queue_during_previous
                and followup_age is not None
                and (
                    action_status.queued_action_id == next_phase.action_id
                    or action_status.active_action_id == next_phase.action_id
                    or action_status.last_started_action_id == next_phase.action_id
                )
            )
            if followup_is_queued_or_started:
                self.pending_chain_action = next_phase.action_id
                self.pending_action_age = followup_age
                del self.pending_followups[next_phase.action_id]
            else:
                self.pending_chain_action = None
                self.pending_action_age = 0
                self.pending_action_power_stocks = None
            remaining_phases = self.combo_scenario.phases[self.combo_phase:]
            remaining_actions = {phase.action_id for phase in remaining_phases}
            self.pending_followups = {
                pending_action: pending_age
                for pending_action, pending_age in self.pending_followups.items()
                if pending_action in remaining_actions
            }
            remaining_alternates = {
                alternate_id
                for phase in remaining_phases
                for alternate_id, _ in phase.alternate_actions
            }
            if self.pending_alternate_action not in remaining_alternates:
                self.pending_alternate_action = None
                self.pending_alternate_reward = 0.0
                self.pending_alternate_age = 0
                self.pending_alternate_min_combo = 0
                self.pending_alternate_power_stocks = None
                self.pending_alternate_hit_timeout = COMBO_PROFILE.action_hit_timeout
            self.frames_since_chain_hit = 0
        elif self.combo_phase > 0:
            self.frames_since_chain_hit += COMBO_PROFILE.action_repeat

        combo_success = self.combo_phase >= len(self.combo_scenario.phases)
        if combo_success:
            reward_parts["complete"] += self.combo_scenario.complete_reward
            return float(sum(reward_parts.values())), reward_parts, True, False

        alternate_power_valid = (
            self.pending_alternate_action not in SUPER_ACTION_IDS
            or (
                self.pending_alternate_power_stocks is not None
                and 0 <= current.p1_advanced_power_stocks
                < self.pending_alternate_power_stocks
            )
        )
        alternate_hit = (
            not phase_advanced
            and self.pending_alternate_action is not None
            and current_combo > previous_combo
            and current_combo >= self.pending_alternate_min_combo
            and self.pending_alternate_age <= self.pending_alternate_hit_timeout
            and alternate_power_valid
            # The queued sibling must actually have fired; otherwise hits from
            # the still-active opener would bank the alternate reward early.
            and (
                action_status.active_action_id == self.pending_alternate_action
                or action_status.last_started_action_id == self.pending_alternate_action
            )
        )
        if alternate_hit:
            reward_parts["alternate"] += self.pending_alternate_reward
            return float(sum(reward_parts.values())), reward_parts, False, True

        p2_ko = previous.p2_health > 0 and 0 <= current.p2_health <= 0
        if p2_ko:
            reward_parts["ko_without_combo"] -= COMBO_PROFILE.ko_without_combo_penalty

        chain_expired = False
        if self.combo_phase > 0 and p1_damage > 0.0:
            chain_expired = True
        if (
            self.combo_phase > 0
            and self.frames_since_chain_hit > COMBO_PROFILE.chain_timeout
        ):
            chain_expired = True
        if (
            self.combo_phase > 0
            and previous_combo > 0
            and current_combo == 0
        ):
            chain_expired = True
        if (
            self.pending_chain_action is not None
            and self.pending_action_age > phase_hit_timeout
        ):
            self.pending_chain_action = None
            self.pending_action_age = 0
            self.pending_action_power_stocks = None
        if (
            self.pending_alternate_action is not None
            and self.pending_alternate_age > self.pending_alternate_hit_timeout
        ):
            self.pending_alternate_action = None
            self.pending_alternate_reward = 0.0
            self.pending_alternate_age = 0
            self.pending_alternate_min_combo = 0
            self.pending_alternate_power_stocks = None
            self.pending_alternate_hit_timeout = COMBO_PROFILE.action_hit_timeout

        if chain_expired:
            reward_parts["phase_reset"] -= COMBO_PROFILE.phase_reset_penalty
            self.combo_phase = 0
            self.pending_chain_action = None
            self.pending_action_age = 0
            self.pending_action_power_stocks = None
            self.pending_followups = {}
            self.pending_alternate_action = None
            self.pending_alternate_reward = 0.0
            self.pending_alternate_age = 0
            self.pending_alternate_min_combo = 0
            self.pending_alternate_power_stocks = None
            self.pending_alternate_hit_timeout = COMBO_PROFILE.action_hit_timeout
            self.frames_since_chain_hit = 0
            self.phase_wait_remaining = 0

        timeout = self.episode_steps >= COMBO_PROFILE.max_episode_steps
        if timeout:
            reward_parts["timeout"] -= COMBO_PROFILE.episode_timeout_penalty

        return float(sum(reward_parts.values())), reward_parts, combo_success, False

    def _track_pending_chain_action(
        self,
        action_id: int,
        input_ready: bool,
        action_accepted: bool,
    ) -> None:
        """input ready(非 queue)路徑的按招登記:按對 = 設 pending_chain、
        按到兄弟收尾 = 設 pending_alternate、按錯 = 全部清空。
        busy 中 queue 的路徑在 _step_combo 裡另外處理。"""
        if not input_ready or not action_accepted:
            return

        phase = self._current_combo_phase()
        expected_action = phase.action_id if phase is not None else None
        alternate_rewards = dict(phase.alternate_actions) if phase is not None else {}
        if action_id == expected_action:
            self.pending_chain_action = action_id
            self.pending_action_age = 0
            if self.previous_observation is not None:
                self.pending_action_power_stocks = max(
                    0,
                    self.previous_observation.p1_advanced_power_stocks,
                )
            self.pending_alternate_action = None
            self.pending_alternate_reward = 0.0
            self.pending_alternate_age = 0
            self.pending_alternate_min_combo = 0
            self.pending_alternate_power_stocks = None
            self.pending_alternate_hit_timeout = COMBO_PROFILE.action_hit_timeout
        elif action_id in alternate_rewards:
            self.pending_alternate_action = action_id
            self.pending_alternate_reward = alternate_rewards[action_id]
            self.pending_alternate_age = 0
            self.pending_alternate_min_combo = (
                self.combo_scenario.phases[self.combo_phase - 1].required_combo + 1
                if self.combo_phase > 0
                else 1
            )
            self.pending_alternate_power_stocks = (
                max(0, self.previous_observation.p1_advanced_power_stocks)
                if self.previous_observation is not None
                else None
            )
            self.pending_alternate_hit_timeout = (
                SEVENTY_FIVE_KAI_FINISHER_HIT_TIMEOUTS.get(
                    action_id,
                    phase.hit_timeout_frames or COMBO_PROFILE.action_hit_timeout,
                )
            )
            self.pending_chain_action = None
            self.pending_action_age = 0
            self.pending_action_power_stocks = None
        elif action_id != IDLE_ACTION_ID:
            self.pending_chain_action = None
            self.pending_action_age = 0
            self.pending_action_power_stocks = None
            self.pending_alternate_action = None
            self.pending_alternate_reward = 0.0
            self.pending_alternate_age = 0
            self.pending_alternate_min_combo = 0
            self.pending_alternate_power_stocks = None
            self.pending_alternate_hit_timeout = COMBO_PROFILE.action_hit_timeout

    def _combo_input_ready(self) -> bool:
        """Combo 視角的「可以出新招了嗎」:DLL 空閒 + 角色動畫可行動 +
        沒有演出等待 + 沒有尚未結算的 pending 招。比 DLL 原生的
        input_ready 更嚴,避免在等待命中判定時搶跑下一招。"""
        if (
            not self.client.input_ready()
            or not self.client.p1_ready_for_action()
            or self.phase_wait_remaining > 0
        ):
            return False
        return self.pending_chain_action is None

    def _make_observation(
        self,
        observation: Kof98Observation,
        previous_observation: Optional[Kof98Observation],
        *,
        p1_damage: float = 0.0,
        step_events: Optional[list[StepEventV5]] = None,
        p2_attack_pressure: bool = False,
        last_action_accepted: bool = False,
        elapsed_frames: Optional[int] = None,
    ) -> np.ndarray:
        input_ready = self.client.input_ready() and self.client.p1_ready_for_action()
        if self.training_profile is TrainingProfile.COMBO:
            input_ready = self._combo_input_ready()
        normalized_phase = float(self.combo_phase) / float(
            max(1, len(self.combo_scenario.phases))
        )
        # Fight observations keep this legacy scalar neutral.  Exposing the
        # recipe-only reward-machine phase here would become a teacher side
        # channel that is absent in unrestricted fights and human play.
        if self.training_profile is TrainingProfile.FIGHT:
            normalized_phase = 0.0
        phase_age = min(
            float(self.frames_since_chain_hit) / COMBO_PHASE_AGE_SCALE_FRAMES,
            1.0,
        )
        strategy_state = None
        timing_state = None
        generic_combat_state = self.generic_combat_state_machine.snapshot()
        if self.observation_version is not ObservationVersion.V1:
            strategy_state = self.client.strategy_state()
            self.combat_phase_machine.update(
                observation,
                strategy_state,
                p1_damage=p1_damage,
            )
            self.previous_strategy_state = strategy_state
        if self.observation_version is ObservationVersion.V3:
            timing_state = self.client.combat_timing_state()
            queue_window_open = (
                not self.client.input_ready()
                and any(
                    self.client.can_queue_action(action_id)
                    for action_id in range(1, ACTION_COUNT)
                )
            )
            generic_combat_state = self.generic_combat_state_machine.update(
                step_events or (),
                timing_state,
                elapsed_frames=(
                    self.action_repeat
                    if elapsed_frames is None
                    else max(0, int(elapsed_frames))
                ),
                p2_attack_pressure=p2_attack_pressure,
                queue_window_open=queue_window_open,
                last_action_accepted=last_action_accepted,
            )
        return encode_observation(
            self.observation_version,
            observation,
            ObservationContext(
                input_ready=bool(input_ready),
                normalized_combo_phase=normalized_phase,
                combo_phase_age=phase_age,
                action_repeat=self.action_repeat,
                combat_phase=self.combat_phase_machine.phase,
                profile_is_fight=self.training_profile is TrainingProfile.FIGHT,
                strategy_state=strategy_state,
                timing_state=timing_state,
                generic_combat_state=generic_combat_state,
                event_features_enabled=self.observation_event_features,
                previous_observation=previous_observation,
            ),
        )

    def _physical_action_mask(self) -> np.ndarray:
        """物理合法性 mask(最寬的一層):空閒時全開(無氣時遮超必),
        忙碌時只開 Idle + DLL 判定當下可 queue 的派生技。
        這是 fight 的常駐 mask,也是 combo 課程 physical 級的 mask。"""
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[IDLE_ACTION_ID] = True
        super_available = can_use_super(self.previous_observation)
        if self.client.input_ready():
            # Action 1..4 是可中斷的 4-frame Hold chunk。角色進入走路或
            # 防禦姿勢後，hitbox 形狀判斷的 p1_ready_for_action() 可能暫時
            # 為 false；此時仍必須允許下一個 chunk，否則方向會在每四幀
            # 之間被迫放開，慢攻擊接觸時只會得到 Auto Guard。
            for action_id in P1_HOLD_ACTION_IDS:
                mask[action_id] = True

            if self.client.p1_ready_for_action():
                mask[:] = True
                if not super_available:
                    for action_id in SUPER_ACTION_IDS:
                        mask[action_id] = False
            return mask

        for action_id in range(1, self.action_space.n):
            if action_id in SUPER_ACTION_IDS and not super_available:
                continue
            if self.client.can_queue_action(action_id):
                mask[action_id] = True
        return mask

    def _fight_teacher_expected_action(self) -> Optional[int]:
        """Fight teacher(實戰連段課程):guided fight 環境中,依
        combo_scenario 的 phase 順序逐招引導;走完或未啟用回傳 None。"""
        if not self.fight_guided or self.fight_teacher_route_completed:
            return None
        if 0 <= self.fight_teacher_phase < len(self.combo_scenario.phases):
            return self.combo_scenario.phases[self.fight_teacher_phase].action_id
        return None

    def _guided_fight_action_mask(self, physical_mask: np.ndarray) -> np.ndarray:
        expected_action = self._fight_teacher_expected_action()
        if expected_action is None:
            return physical_mask

        # Neutral remains a real Fight problem. The teacher only takes over
        # once Close C is in range, then exposes the next move exclusively
        # when the DLL says that move can physically start or be queued.
        if (
            self.fight_teacher_phase == 0
            and self.previous_observation is not None
            and abs(self.previous_observation.distance_x) > COMBO_CLOSE_DISTANCE
        ):
            return physical_mask

        mask = np.zeros(self.action_space.n, dtype=bool)
        if physical_mask[expected_action]:
            mask[expected_action] = True
            if self.fight_teacher_phase > 0:
                mask[IDLE_ACTION_ID] = True
        else:
            mask[IDLE_ACTION_ID] = True
        return mask

    def _targeted_fight_action_mask(self, physical_mask: np.ndarray) -> np.ndarray:
        curriculum = self.fight_curriculum
        if curriculum in (FightCurriculum.COMBO_ROUTE, FightCurriculum.HIT_CONFIRM):
            return self._guided_fight_action_mask(physical_mask)
        if curriculum is FightCurriculum.NONE or self.previous_observation is None:
            return physical_mask

        targeted = np.zeros(self.action_space.n, dtype=bool)
        targeted[IDLE_ACTION_ID] = True
        distance = abs(self.previous_observation.distance_x)
        opponent_action = (
            self.previous_strategy_state.p2_active_action_id
            if self.previous_strategy_state is not None
            else -1
        )
        opponent_attacking = opponent_action >= 6

        if (
            curriculum is FightCurriculum.DEFENSE
            and opponent_attacking
            and distance <= 110
        ):
            for action_id in GUARD_ACTION_IDS:
                if physical_mask[action_id]:
                    targeted[action_id] = True
            return targeted

        if (
            curriculum is FightCurriculum.DEFENSE
            and self.defense_counter_window_frames > 0
        ):
            for action_id in (CLOSE_C_ACTION_ID, CROUCH_B_ACTION_ID, ONIYAKI_ACTION_ID):
                if physical_mask[action_id]:
                    targeted[action_id] = True
            return targeted

        if (
            curriculum is FightCurriculum.ANTI_AIR
            and is_p2_airborne(self.previous_observation)
            and distance <= 130
        ):
            if physical_mask[ONIYAKI_ACTION_ID]:
                targeted[ONIYAKI_ACTION_ID] = True
            return targeted

        if curriculum is FightCurriculum.APPROACH and distance > 90:
            for action_id in (FORWARD_ACTION_ID, 5):
                if physical_mask[action_id]:
                    targeted[action_id] = True
            targeted[IDLE_ACTION_ID] = False
            if targeted.any():
                return targeted

        return physical_mask

    def _update_fight_teacher(
        self,
        action_id: int,
        action_accepted: bool,
        action_status: ActionStatus,
        previous_combo: int,
        current_combo: int,
        p1_damage: float,
    ) -> bool:
        """推進 fight teacher 的狀態機,回傳「本步是否完成整條路線」。
        重置條件:被打、連段掉了(combo 歸零)、起手揮空(進了 phase
        但從未形成 combo 且已回到閒置)—— 重置後 teacher 從頭再教。"""
        if not self.fight_guided:
            return False

        expected_action = self._fight_teacher_expected_action()
        if (
            expected_action is not None
            and action_accepted
            and action_id == expected_action
        ):
            self.fight_teacher_phase += 1

        if current_combo > 0:
            self.fight_teacher_chain_seen = True

        completed_now = False
        final_required_combo = self.combo_scenario.phases[-1].required_combo
        if (
            not self.fight_teacher_route_completed
            and self.fight_teacher_phase >= len(self.combo_scenario.phases)
            and current_combo >= final_required_combo
        ):
            self.fight_teacher_route_completed = True
            completed_now = True

        runtime_idle = (
            action_status.active_action_id < 0
            and action_status.queued_action_id < 0
        )
        chain_dropped = (
            self.fight_teacher_chain_seen
            and previous_combo > 0
            and current_combo == 0
        )
        opener_whiffed = (
            not self.fight_teacher_chain_seen
            and self.fight_teacher_phase > 0
            and current_combo == 0
            and runtime_idle
        )
        if p1_damage > 0.0 or chain_dropped or opener_whiffed:
            self.fight_teacher_phase = 0
            self.fight_teacher_chain_seen = False
            self.fight_teacher_route_completed = False

        return completed_now

    def _add_guided_distractors(
        self,
        mask: np.ndarray,
        physical_mask: np.ndarray,
    ) -> None:
        candidates = [
            action_id
            for action_id in range(1, self.action_space.n)
            if physical_mask[action_id] and not mask[action_id]
        ]
        if not candidates:
            return

        offset = self.combo_phase % len(candidates)
        ordered = candidates[offset:] + candidates[:offset]
        for action_id in ordered[:GUIDED_DISTRACTOR_COUNT]:
            mask[action_id] = True

    def action_masks(self) -> np.ndarray:
        """MaskablePPO 的入口。分派:
        FIGHT → 物理 mask(guided fight 環境再疊 teacher 引導);
        COMBO physical → 物理 mask;
        COMBO strict/guided → 正確招 + Idle(+ guided 的 2 個干擾招),
        距離太遠時逼迫前進(mask 只開 forward)。"""
        physical_mask = self._physical_action_mask()
        if self.training_profile is not TrainingProfile.COMBO:
            # Reverse-curriculum recipes simplify the starting state, not the
            # action space.  Keeping the full physical mask prevents the
            # teacher-side-channel that made the old Targeted scores fail to
            # transfer to unrestricted Fight environments.
            if self.level_recipe is not None:
                return physical_mask
            return self._targeted_fight_action_mask(physical_mask)
        if self.action_mask_level is ActionMaskLevel.PHYSICAL:
            return physical_mask

        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[IDLE_ACTION_ID] = True
        queueable_followup_phase = self._queueable_followup_phase()
        if queueable_followup_phase is not None:
            mask[queueable_followup_phase.action_id] = True
            if self.action_mask_level is ActionMaskLevel.GUIDED:
                self._add_guided_distractors(mask, physical_mask)
            return mask

        if not self._combo_input_ready():
            return mask

        if (
            self.combo_phase == 0
            and self.previous_observation is not None
            and abs(self.previous_observation.distance_x) > COMBO_CLOSE_DISTANCE
        ):
            mask[IDLE_ACTION_ID] = False
            mask[FORWARD_ACTION_ID] = True
            if self.action_mask_level is ActionMaskLevel.GUIDED:
                self._add_guided_distractors(mask, physical_mask)
            return mask

        phase = self._current_combo_phase()
        if phase is not None:
            mask[phase.action_id] = True
        if self.action_mask_level is ActionMaskLevel.GUIDED:
            self._add_guided_distractors(mask, physical_mask)
        return mask

    def close(self):
        self.client.close()

    def _update_attack_safety(
        self,
        action_id: int,
        current: Kof98Observation,
        p1_damage: float,
        p2_damage: float,
    ) -> tuple[float, dict[str, float]]:
        """出招風險狀態機(fight):出特殊技以上未命中 → 開 24 幀觀察窗。
        窗內被打 = 被確反(-4.0);窗滿沒事 → 距離近小罰(貼臉揮空)、
        距離遠小獎(安全施放)。一次只追蹤一筆,命中即銷案。"""
        info = {
            "pending": 0.0,
            "punished": 0.0,
            "unsafe_close": 0.0,
            "safe": 0.0,
        }
        reward = 0.0

        if self.pending_attack_risk is not None:
            self.pending_attack_risk["frames_left"] -= float(self.action_repeat)
            self.pending_attack_risk["p2_damage"] += p2_damage

            close_to_p2 = (
                current.p1_has_position
                and current.p2_has_position
                and abs(current.distance_x) < ATTACK_RISK_CLOSE_DISTANCE
            )
            attack_has_hit = self.pending_attack_risk["p2_damage"] > 0.0

            if attack_has_hit:
                self.pending_attack_risk = None
            elif p1_damage > 0.0:
                reward -= ATTACK_RISK_PUNISH_PENALTY
                info["punished"] = 1.0
                self.pending_attack_risk = None
            elif self.pending_attack_risk["frames_left"] <= 0.0:
                if close_to_p2:
                    reward -= ATTACK_RISK_UNSAFE_CLOSE_PENALTY
                    info["unsafe_close"] = 1.0
                else:
                    reward += ATTACK_RISK_SAFE_REWARD
                    info["safe"] = 1.0
                self.pending_attack_risk = None

        if (
            action_id in ATTACK_RISK_ACTION_IDS
            and p1_damage <= 0.0
            and p2_damage <= 0.0
            and self.pending_attack_risk is None
        ):
            self.pending_attack_risk = {
                "action_id": float(action_id),
                "frames_left": float(ATTACK_RISK_WINDOW_FRAMES),
                "p2_damage": 0.0,
            }

        info["pending"] = float(self.pending_attack_risk is not None)
        return reward, info

    def _reward(
        self,
        previous: Optional[Kof98Observation],
        current: Kof98Observation,
        action_id: int,
        damage_action_id: int,
        combo_hit_action_ids: set[int],
        p1_damage: float,
        p1_attack_overlap_edge: bool,
        p2_attack_overlap_edge: bool,
        p2_attack_pressure: bool,
        super_available: bool,
        p2_airborne: bool,
        frame_scale: float = 1.0,
    ) -> tuple[float, dict[str, float]]:
        """Fight 的核心 reward；版本由 self.fight_reward_version 決定。分項:

        hp:       傷害 ×2 / 受傷 ×2 —— 主收入,但係數壓低過(原 3.0),
                  否則散打的期望值會壓過連段。
        combo:    逐 hit 遞增表(hit2 +1 → hit5 +5,之後 cap)——
                  抵銷 KOF 傷害遞減,讓「延續連段」勝過「重置 neutral」。
        hitbox:   攻擊框初次壓上對方(上升沿一次性)。
        defense:  被壓制時防禦的小獎懲(per-step,乘 frame_scale)。
        anti_air: 鬼燒「實際打中」空中對手(damage_action_id 歸因,
                  不是本 step 按了鬼燒就給)。
        super:    無氣按超必的浪費懲罰。
        fast_win: KO 越快獎金越高(上限 15)。
        time:     微小時間成本,防止拖時間。
        frame_scale = action_repeat/6:per-step 項換算,讓「每模擬秒的
        累積量」不隨決策頻率改變;事件型項目(edge/一次性)不乘。
        """
        if self.fight_reward_version in (
            FightRewardVersion.SYMMETRIC_V2,
            FightRewardVersion.SYMMETRIC_TACTICAL_V3,
        ):
            return self._symmetric_fight_reward(previous, current)

        reward_parts = {
            "hp": 0.0,
            "hitbox": 0.0,
            "distance": 0.0,
            "defense": 0.0,
            "combo": 0.0,
            "super": 0.0,
            "anti_air": 0.0,
            "safety": 0.0,
            "fast_win": 0.0,
            "outcome": 0.0,
            "time": -0.001 * frame_scale,
        }
        if previous is None:
            return 0.0, reward_parts

        if previous.p2_health >= 0 and current.p2_health >= 0:
            p2_health_delta = float(previous.p2_health - current.p2_health)
            if damage_action_id != ONIYAKI_ACTION_ID or p2_airborne:
                reward_parts["hp"] += p2_health_delta * 2.0
            if (
                damage_action_id == ONIYAKI_ACTION_ID
                and p2_airborne
                and p2_health_delta > 0.0
            ):
                reward_parts["anti_air"] += ONIYAKI_ANTI_AIR_BONUS
        if previous.p1_health >= 0 and current.p1_health >= 0:
            reward_parts["hp"] -= float(previous.p1_health - current.p1_health) * 2.0

        # Edge events, not per-step accrual, so no frame_scale here.
        if p1_attack_overlap_edge:
            reward_parts["hitbox"] += P1_ATTACK_OVERLAP_REWARD
        if p2_attack_overlap_edge:
            reward_parts["hitbox"] -= P2_ATTACK_OVERLAP_PENALTY

        guard_action = action_id in GUARD_ACTION_IDS
        if p2_attack_pressure:
            if guard_action and p1_damage <= 0.0:
                reward_parts["defense"] += DEFENSE_GUARD_REWARD * frame_scale
            elif guard_action:
                reward_parts["defense"] -= DEFENSE_BAD_GUARD_PENALTY * frame_scale
            else:
                reward_parts["defense"] -= DEFENSE_UNGUARDED_PRESSURE_PENALTY * frame_scale

        if previous.p1_combo_count >= 0 and current.p1_combo_count >= 0:
            previous_combo = max(0, previous.p1_combo_count)
            current_combo = max(0, current.p1_combo_count)
            if current_combo > previous_combo:
                # A multi-hit step credits every newly reached hit number.
                for hit_number in range(previous_combo + 1, current_combo + 1):
                    if hit_number >= 2:
                        reward_parts["combo"] += FIGHT_COMBO_HIT_REWARDS.get(
                            hit_number,
                            FIGHT_COMBO_HIT_REWARD_CAP,
                        )
            if (
                combo_hit_action_ids.intersection(SUPER_ACTION_IDS)
                and current.p1_combo_count >= 2
            ):
                reward_parts["combo"] += SUPER_COMBO_BONUS

        if action_id in SUPER_ACTION_IDS and not super_available:
            reward_parts["super"] -= SUPER_NO_STOCK_PENALTY

        if previous.p2_health > 0 and 0 <= current.p2_health <= 0:
            remaining_time = max(0.0, min(99.0, float(current.round_time)))
            reward_parts["fast_win"] += FAST_WIN_BONUS_MAX * (remaining_time / 99.0)

        if current.p1_has_position and current.p2_has_position:
            distance = abs(current.distance_x)
            if EFFECTIVE_DISTANCE_MIN <= distance <= EFFECTIVE_DISTANCE_MAX:
                reward_parts["distance"] += DISTANCE_IN_RANGE_REWARD * frame_scale
            elif distance > DISTANCE_TOO_FAR_START:
                penalty_t = min(
                    1.0,
                    (distance - DISTANCE_TOO_FAR_START)
                    / max(1, DISTANCE_MAX_PENALTY_AT - DISTANCE_TOO_FAR_START),
                )
                reward_parts["distance"] -= DISTANCE_FAR_PENALTY * penalty_t * frame_scale

        reward = (
            reward_parts["hp"]
            + reward_parts["hitbox"]
            + reward_parts["distance"]
            + reward_parts["defense"]
            + reward_parts["combo"]
            + reward_parts["super"]
            + reward_parts["anti_air"]
            + reward_parts["safety"]
            + reward_parts["fast_win"]
            + reward_parts["outcome"]
            + reward_parts["time"]
        )
        return reward, reward_parts

    @staticmethod
    def _symmetric_fight_reward(
        previous: Optional[Kof98Observation],
        current: Kof98Observation,
    ) -> tuple[float, dict[str, float]]:
        """Full-fight objective: symmetric damage exchange plus round result.

        Tactical shaping belongs to targeted curricula. Keeping full Fight
        sparse and symmetric prevents distance, overlap, action-name and
        combo bonuses from teaching a shortcut that beats one scripted bot.
        """
        reward_parts = {
            "hp": 0.0,
            "hitbox": 0.0,
            "distance": 0.0,
            "defense": 0.0,
            "combo": 0.0,
            "super": 0.0,
            "anti_air": 0.0,
            "safety": 0.0,
            "fast_win": 0.0,
            "outcome": 0.0,
            "time": 0.0,
        }
        if previous is None:
            return 0.0, reward_parts

        damage_scale = SYMMETRIC_DAMAGE_ROUND_VALUE / 103.0
        if previous.p2_health >= 0 and current.p2_health >= 0:
            reward_parts["hp"] += (
                max(0, previous.p2_health - current.p2_health) * damage_scale
            )
        if previous.p1_health >= 0 and current.p1_health >= 0:
            reward_parts["hp"] -= (
                max(0, previous.p1_health - current.p1_health) * damage_scale
            )

        round_finished = (
            0 <= current.p1_health <= 0
            or 0 <= current.p2_health <= 0
            or current.round_time == 0
        )
        if round_finished:
            if current.p1_health > current.p2_health:
                reward_parts["outcome"] = SYMMETRIC_OUTCOME_REWARD
            elif current.p1_health < current.p2_health:
                reward_parts["outcome"] = -SYMMETRIC_OUTCOME_REWARD

        return reward_parts["hp"] + reward_parts["outcome"], reward_parts
