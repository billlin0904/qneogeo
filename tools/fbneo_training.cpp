#define KOF_ENV_BUILD
#include "fbneo_training.h"

#include <windows.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../gamememreadercore.h"
#include "../kof98movedata.h"
#include "libretro.h"

namespace {

static_assert(sizeof(kof_env_step_event_v1) == 28);
static_assert(sizeof(kof_env_step_events_v1) == 464);
static_assert(sizeof(kof_env_step_event_v2) == 40);
static_assert(sizeof(kof_env_step_events_v2) == 656);
static_assert(sizeof(kof_env_step_event_v3) == 68);
static_assert(sizeof(kof_env_step_events_v3) == 1104);
static_assert(sizeof(kof_env_step_event_v4) == 76);
static_assert(sizeof(kof_env_step_events_v4) == 1232);
static_assert(sizeof(kof_env_player_timing_state_v1) == 24);
static_assert(sizeof(kof_env_combat_timing_state_v1) == 80);
static_assert(sizeof(kof_env_step_event_v5) == 96);
static_assert(sizeof(kof_env_step_events_v5) == 3096);
static_assert(sizeof(kof_env_move_data_v1) == 76);

using retro_set_environment_t = void (*)(retro_environment_t);
using retro_set_video_refresh_t = void (*)(retro_video_refresh_t);
using retro_set_audio_sample_t = void (*)(retro_audio_sample_t);
using retro_set_audio_sample_batch_t = void (*)(retro_audio_sample_batch_t);
using retro_set_input_poll_t = void (*)(retro_input_poll_t);
using retro_set_input_state_t = void (*)(retro_input_state_t);
using retro_set_controller_port_device_t = void (*)(unsigned, unsigned);
using retro_init_t = void (*)();
using retro_deinit_t = void (*)();
using retro_reset_t = void (*)();
using retro_load_game_t = bool (*)(const retro_game_info *);
using retro_unload_game_t = void (*)();
using retro_run_t = void (*)();
using retro_get_system_info_t = void (*)(retro_system_info *);
using retro_get_system_av_info_t = void (*)(retro_system_av_info *);
using retro_serialize_size_t = size_t (*)();
using retro_serialize_t = bool (*)(void *, size_t);
using retro_unserialize_t = bool (*)(const void *, size_t);
using retro_get_memory_data_t = void *(*)(unsigned);
using retro_get_memory_size_t = size_t (*)(unsigned);

std::string utf8FromWide(const wchar_t *text) {
    if (!text || text[0] == L'\0')
        return {};

    const int size = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 1)
        return {};

    std::string result(static_cast<size_t>(size - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, result.data(), size, nullptr, nullptr);
    return result;
}

std::wstring absoluteWidePath(const wchar_t *path) {
    if (!path || path[0] == L'\0')
        return {};

    return std::filesystem::absolute(std::filesystem::path(path)).wstring();
}

std::string absoluteUtf8Path(const wchar_t *path) {
    const std::wstring wide = absoluteWidePath(path);
    return utf8FromWide(wide.c_str());
}

void logPrintf(retro_log_level level, const char *format, ...) {
    const char *prefix = "info";
    switch (level) {
    case RETRO_LOG_DEBUG:
        prefix = "debug";
        break;
    case RETRO_LOG_INFO:
        prefix = "info";
        break;
    case RETRO_LOG_WARN:
        prefix = "warn";
        break;
    case RETRO_LOG_ERROR:
        prefix = "error";
        break;
    }

    std::fprintf(stderr, "[fbneo-training:%s] ", prefix);
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
}

class FbneoTrainingRuntime;
FbneoTrainingRuntime *g_active_runtime = nullptr;

struct InputFrame {
    kof_env_joypad_state state {};
    int32_t frames = 0;
};

// P1 高階 Action 的完整生命週期。腳本播放位置、queue 狀態與事件歸因
// serial 必須一起重設，否則載入 state 後可能把新命中算到舊 Action。
struct ActionRuntimeState {
    std::vector<InputFrame> script;
    size_t script_index = 0;
    int32_t script_remaining_frames = 0;
    int32_t active_action_id = -1;
    int32_t queued_action_id = -1;
    int32_t last_started_action_id = -1;
    uint32_t last_started_action_serial = 0;
    int32_t last_started_action_age_frames = 0;
    bool action_start_event_pending = false;
    int32_t active_action_elapsed_frames = 0;
    bool last_action_accepted = false;
    uint32_t next_action_serial = 0;
    uint32_t active_action_serial = 0;

    void clearScript() {
        script.clear();
        script_index = 0;
        script_remaining_frames = 0;
    }

    int32_t scriptRemainingFrames() const {
        int32_t remaining = std::max(0, script_remaining_frames);
        for (size_t index = script_index + 1; index < script.size(); ++index)
            remaining += std::max(0, script[index].frames);
        return remaining;
    }

    void reset() {
        clearScript();
        active_action_id = -1;
        queued_action_id = -1;
        last_started_action_id = -1;
        last_started_action_serial = 0;
        last_started_action_age_frames = 0;
        action_start_event_pending = false;
        active_action_elapsed_frames = 0;
        last_action_accepted = false;
        active_action_serial = 0;
    }
};

struct DamageAttributionState {
    int32_t action_id = -1;
    uint32_t action_serial = 0;
    int32_t age_frames = 0;
    int32_t target_y = -1;
    bool target_airborne_before = false;
    bool target_airborne_after = false;

    void reset() {
        *this = {};
        action_id = -1;
        target_y = -1;
    }
};

struct ActionHandoffState {
    int32_t action_id = -1;
    uint32_t action_serial = 0;
    int32_t age_frames = 0;

    void reset() {
        *this = {};
        action_id = -1;
    }
};

enum class GuardReactionPhase {
    Idle,
    WaitingForCountdown,
    CountdownActive,
    RefreshPending,
};

// D2:D3/E3 不是全域 blockstun enum；只有先看到真實 BLOCK_CONTACT，
// 才追蹤後續倒數。候選值還必須出現 N→N-1，避免把上一個動作留下的
// 503、11 等 stale counter 誤認成這次防禦硬直。
struct GuardReactionState {
    int32_t attacker_action_id = -1;
    uint32_t attacker_action_serial = 0;
    uint32_t reaction_serial = 0;
    int32_t age_frames = 0;
    int32_t counter_at_contact = -1;
    int32_t candidate_counter = -1;
    uint64_t contact_frame = 0;
    uint64_t not_before_frame = 0;
    uint64_t candidate_counter_frame = 0;
    GuardReactionPhase phase = GuardReactionPhase::Idle;
    bool active = false;
    bool confirmed_block_contact = false;
    bool countdown_loaded = false;
    bool refresh_pending = false;
    bool start_emitted = false;
    bool end_emitted = false;
    bool manual_guard_hold_active = false;
    bool manual_success_emitted = false;

    void begin(int32_t action_id,
               uint32_t action_serial,
               uint32_t new_reaction_serial,
               uint64_t frame,
               int32_t hit_guard_stop,
               int32_t reaction_counter,
               bool manual_guard) {
        reset();
        attacker_action_id = action_id;
        attacker_action_serial = action_serial;
        reaction_serial = new_reaction_serial;
        counter_at_contact = reaction_counter;
        contact_frame = frame;
        not_before_frame =
            frame + static_cast<uint64_t>(std::max(1, hit_guard_stop));
        phase = GuardReactionPhase::WaitingForCountdown;
        active = true;
        confirmed_block_contact = true;
        manual_guard_hold_active = manual_guard;
    }

    void refresh(int32_t action_id,
                 uint32_t action_serial,
                 uint64_t frame,
                 int32_t hit_guard_stop,
                 int32_t reaction_counter,
                 bool manual_guard) {
        attacker_action_id = action_id;
        attacker_action_serial = action_serial;
        age_frames = 0;
        counter_at_contact = reaction_counter;
        contact_frame = frame;
        not_before_frame =
            frame + static_cast<uint64_t>(std::max(1, hit_guard_stop));
        candidate_counter = -1;
        candidate_counter_frame = 0;
        refresh_pending = true;
        phase = GuardReactionPhase::RefreshPending;
        manual_guard_hold_active =
            manual_guard_hold_active || manual_guard;
    }

    void reset() {
        *this = {};
        attacker_action_id = -1;
        counter_at_contact = -1;
        candidate_counter = -1;
        phase = GuardReactionPhase::Idle;
    }
};

// StrategyV4 對稱的受擊/防禦反應快照。這個 tracker 只提供可觀察
// timing state，不負責判定 Reward；真正的防禦事件仍由有 contact 證據
// 的 GuardReactionState 處理。E3 只負責分類，D2:D3 只在已鎖定的反應
// 中提供剩餘子階段幀數。
struct ReactionTimingState {
    int32_t kind = KOF_ENV_REACTION_NONE;
    int32_t remaining = 0;
    int32_t age_frames = 0;
    bool active = false;
    bool countdown_seen = false;

    void beginOrRefresh(int32_t new_kind) {
        if (new_kind != KOF_ENV_REACTION_GUARD &&
            new_kind != KOF_ENV_REACTION_HIT) {
            return;
        }

        // 每次有新的實際接觸證據，都必須等待該次反應自己的 N->N-1；
        // 不沿用前一段命中或多段格擋留下的 remaining。
        remaining = 0;
        countdown_seen = false;
        age_frames = 0;
        kind = new_kind;
        active = true;
    }

    void reset() {
        *this = {};
        kind = KOF_ENV_REACTION_NONE;
    }
};

// 一個 Python step 內的事件批次，以及跨 frame 才能完成的命中歸因狀態。
struct CombatEventState {
    kof_env_step_events_v1 v1 {};
    kof_env_step_events_v2 v2 {};
    kof_env_step_events_v3 v3 {};
    kof_env_step_events_v4 v4 {};
    kof_env_step_events_v5 v5 {};
    DamageAttributionState recent_damage;
    ActionHandoffState handoff;
    std::array<GuardReactionState, 2> guard_reactions;
    std::array<ReactionTimingState, 2> reactions;
    std::array<bool, 2> block_contact_active {};
    uint32_t event_epoch = 1;
    uint32_t next_guard_reaction_serial = 0;

    void clearBatch() {
        v1 = {};
        v1.struct_size = sizeof(kof_env_step_events_v1);
        v1.version = KOF_ENV_STEP_EVENTS_VERSION_1;
        v2 = {};
        v2.struct_size = sizeof(kof_env_step_events_v2);
        v2.version = KOF_ENV_STEP_EVENTS_VERSION_2;
        v3 = {};
        v3.struct_size = sizeof(kof_env_step_events_v3);
        v3.version = KOF_ENV_STEP_EVENTS_VERSION_3;
        v4 = {};
        v4.struct_size = sizeof(kof_env_step_events_v4);
        v4.version = KOF_ENV_STEP_EVENTS_VERSION_4;
        v5 = {};
        v5.struct_size = sizeof(kof_env_step_events_v5);
        v5.version = KOF_ENV_STEP_EVENTS_VERSION_5;
        v5.batch_event_epoch = event_epoch;
    }

    void resetTracking() {
        recent_damage.reset();
        handoff.reset();
        for (auto &reaction : guard_reactions)
            reaction.reset();
        for (auto &reaction : reactions)
            reaction.reset();
        block_contact_active.fill(false);
    }

    uint32_t allocateGuardReactionSerial() {
        ++next_guard_reaction_serial;
        if (next_guard_reaction_serial == 0)
            ++next_guard_reaction_serial;
        return next_guard_reaction_serial;
    }

    void advanceEpoch() {
        ++event_epoch;
        if (event_epoch == 0)
            ++event_epoch;
        next_guard_reaction_serial = 0;
        resetTracking();
        // clearBatch() 只在 step 開始時執行；reset/load 或 step 中途換局
        // 仍必須讓批次 epoch 與 timing state 立即一致。
        v5.batch_event_epoch = event_epoch;
    }

    void removeV5HitEventsForBlock(int32_t frame_offset,
                                   int32_t source_player,
                                   int32_t target_player) {
        uint32_t write_index = 0;
        for (uint32_t read_index = 0;
             read_index < v5.event_count;
             ++read_index) {
            const auto &event = v5.events[read_index];
            const bool same_contact =
                event.frame_offset == frame_offset &&
                event.source_player == source_player &&
                event.target_player == target_player;
            const bool provisional_hit =
                event.event_type == KOF_ENV_STEP_EVENT_COMBO_HIT ||
                event.event_type == KOF_ENV_STEP_EVENT_DAMAGE_ONLY ||
                event.event_type == KOF_ENV_STEP_EVENT_CLEAN_HIT;
            if (same_contact && provisional_hit)
                continue;

            if (write_index != read_index)
                v5.events[write_index] = event;
            ++write_index;
        }
        v5.event_count = write_index;
    }

    void rollbackFrame(int32_t frame_offset) {
        const auto rollback = [frame_offset](auto &batch) {
            while (batch.event_count > 0 &&
                   batch.events[batch.event_count - 1].frame_offset ==
                       frame_offset) {
                --batch.event_count;
            }
        };
        rollback(v1);
        rollback(v2);
        rollback(v3);
        rollback(v4);
        rollback(v5);
    }
};

enum class P2ControlMode {
    Disabled,
    ScriptedStyle,
    ActionApi,
};

// P2 的風格腳本與公開 Action API 共用同一組輸出腳本，但控制模式互斥。
// 單一 enum 可避免 random_ai_enabled/action_ai_enabled 同時為 true。
struct P2ControllerState {
    std::vector<InputFrame> script;
    size_t script_index = 0;
    int32_t script_remaining_frames = 0;
    int32_t training_action_id = -1;
    int32_t training_action_elapsed_frames = 0;
    int32_t last_started_action_id = -1;
    uint32_t next_action_serial = 0;
    uint32_t last_started_action_serial = 0;
    int32_t last_started_action_age_frames = 0;
    bool action_start_event_pending = false;
    int32_t active_action_id = -1;
    int32_t queued_action_id = -1;
    int32_t active_action_elapsed_frames = 0;
    bool last_action_accepted = false;
    P2ControlMode mode = P2ControlMode::Disabled;
    kof_env_p2_style style = KOF_ENV_P2_STYLE_ONIYAKI;
    uint32_t training_cycle = 0;

    bool scriptedStyleEnabled() const {
        return mode == P2ControlMode::ScriptedStyle;
    }

    bool actionApiEnabled() const {
        return mode == P2ControlMode::ActionApi;
    }

    void clearScript() {
        script.clear();
        script_index = 0;
        script_remaining_frames = 0;
        training_action_id = -1;
        training_action_elapsed_frames = 0;
    }

    int32_t scriptRemainingFrames() const {
        int32_t remaining = std::max(0, script_remaining_frames);
        for (size_t index = script_index + 1; index < script.size(); ++index)
            remaining += std::max(0, script[index].frames);
        return remaining;
    }

    void resetActions() {
        clearScript();
        active_action_id = -1;
        queued_action_id = -1;
        active_action_elapsed_frames = 0;
        last_action_accepted = false;
        last_started_action_id = -1;
        last_started_action_serial = 0;
        last_started_action_age_frames = 0;
        action_start_event_pending = false;
    }
};

struct RuntimePaths {
    std::string game;
    std::string system_directory;
    std::string save_directory;
};

struct CachedState {
    std::wstring path;
    std::vector<uint8_t> data;

    void clear() {
        path.clear();
        data.clear();
    }
};

struct FollowUpRule {
    int32_t parent_action_id = -1;
    int32_t child_action_id = -1;
    int32_t queue_open_frame = 0;
    int32_t queue_close_frame_exclusive = 0;
    int32_t execute_frame = 0;
    int32_t script_action_id = -1;
};

struct FrameCombatState {
    // 單一 emulator frame 的戰鬥快照。
    //
    // runFrames() 會在 retro_run() 前後各讀取一次，事件偵測只比較同一
    // frame 的 before/after。不能改用 Python step 的首尾值，因為一個
    // step 通常跨越 4 個以上 frame，會漏掉中間短暫出現的命中、格擋、
    // hitstop 或 blockstun 邊緣。

    // P1/P2 當前血量。預設 -1 表示 RAM 讀取失敗；before > after 代表
    // 該 frame 受到傷害。
    int32_t p1_health = -1;
    int32_t p2_health = -1;

    // 雙方當前連段數。before < after 用來辨識對應玩家的連段命中。
    int32_t p1_combo = -1;
    int32_t p2_combo = -1;

    // P1 Guard Crush 耐久值，位址 player base + 0x147
    // (P1: 0x108247)。數值在格擋時下降，可作為真實 block contact 的
    // 證據之一；它不是 blockstun timer。
    int32_t p1_guard_crush_value = -1;
    int32_t p2_guard_crush_value = -1;

    // Hit/Guard Stop 原始 byte，位址 player base + 0x125
    // (P1: 0x108225、P2: 0x108425)。保留 before/after 值寫入 V3 event，
    // 供除錯與驗證 hitstop；目前不單獨拿它判斷格擋成功。
    int32_t p1_hit_guard_stop_raw = -1;
    int32_t p2_hit_guard_stop_raw = -1;

    // P1 反應子階段倒數，位址 player base + 0xD2:D3
    // (P1: 0x1081D2)。Motorola 68000 採 big-endian，所以必須以 signed
    // 16-bit 解讀；例如 00:10=16、FF:FF=-1。它只在已確認的地面防禦
    // tracker 內解讀，不能全域視為 blockstun timer。
    int32_t p1_reaction_d2_raw = -1;
    int32_t p1_reaction_counter = -1;
    int32_t p2_reaction_d2_raw = -1;
    int32_t p2_reaction_counter = -1;

    // P1 狀態 flags 原始 byte，位址 player base + 0xE3
    // (P1: 0x1081E3)。bit 0x20 在受制狀態期間成立；它不是獨立的
    // blockstun timer，因此必須和已確認的 block contact 及 D2 邊緣
    // 一起使用。
    int32_t p1_reaction_e3_raw = -1;
    int32_t p2_reaction_e3_raw = -1;

    // 角色世界座標。事件需要它判斷左右朝向、P1 是否真的按住「後」，
    // 以及目標在受擊前後是否位於空中。
    game_memory::Point p1_position;
    game_memory::Point p2_position;

    // 對應座標是否成功讀取。座標不可用時不能把預設 Point 當成真值。
    bool has_p1_position = false;
    bool has_p2_position = false;

    // 該 frame 的 P2 Attack/ProjectileAttack box 是否與 P1 Guard box
    // 重疊。這是物理格擋接觸候選，和 Attack × Vulnerability 的命中
    // overlap 不同；仍需搭配 P2 攻擊中、Guard Crush 變化及防禦輸入。
    bool p2_attack_guard_overlap = false;
    bool p1_attack_guard_overlap = false;
};

constexpr size_t P1_PORT = 0;
constexpr size_t P2_PORT = 1;
constexpr size_t PLAYER_PORT_COUNT = 2;
constexpr int32_t COMBO_EVENT_DAMAGE_GRACE_FRAMES = 2;
// age 0、1、2 共涵蓋三個取樣 frame。只用於 queued child 已開始輸入，
// 但畫面上仍是 parent 招式延遲命中的情況。
constexpr int32_t ACTION_HANDOFF_HIT_GRACE_FRAMES = 2;
// Action 腳本結束後，判定可能延遲數幀才造成傷害；fallback 只在有限
// 時間內保留歸因，避免很久以後的飛行道具命中被算到上一個普通技。
constexpr int32_t ACTION_ATTRIBUTION_FALLBACK_FRAMES = 240;
// D2/E3 追蹤只應涵蓋同一段格擋反應。若候選邊緣不完整，超時後丟棄，
// 避免下一次無關的 E3 falling edge 被誤認為這次 blockstun 結束。
constexpr int32_t GUARD_REACTION_TIMEOUT_FRAMES = 240;

constexpr bool shouldAbortUnexpectedReactionCounter(
    bool countdown_seen,
    int32_t counter_after,
    bool reaction_ended) {
    return countdown_seen && counter_after < 0 && !reaction_ended;
}

static_assert(shouldAbortUnexpectedReactionCounter(true, -2, false));
static_assert(shouldAbortUnexpectedReactionCounter(true, -1, false));
static_assert(!shouldAbortUnexpectedReactionCounter(true, -1, true));
static_assert(!shouldAbortUnexpectedReactionCounter(true, 0, false));
constexpr int32_t KOF98_AIRBORNE_Y_THRESHOLD = 185;
constexpr int32_t WALK_FORWARD_ACTION_ID = 1;
constexpr int32_t WALK_BACK_ACTION_ID = 2;
constexpr int32_t CROUCH_GUARD_ACTION_ID = 3;
constexpr int32_t STAND_GUARD_ACTION_ID = 4;
constexpr int32_t STAND_A_ACTION_ID = 6;
constexpr int32_t STAND_B_ACTION_ID = 7;
constexpr int32_t KYO_CROUCH_A_ACTION_ID = 10;
constexpr int32_t KYO_CROUCH_B_ACTION_ID = 11;
constexpr int32_t KYO_CLOSE_C_ACTION_ID = 8;
constexpr int32_t KYO_ARAGAMI_ACTION_ID = 14;
constexpr int32_t KYO_KOTOTSUKI_YOU_ACTION_ID = 15;
constexpr int32_t KYO_ONIYAKI_ACTION_ID = 16;
constexpr int32_t KYO_RED_KICK_ACTION_ID = 17;
constexpr int32_t KYO_OROCHINAGI_ACTION_ID = 18;
constexpr int32_t KYO_MUSHIKI_ACTION_ID = 19;
constexpr int32_t KYO_JUMP_C_ACTION_ID = 20;
constexpr int32_t KYO_JUMP_D_ACTION_ID = 21;
constexpr int32_t KYO_FORWARD_B_ACTION_ID = 22;
constexpr int32_t KYO_POISON_BITE_ACTION_ID = 23;
constexpr int32_t IDLE_ACTION_ID = 0;
constexpr int32_t KYO_TSUMI_YOMI_ACTION_ID = 24;
constexpr int32_t KYO_BATSU_YOMI_ACTION_ID = 25;
constexpr int32_t KYO_SEVENTY_FIVE_SHIKI_KAI_ACTION_ID = 26;
constexpr int32_t KYO_YANO_SABI_ACTION_ID = 27;
constexpr int32_t KYO_MIGIRI_UGACHI_ACTION_ID = 28;
constexpr int32_t KYO_SEVENTY_FIVE_SHIKI_KAI_COMBO_SCRIPT_ID = 1000;
constexpr int32_t KYO_SEVENTY_FIVE_SHIKI_KAI_RED_KICK_SCRIPT_ID = 1001;
constexpr int32_t KYO_CROUCH_A_MUSHIKI_BUFFER_SCRIPT_ID = 1002;
constexpr int32_t KYO_MUSHIKI_FINISH_SCRIPT_ID = 1003;
constexpr int32_t P1_HOLD_ACTION_FIRST = 1;
constexpr int32_t P1_HOLD_ACTION_LAST = 4;
constexpr int32_t KYO_CLOSE_C_SEVENTY_FIVE_SHIKI_KAI_TRIGGER_FRAME = 5;
constexpr int32_t KYO_FORWARD_B_FOLLOW_UP_TRIGGER_FRAME = 37;
constexpr int32_t KYO_FORWARD_B_OROCHINAGI_TRIGGER_FRAME = 19;
constexpr int32_t KYO_KOTOTSUKI_YOU_BUFFER_TRIGGER_FRAME = 24;
constexpr int32_t KYO_BATSU_YOMI_TRIGGER_FRAME = 34;
constexpr int32_t KYO_SEVENTY_FIVE_SHIKI_KAI_RECOVERY_FRAMES = 58;
constexpr int32_t KYO_SEVENTY_FIVE_SHIKI_KAI_OROCHINAGI_TRIGGER_FRAME = 63;
constexpr int32_t KYO_SEVENTY_FIVE_SHIKI_KAI_RED_KICK_TRIGGER_FRAME = 75;
constexpr int32_t KYO_SEVENTY_FIVE_SHIKI_KAI_KOTOTSUKI_TRIGGER_FRAME = 75;
constexpr int32_t KYO_SEVENTY_FIVE_SHIKI_KAI_ARAGAMI_TRIGGER_FRAME = 85;
constexpr int32_t KYO_ARAGAMI_YANO_SABI_TRIGGER_FRAME = 10;
constexpr int32_t KYO_YANO_SABI_MIGIRI_UGACHI_TRIGGER_FRAME = 29;
constexpr int32_t KYO_CROUCH_B_CROUCH_A_TRIGGER_FRAME = 17;
constexpr int32_t KYO_CROUCH_A_MUSHIKI_TRIGGER_FRAME = 9;

bool hitboxRectsOverlap(const game_memory::HitboxRect &a,
                        const game_memory::HitboxRect &b) {
    return a.left < b.left + b.width &&
           a.left + a.width > b.left &&
           a.top < b.top + b.height &&
           a.top + a.height > b.top;
}

bool hasAttackGuardboxOverlap(const game_memory::HitboxOverlay &overlay,
                              int32_t attack_owner,
                              int32_t target_owner) {
    for (const game_memory::HitboxRect &attack : overlay.boxes) {
        if (attack.owner != attack_owner ||
            (attack.type != game_memory::HitboxAttack &&
             attack.type != game_memory::HitboxProjectileAttack)) {
            continue;
        }

        for (const game_memory::HitboxRect &guardbox : overlay.boxes) {
            if (guardbox.owner != target_owner ||
                guardbox.type != game_memory::HitboxGuard) {
                continue;
            }
            if (hitboxRectsOverlap(attack, guardbox))
                return true;
        }
    }
    return false;
}

enum class CharacterID {
    Kyo,
};

using CharacterActionTable = std::map<int32_t, std::vector<InputFrame>>;
using FollowUpRuleTable = std::vector<FollowUpRule>;

void setForwardOn(kof_env_joypad_state &input, bool forward_is_right) {
    if (forward_is_right)
        input.right = 1;
    else
        input.left = 1;
}

void setBackOn(kof_env_joypad_state &input, bool forward_is_right) {
    if (forward_is_right)
        input.left = 1;
    else
        input.right = 1;
}

std::vector<InputFrame> simpleAction(const kof_env_joypad_state &input) {
    return {
        { input, 4 },
        { {}, 2 },
    };
}

bool isPublicActionId(int32_t action_id) {
    return action_id >= 0 && action_id < KOF_ENV_PUBLIC_ACTION_COUNT;
}

bool isP1HoldActionId(int32_t action_id) {
    return action_id >= P1_HOLD_ACTION_FIRST && action_id <= P1_HOLD_ACTION_LAST;
}

std::vector<InputFrame> p1ActionScript(
    int32_t action_id,
    const std::vector<InputFrame> &script) {
    if (!isP1HoldActionId(action_id))
        return script;

    // 共用 LUT 保持原樣，避免改變 P2 腳本節奏。P1 的移動／防禦只送出
    // 一個 decision-sized chunk，下一個 repeat4 決策便能繼續保持、放開，
    // 或立刻切換成攻擊，不必等待額外的 neutral 尾幀。
    if (script.size() == 2 &&
        script[0].frames == KOF_ENV_P1_HOLD_CHUNK_FRAMES &&
        script[1].frames == 2) {
        return { script[0] };
    }

    return script;
}

CharacterActionTable buildCharacterActions(bool forward_is_right) {
    auto forward = [forward_is_right] {
        kof_env_joypad_state input {};
        setForwardOn(input, forward_is_right);
        return input;
    };
    auto back = [forward_is_right] {
        kof_env_joypad_state input {};
        setBackOn(input, forward_is_right);
        return input;
    };
    auto down = [] {
        kof_env_joypad_state input {};
        input.down = 1;
        return input;
    };
    auto up = [] {
        kof_env_joypad_state input {};
        input.up = 1;
        return input;
    };
    auto up_forward = [forward_is_right] {
        kof_env_joypad_state input {};
        input.up = 1;
        setForwardOn(input, forward_is_right);
        return input;
    };
    auto down_forward = [forward_is_right] {
        kof_env_joypad_state input {};
        input.down = 1;
        setForwardOn(input, forward_is_right);
        return input;
    };
    auto down_back = [forward_is_right] {
        kof_env_joypad_state input {};
        input.down = 1;
        setBackOn(input, forward_is_right);
        return input;
    };

    kof_env_joypad_state crouch_back = down_back();
    kof_env_joypad_state neutral_a {};
    neutral_a.a = 1;
    kof_env_joypad_state neutral_b {};
    neutral_b.b = 1;
    kof_env_joypad_state neutral_c {};
    neutral_c.c = 1;
    kof_env_joypad_state neutral_d {};
    neutral_d.d = 1;
    kof_env_joypad_state forward_d = forward();
    forward_d.d = 1;
    kof_env_joypad_state crouch_a = down();
    crouch_a.a = 1;
    kof_env_joypad_state crouch_b = down();
    crouch_b.b = 1;
    kof_env_joypad_state crouch_c = down();
    crouch_c.c = 1;
    kof_env_joypad_state crouch_d = down();
    crouch_d.d = 1;

    kof_env_joypad_state qcf_a = forward();
    qcf_a.a = 1;
    kof_env_joypad_state poison_bite = forward();
    poison_bite.c = 1;
    kof_env_joypad_state zai_ei = back();
    zai_ei.c = 1;
    kof_env_joypad_state zai_ei_transition_up = up();
    zai_ei_transition_up.c = 1;
    kof_env_joypad_state zai_ei_transition_up_forward = up_forward();
    zai_ei_transition_up_forward.c = 1;
    kof_env_joypad_state hcb_d = back();
    hcb_d.d = 1;
    kof_env_joypad_state hcb_a = back();
    hcb_a.a = 1;
    kof_env_joypad_state dp_a = down_forward();
    dp_a.a = 1;
    kof_env_joypad_state red_kick_b = down_back();
    red_kick_b.b = 1;
    kof_env_joypad_state back_b = back();
    back_b.b = 1;
    kof_env_joypad_state super_a = forward();
    super_a.a = 1;
    kof_env_joypad_state orochinagi_c = forward();
    orochinagi_c.c = 1;
    kof_env_joypad_state jump_forward_c = forward();
    jump_forward_c.c = 1;
    kof_env_joypad_state jump_forward_d = forward();
    jump_forward_d.d = 1;
    kof_env_joypad_state forward_b = forward();
    forward_b.b = 1;
    kof_env_joypad_state zai_ei_followup = forward();
    zai_ei_followup.c = 1;

    CharacterActionTable kyo {
        { 0, { { {}, 1 } } },
        { 1, simpleAction(forward()) },
        { 2, simpleAction(back()) },
        { 3, simpleAction(crouch_back) },
        { 4, simpleAction(back()) },
        { 5, simpleAction(up_forward()) },
        { 6, simpleAction(neutral_a) },
        { 7, simpleAction(neutral_b) },
        { 8, simpleAction(neutral_c) },
        { 9, simpleAction(neutral_d) },
        { 10, {
            { crouch_a, 5 },
            { down(), 1 },
            { {}, 2 },
        } },
        { 11, {
            { crouch_b, 5 },
            { down(), 12 },
            { {}, 2 },
        } },
        { 12, simpleAction(crouch_c) },
        { 13, simpleAction(crouch_d) },
        { 14, {
            { down(), 2 },
            { down_forward(), 2 },
            { qcf_a, 4 },
            { {}, 4 },
        } },
        { 15, {
            { {}, 9 },
            { forward(), 2 },
            { down_forward(), 2 },
            { down(), 2 },
            { down_back(), 2 },
            { hcb_d, 4 },
            { {}, 12 },
        } },
        { 16, {
            { forward(), 2 },
            { down(), 2 },
            { dp_a, 4 },
            { {}, 8 },
        } },
        { 17, {
            { back(), 2 },
            { down(), 2 },
            { red_kick_b, 4 },
            { {}, 8 },
        } },
        { 18, {
            { down(), 2 },
            { down_back(), 4 },
            { back(), 8 },
            { down_back(), 3 },
            { down(), 1 },
            { down_forward(), 4 },
            { forward(), 1 },
            { orochinagi_c, 1 },
            { neutral_c, 29 },
            { {}, 12 },
        } },
        { 19, {
            { down(), 2 },
            { down_forward(), 2 },
            { forward(), 2 },
            { down(), 2 },
            { down_forward(), 2 },
            { super_a, 5 },
            { {}, 12 },
        } },
        { 20, {
            { up_forward(), 2 },
            { forward(), 12 },
            { jump_forward_c, 5 },
            { {}, 18 },
        } },
        { 21, {
            { up_forward(), 2 },
            { forward(), 12 },
            { jump_forward_d, 5 },
            { {}, 10 },
        } },
        { 22, {
            { forward_b, 5 },
            { {}, 12 },
        } },
        { KYO_POISON_BITE_ACTION_ID, {
            { down(), 2 },
            { down_forward(), 2 },
            { poison_bite, 4 },
            { {}, 8 },
        } },
        { 24, {
            { forward(), 2 },
            { down_forward(), 2 },
            { down(), 2 },
            { down_back(), 2 },
            { back(), 1 },
            { zai_ei, 2 },
            { zai_ei_transition_up, 1 },
            { zai_ei_transition_up_forward, 4 },
            { zai_ei_followup, 2 },
            { forward(), KYO_BATSU_YOMI_TRIGGER_FRAME - 18 },
        } },
        { 25, {
            { zai_ei_followup, 13 },
            { {}, 8 },
        } },
        { KYO_SEVENTY_FIVE_SHIKI_KAI_ACTION_ID, {
            { down(), 7 },
            { down_forward(), 6 },
            { forward(), 4 },
            { {}, 2 },
            { neutral_d, 8 },
            { {}, 22 },
            { neutral_d, 13 },
            { {}, KYO_SEVENTY_FIVE_SHIKI_KAI_RECOVERY_FRAMES },
        } },
        { KYO_YANO_SABI_ACTION_ID, {
            { forward(), 2 },
            { down_forward(), 2 },
            { down(), 2 },
            { down_back(), 2 },
            { hcb_a, 4 },
            { {}, 12 },
        } },
        { KYO_MIGIRI_UGACHI_ACTION_ID, {
            { neutral_c, 4 },
            { {}, 8 },
        } },
        { KYO_CROUCH_A_MUSHIKI_BUFFER_SCRIPT_ID, {
            { crouch_a, 5 },
            { down(), 1 },
            { down_forward(), 2 },
            { forward(), 1 },
            { {}, 12 },
        } },
        { KYO_MUSHIKI_FINISH_SCRIPT_ID, {
            { down(), 1 },
            { down_forward(), 2 },
            { super_a, 5 },
            { {}, 12 },
        } },
        { KYO_SEVENTY_FIVE_SHIKI_KAI_COMBO_SCRIPT_ID, {
            { down(), 2 },
            { down_forward(), 2 },
            { forward_d, 4 },
            { {}, 18 },
            { neutral_d, 8 },
            { {}, 28 },
        } },
        { KYO_SEVENTY_FIVE_SHIKI_KAI_RED_KICK_SCRIPT_ID, {
            { back(), 6 },
            { down_back(), 2 },
            { down(), 4 },
            { red_kick_b, 1 },
            { back_b, 2 },
            { {}, 8 },
        } },
    };

    return kyo;
}


class CharacterActionMapLut {
public:
    CharacterActionMapLut() {
        lut_.insert(std::make_pair(
            CharacterID::Kyo,
            std::make_pair(buildCharacterActions(true), buildCharacterActions(false))));
        // Forward B's second hit can cancel into these queued follow-up actions.
        follow_up_lut_.insert(std::make_pair(
            CharacterID::Kyo,
            FollowUpRuleTable {
                {
                    KYO_CROUCH_B_ACTION_ID,
                    KYO_CROUCH_A_ACTION_ID,
                    0,
                    KYO_CROUCH_B_CROUCH_A_TRIGGER_FRAME,
                    KYO_CROUCH_B_CROUCH_A_TRIGGER_FRAME,
                    KYO_CROUCH_A_MUSHIKI_BUFFER_SCRIPT_ID,
                },
                {
                    KYO_CROUCH_A_ACTION_ID,
                    KYO_MUSHIKI_ACTION_ID,
                    0,
                    KYO_CROUCH_A_MUSHIKI_TRIGGER_FRAME,
                    KYO_CROUCH_A_MUSHIKI_TRIGGER_FRAME,
                    KYO_MUSHIKI_FINISH_SCRIPT_ID,
                },
                {
                    KYO_CLOSE_C_ACTION_ID,
                    KYO_SEVENTY_FIVE_SHIKI_KAI_ACTION_ID,
                    0,
                    KYO_CLOSE_C_SEVENTY_FIVE_SHIKI_KAI_TRIGGER_FRAME,
                    KYO_CLOSE_C_SEVENTY_FIVE_SHIKI_KAI_TRIGGER_FRAME,
                    KYO_SEVENTY_FIVE_SHIKI_KAI_COMBO_SCRIPT_ID,
                },
                {
                    KYO_FORWARD_B_ACTION_ID,
                    KYO_ARAGAMI_ACTION_ID,
                    0,
                    KYO_FORWARD_B_FOLLOW_UP_TRIGGER_FRAME,
                    KYO_FORWARD_B_FOLLOW_UP_TRIGGER_FRAME,
                },
                {
                    KYO_FORWARD_B_ACTION_ID,
                    KYO_ONIYAKI_ACTION_ID,
                    0,
                    KYO_FORWARD_B_FOLLOW_UP_TRIGGER_FRAME,
                    KYO_FORWARD_B_FOLLOW_UP_TRIGGER_FRAME,
                },
                {
                    KYO_FORWARD_B_ACTION_ID,
                    KYO_KOTOTSUKI_YOU_ACTION_ID,
                    0,
                    KYO_KOTOTSUKI_YOU_BUFFER_TRIGGER_FRAME,
                    KYO_KOTOTSUKI_YOU_BUFFER_TRIGGER_FRAME,
                },
                {
                    KYO_FORWARD_B_ACTION_ID,
                    KYO_OROCHINAGI_ACTION_ID,
                    0,
                    KYO_FORWARD_B_OROCHINAGI_TRIGGER_FRAME,
                    KYO_FORWARD_B_OROCHINAGI_TRIGGER_FRAME,
                },
                {
                    KYO_FORWARD_B_ACTION_ID,
                    KYO_RED_KICK_ACTION_ID,
                    0,
                    KYO_FORWARD_B_FOLLOW_UP_TRIGGER_FRAME,
                    KYO_FORWARD_B_FOLLOW_UP_TRIGGER_FRAME,
                },
                {
                    KYO_FORWARD_B_ACTION_ID,
                    KYO_POISON_BITE_ACTION_ID,
                    0,
                    KYO_FORWARD_B_FOLLOW_UP_TRIGGER_FRAME,
                    KYO_FORWARD_B_FOLLOW_UP_TRIGGER_FRAME,
                },
                {
                    KYO_TSUMI_YOMI_ACTION_ID,
                    KYO_BATSU_YOMI_ACTION_ID,
                    0,
                    KYO_BATSU_YOMI_TRIGGER_FRAME,
                    KYO_BATSU_YOMI_TRIGGER_FRAME,
                },
                {
                    KYO_SEVENTY_FIVE_SHIKI_KAI_ACTION_ID,
                    KYO_OROCHINAGI_ACTION_ID,
                    0,
                    KYO_SEVENTY_FIVE_SHIKI_KAI_OROCHINAGI_TRIGGER_FRAME,
                    KYO_SEVENTY_FIVE_SHIKI_KAI_OROCHINAGI_TRIGGER_FRAME,
                },
                {
                    KYO_SEVENTY_FIVE_SHIKI_KAI_ACTION_ID,
                    KYO_RED_KICK_ACTION_ID,
                    0,
                    KYO_SEVENTY_FIVE_SHIKI_KAI_RED_KICK_TRIGGER_FRAME,
                    KYO_SEVENTY_FIVE_SHIKI_KAI_RED_KICK_TRIGGER_FRAME,
                    KYO_SEVENTY_FIVE_SHIKI_KAI_RED_KICK_SCRIPT_ID,
                },
                {
                    KYO_SEVENTY_FIVE_SHIKI_KAI_ACTION_ID,
                    KYO_KOTOTSUKI_YOU_ACTION_ID,
                    0,
                    KYO_SEVENTY_FIVE_SHIKI_KAI_KOTOTSUKI_TRIGGER_FRAME,
                    KYO_SEVENTY_FIVE_SHIKI_KAI_KOTOTSUKI_TRIGGER_FRAME,
                },
                {
                    KYO_SEVENTY_FIVE_SHIKI_KAI_ACTION_ID,
                    KYO_ARAGAMI_ACTION_ID,
                    0,
                    KYO_SEVENTY_FIVE_SHIKI_KAI_ARAGAMI_TRIGGER_FRAME,
                    KYO_SEVENTY_FIVE_SHIKI_KAI_ARAGAMI_TRIGGER_FRAME,
                },
                {
                    KYO_ARAGAMI_ACTION_ID,
                    KYO_YANO_SABI_ACTION_ID,
                    0,
                    KYO_ARAGAMI_YANO_SABI_TRIGGER_FRAME,
                    KYO_ARAGAMI_YANO_SABI_TRIGGER_FRAME,
                },
                {
                    KYO_YANO_SABI_ACTION_ID,
                    KYO_MIGIRI_UGACHI_ACTION_ID,
                    0,
                    KYO_YANO_SABI_MIGIRI_UGACHI_TRIGGER_FRAME,
                    KYO_YANO_SABI_MIGIRI_UGACHI_TRIGGER_FRAME,
                },
            }));
    }

    void setCharacter(CharacterID id) {
        current_character_ = id;
    }

    const CharacterActionTable *getAction(bool forward_is_right) const {
        const auto it = lut_.find(current_character_);
        if (it != lut_.end()) {
            return (forward_is_right ? &it->second.first : &it->second.second);
        }
        return nullptr;
    }

    const FollowUpRule *findFollowUpRule(
        int32_t parent_action_id,
        int32_t child_action_id) const {
        const auto table_it = follow_up_lut_.find(current_character_);
        if (table_it == follow_up_lut_.cend())
            return nullptr;

        const auto rule_it = std::find_if(
            table_it->second.cbegin(),
            table_it->second.cend(),
            [parent_action_id, child_action_id](const FollowUpRule &rule) {
                return rule.parent_action_id == parent_action_id &&
                       rule.child_action_id == child_action_id;
            });
        return rule_it != table_it->second.cend() ? &*rule_it : nullptr;
    }

    bool hasPendingFollowUpWindow(
        int32_t parent_action_id,
        int32_t elapsed_frames) const {
        const auto table_it = follow_up_lut_.find(current_character_);
        if (table_it == follow_up_lut_.cend())
            return false;

        return std::any_of(
            table_it->second.cbegin(),
            table_it->second.cend(),
            [parent_action_id, elapsed_frames](const FollowUpRule &rule) {
                return rule.parent_action_id == parent_action_id &&
                       elapsed_frames < rule.queue_close_frame_exclusive;
            });
    }

private:
	CharacterID current_character_ = CharacterID::Kyo;
	std::map<CharacterID, std::pair<CharacterActionTable, CharacterActionTable>> lut_;
	std::map<CharacterID, FollowUpRuleTable> follow_up_lut_;
};

class FbneoTrainingRuntime {
public:
    FbneoTrainingRuntime() {
        clearStepEvents();
    }

    ~FbneoTrainingRuntime() {
        close();
    }

    bool loadCore(const wchar_t *core_path) {
        close();

        const std::wstring core_path_w = absoluteWidePath(core_path);
        if (core_path_w.empty())
            return fail("Core path is empty.");

        library_ = LoadLibraryW(core_path_w.c_str());
        if (!library_)
            return fail("Could not load FBNeo libretro core.");

        try {
            loadSymbols();
        } catch (const std::exception &exception) {
            close();
            last_error_ = std::string("FBNeo libretro core is missing required export: ") + exception.what();
            return false;
        }

        return true;
    }

    bool loadGame(const wchar_t *game_path, const wchar_t *system_directory, const wchar_t *save_directory) {
        if (!library_ || !retro_load_game_)
            return fail("Core is not loaded.");

        if (game_loaded_)
            unloadGame();
        clearStateCache();

        paths_.game = absoluteUtf8Path(game_path);
        paths_.system_directory = absoluteUtf8Path(system_directory);
        paths_.save_directory = absoluteUtf8Path(save_directory);
        if (paths_.game.empty())
            return fail("Game path is empty.");

        if (!initialized_) {
            g_active_runtime = this;
            installCallbacks();
            retro_init_();
            initialized_ = true;
        }

        if (retro_set_controller_port_device_) {
            retro_set_controller_port_device_(static_cast<unsigned>(P1_PORT), RETRO_DEVICE_JOYPAD);
            retro_set_controller_port_device_(static_cast<unsigned>(P2_PORT), RETRO_DEVICE_JOYPAD);
        }

        retro_game_info info {};
        info.path = paths_.game.c_str();
        info.data = nullptr;
        info.size = 0;
        info.meta = nullptr;

        if (!retro_load_game_(&info))
            return fail("FBNeo could not load game content.");

        game_loaded_ = true;
        size_t ram_size = 0;
        if (!systemRam(&ram_size) || ram_size == 0) {
            // Some FBNeo instances publish RETRO_MEMORY_SYSTEM_RAM only after
            // their first frame.  Prime it before Python can load a state and
            // request an observation; the state immediately replaces this
            // zero-input warm-up frame.
            joypads_[P1_PORT] = {};
            joypads_[P2_PORT] = {};
            g_active_runtime = this;
            retro_run_();
            if (!systemRam(&ram_size) || ram_size == 0) {
                unloadGame();
                return fail("FBNeo did not expose System RAM after its warm-up frame.");
            }
        }
        return true;
    }

    bool reset() {
        if (!game_loaded_ || !retro_reset_)
            return fail("No loaded game to reset.");

        joypads_[P1_PORT] = {};
        joypads_[P2_PORT] = {};
        last_frame_joypads_ = {};
        previous_frame_joypads_ = {};
        previous_p1_guard_action_active_ = false;
        engine_frame_index_ = 0;
        combat_boundary_in_batch_ = false;
        clearActionState();
        clearP2ActionState();
        p2_controller_.training_cycle = 0;
        clearStepEvents();
        combat_events_.advanceEpoch();
        retro_reset_();
        return true;
    }

    bool loadState(const wchar_t *state_path) {
        if (!game_loaded_ || !retro_unserialize_)
            return fail("No loaded game for state load.");

        const std::wstring path = absoluteWidePath(state_path);
        if (path.empty())
            return fail("State path is empty.");
        if (!cacheStateFile(path))
            return false;

        if (!retro_unserialize_(cached_state_.data.data(), cached_state_.data.size()))
            return fail("Core rejected state data.");

        joypads_[P1_PORT] = {};
        joypads_[P2_PORT] = {};
        last_frame_joypads_ = {};
        previous_frame_joypads_ = {};
        previous_p1_guard_action_active_ = false;
        engine_frame_index_ = 0;
        combat_boundary_in_batch_ = false;
        clearActionState();
        clearP2ActionState();
        p2_controller_.training_cycle = 0;
        clearStepEvents();
        combat_events_.advanceEpoch();
        return true;
    }

    bool saveState(const wchar_t *state_path) {
        if (!game_loaded_ || !retro_serialize_size_ || !retro_serialize_)
            return fail("No loaded game for state save.");

        const size_t size = retro_serialize_size_();
        if (size == 0)
            return fail("Core reported zero serialize size.");

        std::vector<uint8_t> data(size);
        if (!retro_serialize_(data.data(), data.size()))
            return fail("Core could not serialize state.");

        const std::wstring path = absoluteWidePath(state_path);
        std::ofstream file(path, std::ios::binary);
        if (!file)
            return fail("Could not open state file for writing.");

        file.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!file)
            return fail("Could not write state file.");

        cached_state_.path = path;
        cached_state_.data = std::move(data);
        return true;
    }

    void setJoypadForPort(unsigned port, const kof_env_joypad_state *state) {
        if (port >= PLAYER_PORT_COUNT)
            return;

        joypads_[port] = state ? *state : kof_env_joypad_state {};
    }

    void setJoypad(const kof_env_joypad_state *state) {
        setJoypadForPort(static_cast<unsigned>(P1_PORT), state);
    }

    bool getLastJoypadForPort(unsigned port, kof_env_joypad_state *state) const {
        if (port >= PLAYER_PORT_COUNT || !state)
            return false;

        *state = last_frame_joypads_[port];
        return true;
    }

    void setVideoRefresh(kof_env_video_refresh_t callback, void *user_data) {
        video_refresh_callback_ = callback;
        video_refresh_user_data_ = user_data;
    }

    void setP2RandomAiEnabled(bool enabled) {
        if (enabled) {
            p2_controller_.mode = P2ControlMode::ScriptedStyle;
        } else if (p2_controller_.scriptedStyleEnabled()) {
            p2_controller_.mode = P2ControlMode::Disabled;
        }
        clearP2ActionState();
        p2_controller_.training_cycle = 0;
        joypads_[P2_PORT] = {};
    }

    void setP2ActionAiEnabled(bool enabled) {
        if (enabled) {
            p2_controller_.mode = P2ControlMode::ActionApi;
        } else if (p2_controller_.actionApiEnabled()) {
            p2_controller_.mode = P2ControlMode::Disabled;
        }
        clearP2ActionState();
        joypads_[P2_PORT] = {};
    }

    bool setP2Style(int32_t style) {
        if (style < KOF_ENV_P2_STYLE_ONIYAKI ||
            style >= KOF_ENV_P2_STYLE_COUNT) {
            return fail("P2 style is out of range.");
        }

        p2_controller_.style = static_cast<kof_env_p2_style>(style);
        // 切換風格時也清除上一個腳本的 action/serial 歸因，避免新一局
        // 的第一個受擊事件被錯算成前一種 P2 風格的招式。
        clearP2ActionState();
        p2_controller_.training_cycle = 0;
        joypads_[P2_PORT] = {};
        return true;
    }

    bool inputReady() const {
        return p1_action_.active_action_id < 0;
    }

    bool snapshotSafe() const {
        kof_env_observation observation {};
        return game_loaded_ &&
               getObservation(&observation) &&
               observation.round_time > 0 &&
               observation.p1_health > 0 &&
               observation.p2_health > 0 &&
               observation.p1_has_position != 0 &&
               observation.p2_has_position != 0 &&
               p1ReadyForAction() &&
               p2ReadyForAction() &&
               p1_action_.active_action_id < 0 &&
               p1_action_.queued_action_id < 0 &&
               p1_action_.script.empty() &&
               !p1_action_.action_start_event_pending &&
               p2_controller_.active_action_id < 0 &&
               p2_controller_.queued_action_id < 0 &&
               p2_controller_.script.empty() &&
               !p2_controller_.action_start_event_pending;
    }

    bool canQueueAction(int32_t action_id) const {
        if (!isPublicActionId(action_id) ||
            p1_action_.active_action_id < 0 ||
            p1_action_.queued_action_id >= 0) {
            return false;
        }

        const FollowUpRule *rule = action_lut_.findFollowUpRule(
            p1_action_.active_action_id,
            action_id);
        return rule &&
               p1_action_.active_action_elapsed_frames >= rule->queue_open_frame &&
               p1_action_.active_action_elapsed_frames < rule->queue_close_frame_exclusive;
    }

    bool p2InputReady() const {
        return p2_controller_.active_action_id < 0;
    }

    // P2 有兩條控制路徑：setP2Action 使用 p2_controller_.active_action_id，訓練用
    // scripted style 則使用 p2_controller_.training_action_id。腳本輸入通常比畫面上的
    // 攻擊判定先結束，因此再保留一個有上限的 last-started fallback。
    int32_t currentP2ActionId() const {
        if (p2_controller_.actionApiEnabled() &&
            p2_controller_.active_action_id >= 0) {
            return p2_controller_.active_action_id;
        }
        if (p2_controller_.training_action_id >= 0)
            return p2_controller_.training_action_id;
        return p2_controller_.last_started_action_age_frames <= ACTION_ATTRIBUTION_FALLBACK_FRAMES
            ? p2_controller_.last_started_action_id
            : -1;
    }

    uint32_t currentP2ActionSerial() const {
        return currentP2ActionId() >= 0 ? p2_controller_.last_started_action_serial : 0;
    }

    int32_t currentP2ActionElapsedFrames() const {
        if (p2_controller_.actionApiEnabled() &&
            p2_controller_.active_action_id >= 0) {
            return p2_controller_.active_action_elapsed_frames;
        }
        if (p2_controller_.training_action_id >= 0)
            return p2_controller_.training_action_elapsed_frames;
        return p2_controller_.last_started_action_age_frames;
    }

    bool canQueueP2Action(int32_t action_id) const {
        if (!isPublicActionId(action_id) ||
            !p2_controller_.actionApiEnabled() ||
            p2_controller_.active_action_id < 0 ||
            p2_controller_.queued_action_id >= 0) {
            return false;
        }

        const FollowUpRule *rule = action_lut_.findFollowUpRule(
            p2_controller_.active_action_id,
            action_id);
        return rule &&
               p2_controller_.active_action_elapsed_frames >= rule->queue_open_frame &&
               p2_controller_.active_action_elapsed_frames < rule->queue_close_frame_exclusive;
    }

    bool getActionStatus(kof_env_action_status *status) const {
        if (!status)
            return false;

        status->active_action_id = p1_action_.active_action_id;
        status->queued_action_id = p1_action_.queued_action_id;
        status->last_started_action_id = p1_action_.last_started_action_id;
        status->action_accepted = p1_action_.last_action_accepted ? 1 : 0;
        return true;
    }

    bool getStrategyStateV1(kof_env_strategy_state_v1 *state) const {
        if (!state)
            return fail("Strategy state output pointer is null.");
        if (state->struct_size != sizeof(kof_env_strategy_state_v1))
            return fail("Strategy state v1 struct size does not match the DLL ABI.");
        if (state->version != KOF_ENV_STRATEGY_STATE_VERSION_1)
            return fail("Unsupported strategy state version.");

        kof_env_strategy_state_v1 result {};
        result.struct_size = sizeof(kof_env_strategy_state_v1);
        result.version = KOF_ENV_STRATEGY_STATE_VERSION_1;
        result.p1_active_action_id = p1_action_.active_action_id;
        result.p1_queued_action_id = p1_action_.queued_action_id;
        result.p1_last_started_action_id = p1_action_.last_started_action_id;
        result.p1_action_elapsed_frames = p1_action_.active_action_elapsed_frames;
        result.p1_action_remaining_frames = actionScriptRemainingFrames();

        const bool p2_uses_action_ai = p2_controller_.actionApiEnabled();
        result.p2_active_action_id = p2_uses_action_ai
            ? p2_controller_.active_action_id
            : p2_controller_.training_action_id;
        result.p2_queued_action_id = p2_uses_action_ai
            ? p2_controller_.queued_action_id
            : -1;
        result.p2_action_elapsed_frames = p2_uses_action_ai
            ? p2_controller_.active_action_elapsed_frames
            : p2_controller_.training_action_elapsed_frames;
        result.p2_action_remaining_frames = p2ActionScriptRemainingFrames();
        result.p1_input_ready = inputReady() ? 1 : 0;
        result.p2_input_ready = (p2_uses_action_ai
                ? p2InputReady()
                : p2_controller_.script.empty())
            ? 1
            : 0;
        result.p2_scripted = p2_controller_.mode != P2ControlMode::Disabled ? 1 : 0;

        size_t ram_size = 0;
        const uint8_t *ram_ptr = systemRam(&ram_size);
        if (ram_ptr && ram_size > 0) {
            const game_memory::GameMemReaderCore mem_reader(ram_ptr, ram_size);
            // ABI 欄位仍名為 status，但內容是 base+0x7c 的 hitbox slot
            // active mask；不可拿數值 7 當作 blockstun 狀態。
            result.p1_status = mem_reader.readP1HitboxActiveMask();
            result.p2_status = mem_reader.readP2HitboxActiveMask();
            result.p1_ready = mem_reader.p1ReadyForAction() ? 1 : 0;
            result.p2_ready = mem_reader.p2ReadyForAction() ? 1 : 0;
            bool p1_facing_left = false;
            if (mem_reader.readP1FacingLeft(p1_facing_left))
                result.p1_facing_left = p1_facing_left ? 1 : 0;
            bool p2_facing_left = false;
            if (mem_reader.readP2FacingLeft(p2_facing_left))
                result.p2_facing_left = p2_facing_left ? 1 : 0;
        } else {
            result.p1_status = -1;
            result.p2_status = -1;
        }

        *state = result;
        return true;
    }

    bool getStepEventsV1(kof_env_step_events_v1 *events) const {
        if (!events)
            return fail("Step events output pointer is null.");
        if (events->struct_size != sizeof(kof_env_step_events_v1))
            return fail("Step events v1 struct size does not match the DLL ABI.");
        if (events->version != KOF_ENV_STEP_EVENTS_VERSION_1)
            return fail("Unsupported step events version.");

        *events = combat_events_.v1;
        return true;
    }

    bool getStepEventsV2(kof_env_step_events_v2 *events) const {
        if (!events)
            return fail("Step events output pointer is null.");
        if (events->struct_size != sizeof(kof_env_step_events_v2))
            return fail("Step events v2 struct size does not match the DLL ABI.");
        if (events->version != KOF_ENV_STEP_EVENTS_VERSION_2)
            return fail("Unsupported step events version.");

        *events = combat_events_.v2;
        return true;
    }

    bool getStepEventsV3(kof_env_step_events_v3 *events) const {
        if (!events)
            return fail("Step events output pointer is null.");
        if (events->struct_size != sizeof(kof_env_step_events_v3))
            return fail("Step events v3 struct size does not match the DLL ABI.");
        if (events->version != KOF_ENV_STEP_EVENTS_VERSION_3)
            return fail("Unsupported step events version.");

        *events = combat_events_.v3;
        return true;
    }

    bool getStepEventsV4(kof_env_step_events_v4 *events) const {
        if (!events)
            return fail("Step events output pointer is null.");
        if (events->struct_size != sizeof(kof_env_step_events_v4))
            return fail("Step events v4 struct size does not match the DLL ABI.");
        if (events->version != KOF_ENV_STEP_EVENTS_VERSION_4)
            return fail("Unsupported step events version.");

        *events = combat_events_.v4;
        return true;
    }

    bool getStepEventsV5(kof_env_step_events_v5 *events) const {
        if (!events)
            return fail("Step events output pointer is null.");
        if (events->struct_size != sizeof(kof_env_step_events_v5))
            return fail("Step events v5 struct size does not match the DLL ABI.");
        if (events->version != KOF_ENV_STEP_EVENTS_VERSION_5)
            return fail("Unsupported step events version.");

        *events = combat_events_.v5;
        return true;
    }

    bool getCombatTimingStateV1(kof_env_combat_timing_state_v1 *state) const {
        if (!state)
            return fail("Combat timing state output pointer is null.");
        if (state->struct_size != sizeof(kof_env_combat_timing_state_v1))
            return fail("Combat timing state v1 struct size does not match the DLL ABI.");
        if (state->version != KOF_ENV_COMBAT_TIMING_STATE_VERSION_1)
            return fail("Unsupported combat timing state version.");

        kof_env_combat_timing_state_v1 result {};
        result.struct_size = sizeof(kof_env_combat_timing_state_v1);
        result.version = KOF_ENV_COMBAT_TIMING_STATE_VERSION_1;
        result.engine_frame = engine_frame_index_;
        result.event_epoch = combat_events_.event_epoch;

        const auto fill_player = [](
                                     kof_env_player_timing_state_v1 &target,
                                     bool input_script_ready,
                                     int32_t input_script_remaining,
                                     const ReactionTimingState &reaction) {
            target.input_script_ready = input_script_ready ? 1 : 0;
            target.input_script_remaining =
                std::max(0, input_script_remaining);
            target.reaction_valid = reaction.active ? 1 : 0;
            target.reaction_kind = reaction.active
                ? reaction.kind
                : KOF_ENV_REACTION_NONE;
            target.reaction_remaining =
                reaction.active && reaction.countdown_seen
                ? std::max(0, reaction.remaining)
                : 0;
            target.reaction_remaining_valid =
                reaction.active && reaction.countdown_seen ? 1 : 0;

            // 尚未以 Actionable Frame Oracle 驗證攻擊方 recovery RAM。
            // 明確標成 unknown，不能拿 input-script ready 代替。
            target.actionable_valid = 0;
            target.actionable = 0;
            target.recovery_valid = 0;
            target.recovery_remaining = 0;
        };

        fill_player(
            result.p1,
            inputReady(),
            actionScriptRemainingFrames(),
            combat_events_.reactions[P1_PORT]);
        const bool p2_script_ready = p2_controller_.actionApiEnabled()
            ? p2InputReady()
            : p2_controller_.script.empty();
        fill_player(
            result.p2,
            p2_script_ready,
            p2ActionScriptRemainingFrames(),
            combat_events_.reactions[P2_PORT]);

        result.frame_advantage_valid = 0;
        result.frame_advantage = 0;
        *state = result;
        return true;
    }

    void clearStepEvents() {
        // 每次 step/runFrames 都產生一批獨立事件；V1、V2 分開清空，
        // 舊客戶端不會讀到 V2 新增的防禦與 P1 受傷事件。
        combat_events_.clearBatch();
    }

    void clearCombatEventState() {
        combat_events_.recent_damage.reset();
    }

    void clearActionHandoffState() {
        combat_events_.handoff.reset();
    }

    void handleInStepCombatBoundary(int32_t frame_offset) {
        // ACTION_STARTED 會在 retro_run() 前先寫入；如果同一幀其實發生
        // 換局，該事件屬於舊回合的無效輸入，必須連同本幀其他事件回滾。
        // 本批較早幀的 KO/傷害事件仍保留，並以舊 epoch 回傳給 Python。
        combat_events_.rollbackFrame(frame_offset);
        combat_events_.advanceEpoch();
        combat_boundary_in_batch_ = true;

        joypads_[P1_PORT] = {};
        joypads_[P2_PORT] = {};
        last_frame_joypads_ = {};
        previous_frame_joypads_ = {};
        previous_p1_guard_action_active_ = false;
        clearActionState();
        clearP2ActionState();
    }

    FrameCombatState readFrameCombatState() const {
        // 同一幀前後各呼叫一次。這裡只留下事件判定真正需要的 RAM 與
        // hitbox 摘要，避免把 GameMemReaderCore 的生命週期帶出本函式。
        FrameCombatState state;
        size_t ram_size = 0;
        const uint8_t *ram_ptr = systemRam(&ram_size);
        if (!ram_ptr || ram_size == 0)
            return state;

        const game_memory::GameMemReaderCore mem_reader(ram_ptr, ram_size);
        state.p1_health = mem_reader.readP1Health();
        state.p2_health = mem_reader.readP2Health();
        state.p1_combo = mem_reader.readP1ComboCount();
        state.p2_combo = mem_reader.readP2ComboCount();
        state.p1_guard_crush_value = mem_reader.readP1GuardCrushValue();
        state.p2_guard_crush_value = mem_reader.readP2GuardCrushValue();
        state.p1_hit_guard_stop_raw = mem_reader.readP1HitGuardStopRaw();
        state.p2_hit_guard_stop_raw = mem_reader.readP2HitGuardStopRaw();
        const auto p1_reaction = mem_reader.readP1ReactionDebugState();
        state.p1_reaction_d2_raw = p1_reaction.reaction_d2;
        state.p1_reaction_counter = p1_reaction.reaction_d2d3_signed;
        state.p1_reaction_e3_raw = p1_reaction.reaction_e3;
        const auto p2_reaction = mem_reader.readP2ReactionDebugState();
        state.p2_reaction_d2_raw = p2_reaction.reaction_d2;
        state.p2_reaction_counter = p2_reaction.reaction_d2d3_signed;
        state.p2_reaction_e3_raw = p2_reaction.reaction_e3;
        state.has_p1_position = mem_reader.readP1Position(state.p1_position);
        state.has_p2_position = mem_reader.readP2Position(state.p2_position);
        const game_memory::HitboxOverlay overlay = mem_reader.getHitboxOverlay();
        state.p2_attack_guard_overlap = hasAttackGuardboxOverlap(
            overlay,
            2,
            1);
        state.p1_attack_guard_overlap = hasAttackGuardboxOverlap(
            overlay,
            1,
            2);
        return state;
    }

    bool joypadHoldingBack(const kof_env_joypad_state &input,
                           const FrameCombatState &state,
                           size_t player_port) const {
        if (state.has_p1_position && state.has_p2_position) {
            const bool opponent_is_right = player_port == P1_PORT
                ? state.p2_position.x >= state.p1_position.x
                : state.p1_position.x >= state.p2_position.x;
            return opponent_is_right
                       ? input.left != 0
                       : input.right != 0;
        }
        return input.left != 0 || input.right != 0;
    }

    bool p1HoldingBack(const FrameCombatState &state) const {
        return joypadHoldingBack(last_frame_joypads_[P1_PORT], state, P1_PORT);
    }

    bool p1WasHoldingBack(const FrameCombatState &state) const {
        return joypadHoldingBack(
            previous_frame_joypads_[P1_PORT],
            state,
            P1_PORT);
    }

    bool p2HoldingBack(const FrameCombatState &state) const {
        return joypadHoldingBack(last_frame_joypads_[P2_PORT], state, P2_PORT);
    }

    bool p2WasHoldingBack(const FrameCombatState &state) const {
        return joypadHoldingBack(
            previous_frame_joypads_[P2_PORT],
            state,
            P2_PORT);
    }

    void appendStepEvent(int32_t frame_offset,
                         kof_env_step_event_type event_type,
                         int32_t action_id,
                         uint32_t action_serial,
                         int32_t combo_before,
                         int32_t combo_after,
                         int32_t p2_hp_delta,
                         int32_t p1_hp_delta = 0,
                         int32_t target_y_at_event = -1,
                         bool target_airborne_before_event = false,
                         bool target_airborne_after_event = false,
                         bool hit_contact = false,
                         bool block_contact = false,
                         int32_t p1_hit_guard_stop_before = -1,
                         int32_t p1_hit_guard_stop_after = -1,
                         int32_t p2_hit_guard_stop_before = -1,
                          int32_t p2_hit_guard_stop_after = -1,
                          int32_t action_elapsed_frames_at_event = -1,
                          uint32_t guard_reaction_serial = 0,
                          int32_t source_player = 0,
                          int32_t target_player = 0) {
        if (source_player == 0 && target_player == 0) {
            switch (event_type) {
            case KOF_ENV_STEP_EVENT_ACTION_STARTED:
                source_player = 1;
                break;
            case KOF_ENV_STEP_EVENT_COMBO_HIT:
            case KOF_ENV_STEP_EVENT_DAMAGE_ONLY:
                source_player = 1;
                target_player = 2;
                break;
            default:
                source_player = 2;
                target_player = 1;
                break;
            }
        }

        // V1 只接受原本三種事件，維持既有 ABI 與舊前端行為。
        if (event_type <= KOF_ENV_STEP_EVENT_DAMAGE_ONLY) {
            if (combat_events_.v1.event_count >= KOF_ENV_STEP_EVENT_CAPACITY_V1) {
                ++combat_events_.v1.dropped_event_count;
            } else {
                auto &event = combat_events_.v1.events[combat_events_.v1.event_count++];
                event.frame_offset = frame_offset;
                event.event_type = event_type;
                event.action_id = action_id;
                event.action_serial = action_serial;
                event.combo_before = combo_before;
                event.combo_after = combo_after;
                event.p2_hp_delta = p2_hp_delta;
            }
        }

        if (combat_events_.v2.event_count >= KOF_ENV_STEP_EVENT_CAPACITY_V2) {
            ++combat_events_.v2.dropped_event_count;
        } else {
            auto &event = combat_events_.v2.events[combat_events_.v2.event_count++];
            event.frame_offset = frame_offset;
            event.event_type = event_type;
            event.action_id = action_id;
            event.action_serial = action_serial;
            event.combo_before = combo_before;
            event.combo_after = combo_after;
            event.p1_hp_delta = p1_hp_delta;
            event.p2_hp_delta = p2_hp_delta;
            event.target_y_at_event = target_y_at_event;
            event.target_airborne_at_event = target_airborne_before_event ? 1 : 0;
            event.hit_contact = hit_contact ? 1 : 0;
            event.block_contact = block_contact ? 1 : 0;
            event.target_airborne_after_event = target_airborne_after_event ? 1 : 0;
        }

        if (combat_events_.v3.event_count >= KOF_ENV_STEP_EVENT_CAPACITY_V3) {
            ++combat_events_.v3.dropped_event_count;
        } else {
            auto &event_v3 =
                combat_events_.v3.events[combat_events_.v3.event_count++];
            event_v3.frame_offset = frame_offset;
            event_v3.event_type = event_type;
            event_v3.action_id = action_id;
            event_v3.action_serial = action_serial;
            event_v3.combo_before = combo_before;
            event_v3.combo_after = combo_after;
            event_v3.p1_hp_delta = p1_hp_delta;
            event_v3.p2_hp_delta = p2_hp_delta;
            event_v3.target_y_at_event = target_y_at_event;
            event_v3.target_airborne_at_event =
                target_airborne_before_event ? 1 : 0;
            event_v3.hit_contact = hit_contact ? 1 : 0;
            event_v3.block_contact = block_contact ? 1 : 0;
            event_v3.target_airborne_after_event =
                target_airborne_after_event ? 1 : 0;
            event_v3.p1_hit_guard_stop_before = p1_hit_guard_stop_before;
            event_v3.p1_hit_guard_stop_after = p1_hit_guard_stop_after;
            event_v3.p2_hit_guard_stop_before = p2_hit_guard_stop_before;
            event_v3.p2_hit_guard_stop_after = p2_hit_guard_stop_after;
            event_v3.action_elapsed_frames_at_event =
                action_elapsed_frames_at_event;
            const bool has_contact = hit_contact || block_contact;
            event_v3.expected_blockstun_frames = has_contact
                ? kof98::expectedBlockstunFrames(
                      action_id,
                      target_airborne_before_event)
                : kof98::UnknownMoveValue;
            event_v3.expected_blockstun_source =
                event_v3.expected_blockstun_frames >= 0
                    ? static_cast<int32_t>(
                          kof98::MoveDataSource::PublishedTable)
                    : static_cast<int32_t>(kof98::MoveDataSource::Unknown);
        }

        const bool has_contact = hit_contact || block_contact;
        const int32_t expected_blockstun_frames = has_contact
            ? kof98::expectedBlockstunFrames(
                  action_id,
                  target_airborne_before_event)
            : kof98::UnknownMoveValue;
        const int32_t expected_blockstun_source =
            expected_blockstun_frames >= 0
                ? static_cast<int32_t>(kof98::MoveDataSource::PublishedTable)
                : static_cast<int32_t>(kof98::MoveDataSource::Unknown);

        if (combat_events_.v4.event_count >= KOF_ENV_STEP_EVENT_CAPACITY_V4) {
            ++combat_events_.v4.dropped_event_count;
        } else {
            auto &event_v4 =
                combat_events_.v4.events[combat_events_.v4.event_count++];
            event_v4.frame_offset = frame_offset;
            event_v4.event_type = event_type;
            event_v4.action_id = action_id;
            event_v4.action_serial = action_serial;
            event_v4.combo_before = combo_before;
            event_v4.combo_after = combo_after;
            event_v4.p1_hp_delta = p1_hp_delta;
            event_v4.p2_hp_delta = p2_hp_delta;
            event_v4.target_y_at_event = target_y_at_event;
            event_v4.target_airborne_at_event =
                target_airborne_before_event ? 1 : 0;
            event_v4.hit_contact = hit_contact ? 1 : 0;
            event_v4.block_contact = block_contact ? 1 : 0;
            event_v4.target_airborne_after_event =
                target_airborne_after_event ? 1 : 0;
            event_v4.p1_hit_guard_stop_before = p1_hit_guard_stop_before;
            event_v4.p1_hit_guard_stop_after = p1_hit_guard_stop_after;
            event_v4.p2_hit_guard_stop_before = p2_hit_guard_stop_before;
            event_v4.p2_hit_guard_stop_after = p2_hit_guard_stop_after;
            event_v4.action_elapsed_frames_at_event =
                action_elapsed_frames_at_event;
            event_v4.expected_blockstun_frames = expected_blockstun_frames;
            event_v4.expected_blockstun_source = expected_blockstun_source;
            event_v4.event_epoch = combat_events_.event_epoch;
            event_v4.guard_reaction_serial = guard_reaction_serial;
        }

        if (combat_events_.v5.event_count >= KOF_ENV_STEP_EVENT_CAPACITY_V5) {
            ++combat_events_.v5.dropped_event_count;
            return;
        }

        auto &event_v5 = combat_events_.v5.events[combat_events_.v5.event_count++];
        event_v5.frame_offset = frame_offset;
        event_v5.event_type = event_type;
        event_v5.action_id = action_id;
        event_v5.action_serial = action_serial;
        event_v5.combo_before = combo_before;
        event_v5.combo_after = combo_after;
        event_v5.p1_hp_delta = p1_hp_delta;
        event_v5.p2_hp_delta = p2_hp_delta;
        event_v5.target_y_at_event = target_y_at_event;
        event_v5.target_airborne_at_event =
            target_airborne_before_event ? 1 : 0;
        event_v5.hit_contact = hit_contact ? 1 : 0;
        event_v5.block_contact = block_contact ? 1 : 0;
        event_v5.target_airborne_after_event =
            target_airborne_after_event ? 1 : 0;
        event_v5.p1_hit_guard_stop_before = p1_hit_guard_stop_before;
        event_v5.p1_hit_guard_stop_after = p1_hit_guard_stop_after;
        event_v5.p2_hit_guard_stop_before = p2_hit_guard_stop_before;
        event_v5.p2_hit_guard_stop_after = p2_hit_guard_stop_after;
        event_v5.action_elapsed_frames_at_event =
            action_elapsed_frames_at_event;
        event_v5.expected_blockstun_frames = expected_blockstun_frames;
        event_v5.expected_blockstun_source = expected_blockstun_source;
        event_v5.event_epoch = combat_events_.event_epoch;
        event_v5.guard_reaction_serial = guard_reaction_serial;
        event_v5.source_player = source_player;
        event_v5.target_player = target_player;
        event_v5.absolute_engine_frame = engine_frame_index_;
    }

    void appendStepEventV5Only(
        int32_t frame_offset,
        kof_env_step_event_type event_type,
        int32_t action_id,
        uint32_t action_serial,
        int32_t combo_before,
        int32_t combo_after,
        int32_t p1_hp_delta,
        int32_t p2_hp_delta,
        int32_t target_y_at_event,
        bool target_airborne_before_event,
        bool target_airborne_after_event,
        bool hit_contact,
        bool block_contact,
        int32_t p1_hit_guard_stop_before,
        int32_t p1_hit_guard_stop_after,
        int32_t p2_hit_guard_stop_before,
        int32_t p2_hit_guard_stop_after,
        int32_t action_elapsed_frames_at_event,
        uint32_t guard_reaction_serial,
        int32_t source_player,
        int32_t target_player) {
        if (combat_events_.v5.event_count >= KOF_ENV_STEP_EVENT_CAPACITY_V5) {
            ++combat_events_.v5.dropped_event_count;
            return;
        }

        const bool has_contact = hit_contact || block_contact;
        auto &event = combat_events_.v5.events[combat_events_.v5.event_count++];
        event.frame_offset = frame_offset;
        event.event_type = event_type;
        event.action_id = action_id;
        event.action_serial = action_serial;
        event.combo_before = combo_before;
        event.combo_after = combo_after;
        event.p1_hp_delta = p1_hp_delta;
        event.p2_hp_delta = p2_hp_delta;
        event.target_y_at_event = target_y_at_event;
        event.target_airborne_at_event =
            target_airborne_before_event ? 1 : 0;
        event.hit_contact = hit_contact ? 1 : 0;
        event.block_contact = block_contact ? 1 : 0;
        event.target_airborne_after_event =
            target_airborne_after_event ? 1 : 0;
        event.p1_hit_guard_stop_before = p1_hit_guard_stop_before;
        event.p1_hit_guard_stop_after = p1_hit_guard_stop_after;
        event.p2_hit_guard_stop_before = p2_hit_guard_stop_before;
        event.p2_hit_guard_stop_after = p2_hit_guard_stop_after;
        event.action_elapsed_frames_at_event =
            action_elapsed_frames_at_event;
        event.expected_blockstun_frames = has_contact
            ? kof98::expectedBlockstunFrames(
                  action_id,
                  target_airborne_before_event)
            : kof98::UnknownMoveValue;
        event.expected_blockstun_source =
            event.expected_blockstun_frames >= 0
                ? static_cast<int32_t>(kof98::MoveDataSource::PublishedTable)
                : static_cast<int32_t>(kof98::MoveDataSource::Unknown);
        event.event_epoch = combat_events_.event_epoch;
        event.guard_reaction_serial = guard_reaction_serial;
        event.source_player = source_player;
        event.target_player = target_player;
        event.absolute_engine_frame = engine_frame_index_;
    }

    void updateReactionTiming(size_t player_port,
                              int32_t reaction_counter_before,
                              int32_t reaction_counter_after,
                              int32_t reaction_e3_before,
                              int32_t reaction_e3_after,
                              bool guard_contact,
                              bool hit_contact) {
        if (player_port >= PLAYER_PORT_COUNT)
            return;

        auto &reaction = combat_events_.reactions[player_port];
        const int32_t signature_before =
            reaction_e3_before >= 0 ? reaction_e3_before & 0x60 : -1;
        const int32_t signature_after =
            reaction_e3_after >= 0 ? reaction_e3_after & 0x60 : -1;
        if (guard_contact && signature_after == 0x20) {
            reaction.beginOrRefresh(KOF_ENV_REACTION_GUARD);
        } else if (hit_contact && signature_after == 0x60) {
            reaction.beginOrRefresh(KOF_ENV_REACTION_HIT);
        }

        if (!reaction.active)
            return;

        ++reaction.age_frames;
        const bool countdown_progressed =
            reaction_counter_before > 0 &&
            reaction_counter_after == reaction_counter_before - 1;
        if (countdown_progressed) {
            reaction.countdown_seen = true;
            reaction.remaining = reaction_counter_after + 1;
        } else if (reaction.countdown_seen && reaction_counter_after >= 0) {
            reaction.remaining = reaction_counter_after + 1;
        }

        const bool reaction_ended =
            reaction.countdown_seen &&
            reaction_counter_before == 0 &&
            reaction_counter_after == -1 &&
            (reaction.kind != KOF_ENV_REACTION_GUARD ||
             (signature_before == 0x20 && signature_after == 0x00));
        // 已確認倒數後，只有 0 -> -1 是目前驗證過的正常結束。
        // 0 -> -2、正數直接跳負值或其他未知負值一律保守失效，
        // 避免把舊 remaining 暴露給 PPO 直到 240F timeout。
        const bool unexpected_negative =
            shouldAbortUnexpectedReactionCounter(
                reaction.countdown_seen,
                reaction_counter_after,
                reaction_ended);
        if (reaction_ended ||
            unexpected_negative ||
            reaction.age_frames > GUARD_REACTION_TIMEOUT_FRAMES) {
            reaction.reset();
        }
    }

    void detectFrameCombatEvents(int32_t frame_offset,
                                 int32_t acting_action_id,
                                 uint32_t acting_action_serial,
                                 bool p1_guard_action_active,
                                 bool p1_guard_action_was_active,
                                 const FrameCombatState &before,
                                 const FrameCombatState &after) {
        // runFrames() 中途若換局，保留本批次較早已寫入的合法事件，只清除
        // 跨幀 tracker。這和 explicit reset/loadState 的整批 clear 不同。
        const bool combat_boundary =
            (before.p1_health >= 0 && after.p1_health > before.p1_health) ||
            (before.p2_health >= 0 && after.p2_health > before.p2_health);
        if (combat_boundary) {
            handleInStepCombatBoundary(frame_offset);
            return;
        }

        if (combat_events_.recent_damage.action_id >= 0)
            ++combat_events_.recent_damage.age_frames;

        const int32_t p2_hp_delta =
            before.p2_health >= 0 && after.p2_health >= 0
                ? std::max(0, before.p2_health - after.p2_health)
                : 0;
        const int32_t p1_hp_delta =
            before.p1_health >= 0 && after.p1_health >= 0
                ? std::max(0, before.p1_health - after.p1_health)
                : 0;
        const bool combo_rose =
            before.p1_combo >= 0 &&
            after.p1_combo >= 0 &&
            after.p1_combo > before.p1_combo;
        const bool p2_combo_rose =
            before.p2_combo >= 0 &&
            after.p2_combo >= 0 &&
            after.p2_combo > before.p2_combo;
        int32_t target_y = before.has_p2_position
            ? before.p2_position.y
            : (after.has_p2_position ? after.p2_position.y : -1);
        bool target_airborne_before =
            before.has_p2_position &&
            before.p2_position.y < KOF98_AIRBORNE_Y_THRESHOLD;
        bool target_airborne_after =
            after.has_p2_position &&
            after.p2_position.y < KOF98_AIRBORNE_Y_THRESHOLD;

        if (combo_rose) {
            int32_t hit_action_id = acting_action_id;
            uint32_t hit_action_serial = acting_action_serial;
            // Queue 觸發後，child 的輸入腳本可能已開始，但畫面上的 parent
            // 判定會晚 1~2 幀才命中。handoff grace 只修正這段短窗口。
            if (combat_events_.handoff.action_id >= 0 &&
                combat_events_.handoff.age_frames <= ACTION_HANDOFF_HIT_GRACE_FRAMES) {
                hit_action_id = combat_events_.handoff.action_id;
                hit_action_serial = combat_events_.handoff.action_serial;
            } else if (p2_hp_delta == 0 &&
                combat_events_.recent_damage.action_id >= 0 &&
                combat_events_.recent_damage.age_frames <= COMBO_EVENT_DAMAGE_GRACE_FRAMES) {
                hit_action_id = combat_events_.recent_damage.action_id;
                hit_action_serial = combat_events_.recent_damage.action_serial;
                target_y = combat_events_.recent_damage.target_y;
                target_airborne_before = combat_events_.recent_damage.target_airborne_before;
                target_airborne_after = combat_events_.recent_damage.target_airborne_after;
            }

            appendStepEvent(
                frame_offset,
                KOF_ENV_STEP_EVENT_COMBO_HIT,
                hit_action_id,
                hit_action_serial,
                before.p1_combo,
                after.p1_combo,
                p2_hp_delta,
                0,
                target_y,
                target_airborne_before,
                target_airborne_after,
                true,
                false,
                before.p1_hit_guard_stop_raw,
                after.p1_hit_guard_stop_raw,
                before.p2_hit_guard_stop_raw,
                after.p2_hit_guard_stop_raw,
                p1_action_.active_action_elapsed_frames);
            clearCombatEventState();
            clearActionHandoffState();
        } else if (p2_hp_delta > 0) {
            appendStepEvent(
                frame_offset,
                KOF_ENV_STEP_EVENT_DAMAGE_ONLY,
                acting_action_id,
                acting_action_serial,
                before.p1_combo,
                after.p1_combo,
                p2_hp_delta,
                0,
                target_y,
                target_airborne_before,
                target_airborne_after,
                true,
                false,
                before.p1_hit_guard_stop_raw,
                after.p1_hit_guard_stop_raw,
                before.p2_hit_guard_stop_raw,
                after.p2_hit_guard_stop_raw,
                p1_action_.active_action_elapsed_frames);
            combat_events_.recent_damage.action_id = acting_action_id;
            combat_events_.recent_damage.action_serial = acting_action_serial;
            combat_events_.recent_damage.age_frames = 0;
            combat_events_.recent_damage.target_y = target_y;
            combat_events_.recent_damage.target_airborne_before = target_airborne_before;
            combat_events_.recent_damage.target_airborne_after = target_airborne_after;
        } else if (combat_events_.recent_damage.age_frames > COMBO_EVENT_DAMAGE_GRACE_FRAMES) {
            clearCombatEventState();
        }

        const int32_t p2_action_id = currentP2ActionId();
        const uint32_t p2_action_serial = currentP2ActionSerial();
        const bool guard_value_spent =
            before.p1_guard_crush_value >= 0 &&
            after.p1_guard_crush_value >= 0 &&
            before.p1_guard_crush_value > after.p1_guard_crush_value;
        const bool guardbox_contact =
            before.p2_attack_guard_overlap || after.p2_attack_guard_overlap;
        const int32_t reaction_kind_before =
            before.p1_reaction_e3_raw >= 0
                ? before.p1_reaction_e3_raw & 0x60
                : -1;
        const int32_t reaction_kind_after =
            after.p1_reaction_e3_raw >= 0
                ? after.p1_reaction_e3_raw & 0x60
                : -1;
        const bool guard_signature_after = reaction_kind_after == 0x20;
        const bool manual_guard_input =
            (p1_guard_action_active &&
             (p1HoldingBack(before) || p1HoldingBack(after))) ||
            (p1_guard_action_was_active &&
             (p1WasHoldingBack(before) || p1WasHoldingBack(after)));
        const bool chip_block_fallback =
            guard_signature_after &&
            p1_hp_delta > 0;

        // 0x7C 是 hitbox slot active mask，不是 blockstun enum。物理格擋
        // 以 Guard Crush 消耗為主要證據；E3=0x20 下的 chip damage 與
        // Attack/Guard box overlap 只作補強。物理結果不依賴 P2 高階
        // Action ID，因此 raw joypad 與遊戲內 CPU 也能建立 tracker；
        // 這些來源的 attacker action 合法為 -1/0。
        const bool block_contact =
            guard_signature_after &&
            (guard_value_spent ||
             chip_block_fallback ||
             guardbox_contact);
        const bool block_edge =
            block_contact &&
            (!combat_events_.block_contact_active[P1_PORT] ||
             guard_value_spent ||
             p1_hp_delta > 0);

        if (block_edge) {
            const int32_t p1_y = before.has_p1_position
                ? before.p1_position.y
                : (after.has_p1_position ? after.p1_position.y : -1);
            const bool p1_airborne_before =
                before.has_p1_position &&
                before.p1_position.y < KOF98_AIRBORNE_Y_THRESHOLD;
            const bool p1_airborne_after =
                after.has_p1_position &&
                after.p1_position.y < KOF98_AIRBORNE_Y_THRESHOLD;

            // 多段攻擊只維持一個 tracker。新 contact 會更新最後攻擊來源、
            // 重設 hitstop gate，等新倒數再次出現 N→N-1 才解除 refresh。
            auto &guard_reaction = combat_events_.guard_reactions[P1_PORT];
            const int32_t hit_guard_stop = std::max(
                before.p1_hit_guard_stop_raw,
                after.p1_hit_guard_stop_raw);
            if (!guard_reaction.active) {
                guard_reaction.begin(
                    p2_action_id,
                    p2_action_serial,
                    combat_events_.allocateGuardReactionSerial(),
                    engine_frame_index_,
                    hit_guard_stop,
                    after.p1_reaction_counter,
                    manual_guard_input);
            } else {
                guard_reaction.refresh(
                    p2_action_id,
                    p2_action_serial,
                    engine_frame_index_,
                    hit_guard_stop,
                    after.p1_reaction_counter,
                    manual_guard_input);
            }

            appendStepEvent(
                frame_offset,
                KOF_ENV_STEP_EVENT_BLOCK_CONTACT,
                p2_action_id,
                p2_action_serial,
                before.p1_combo,
                after.p1_combo,
                0,
                0,
                p1_y,
                p1_airborne_before,
                p1_airborne_after,
                false,
                true,
                before.p1_hit_guard_stop_raw,
                after.p1_hit_guard_stop_raw,
                before.p2_hit_guard_stop_raw,
                after.p2_hit_guard_stop_raw,
                currentP2ActionElapsedFrames(),
                guard_reaction.reaction_serial);

            // Manual 需要正面證據，而且同一條 guard string 最多發一次。
            // 沒有手動證據時保持 Unknown；AUTO_GUARD 只保留給未來找到
            // autoguard RAM source 後使用。
            if (manual_guard_input && !guard_reaction.manual_success_emitted) {
                appendStepEvent(
                    frame_offset,
                    KOF_ENV_STEP_EVENT_MANUAL_BLOCK_SUCCESS,
                    p2_action_id,
                    p2_action_serial,
                    before.p1_combo,
                    after.p1_combo,
                    0,
                    0,
                    p1_y,
                    p1_airborne_before,
                    p1_airborne_after,
                    false,
                    true,
                    before.p1_hit_guard_stop_raw,
                    after.p1_hit_guard_stop_raw,
                    before.p2_hit_guard_stop_raw,
                    after.p2_hit_guard_stop_raw,
                    currentP2ActionElapsedFrames(),
                    guard_reaction.reaction_serial);
                guard_reaction.manual_success_emitted = true;
            }
        }

        auto &guard_reaction = combat_events_.guard_reactions[P1_PORT];
        const int32_t p1_y = before.has_p1_position
            ? before.p1_position.y
            : (after.has_p1_position ? after.p1_position.y : -1);
        const bool p1_airborne_before =
            before.has_p1_position &&
            before.p1_position.y < KOF98_AIRBORNE_Y_THRESHOLD;
        const bool p1_airborne_after =
            after.has_p1_position &&
            after.p1_position.y < KOF98_AIRBORNE_Y_THRESHOLD;

        // 合法 END 必須先於「E3 異常消失」處理。只接受完整 signed
        // D2:D3 的 0→-1，且 E3 由 Guard signature 0x20 回到中性 0x00。
        // 0→-2 或 0x20→0x60 都保守丟棄，不開啟訓練確反窗口。
        const bool blockstun_ended =
            guard_reaction.active &&
            guard_reaction.confirmed_block_contact &&
            guard_reaction.countdown_loaded &&
            !guard_reaction.refresh_pending &&
            !guard_reaction.end_emitted &&
            before.p1_reaction_counter == 0 &&
            after.p1_reaction_counter == -1 &&
            reaction_kind_before == 0x20 &&
            reaction_kind_after == 0x00;
        if (blockstun_ended) {
            appendStepEvent(
                frame_offset,
                KOF_ENV_STEP_EVENT_BLOCKSTUN_ENDED,
                guard_reaction.attacker_action_id,
                guard_reaction.attacker_action_serial,
                before.p1_combo,
                after.p1_combo,
                0,
                0,
                p1_y,
                p1_airborne_before,
                p1_airborne_after,
                false,
                false,
                before.p1_hit_guard_stop_raw,
                after.p1_hit_guard_stop_raw,
                before.p2_hit_guard_stop_raw,
                after.p2_hit_guard_stop_raw,
                currentP2ActionElapsedFrames(),
                guard_reaction.reaction_serial);
            guard_reaction.end_emitted = true;
            guard_reaction.reset();
        } else if (guard_reaction.active) {
            const bool gate_open =
                engine_frame_index_ >= guard_reaction.not_before_frame;
            const bool waiting_for_countdown =
                guard_reaction.phase == GuardReactionPhase::WaitingForCountdown ||
                guard_reaction.phase == GuardReactionPhase::RefreshPending;
            const bool countdown_progressed =
                gate_open &&
                waiting_for_countdown &&
                reaction_kind_before == 0x20 &&
                reaction_kind_after == 0x20 &&
                before.p1_reaction_counter > 0 &&
                after.p1_reaction_counter ==
                    before.p1_reaction_counter - 1;

            if (countdown_progressed) {
                // 直接使用同一 emulator frame 的 before/after 驗證 N→N-1。
                // 這同時涵蓋 1→0；舊的「after > 0」外層條件會漏掉它。
                guard_reaction.countdown_loaded = true;
                guard_reaction.refresh_pending = false;
                guard_reaction.phase = GuardReactionPhase::CountdownActive;
                guard_reaction.candidate_counter = -1;
                guard_reaction.candidate_counter_frame = 0;

                if (!guard_reaction.start_emitted) {
                    // START 是確認 N→N-1 的 telemetry；Python 的確反邏輯
                    // 只依可靠 END，因此不需要回填到前一幀。
                    appendStepEvent(
                        frame_offset,
                        KOF_ENV_STEP_EVENT_BLOCKSTUN_STARTED,
                        guard_reaction.attacker_action_id,
                        guard_reaction.attacker_action_serial,
                        before.p1_combo,
                        after.p1_combo,
                        0,
                        0,
                        p1_y,
                        p1_airborne_before,
                        p1_airborne_after,
                        false,
                        false,
                        before.p1_hit_guard_stop_raw,
                        after.p1_hit_guard_stop_raw,
                        before.p2_hit_guard_stop_raw,
                        after.p2_hit_guard_stop_raw,
                        currentP2ActionElapsedFrames(),
                        guard_reaction.reaction_serial);
                    guard_reaction.start_emitted = true;
                }
            } else if (gate_open && reaction_kind_after != 0x20) {
                // 合法 END 已在上方先處理；其餘 E3 消失或轉成 Hit
                // signature 都是不完整/未知反應，不產生訓練事件。
                guard_reaction.reset();
            }
        }

        if (p1_hp_delta > 0) {
            const int32_t p1_y = before.has_p1_position
                ? before.p1_position.y
                : (after.has_p1_position ? after.p1_position.y : -1);
            const bool p1_airborne_before =
                before.has_p1_position &&
                before.p1_position.y < KOF98_AIRBORNE_Y_THRESHOLD;
            const bool p1_airborne_after =
                after.has_p1_position &&
                after.p1_position.y < KOF98_AIRBORNE_Y_THRESHOLD;
            const bool chip_damage = block_contact;
            appendStepEvent(
                frame_offset,
                chip_damage
                    ? KOF_ENV_STEP_EVENT_CHIP_DAMAGE
                    : KOF_ENV_STEP_EVENT_CLEAN_HIT,
                p2_action_id,
                p2_action_serial,
                before.p1_combo,
                after.p1_combo,
                0,
                p1_hp_delta,
                p1_y,
                p1_airborne_before,
                p1_airborne_after,
                true,
                chip_damage,
                before.p1_hit_guard_stop_raw,
                after.p1_hit_guard_stop_raw,
                before.p2_hit_guard_stop_raw,
                after.p2_hit_guard_stop_raw,
                currentP2ActionElapsedFrames(),
                block_contact
                    ? combat_events_.guard_reactions[P1_PORT].reaction_serial
                    : 0);
        }

        if (p1_hp_delta > 0 && !block_contact)
            combat_events_.guard_reactions[P1_PORT].reset();

        if (combat_events_.guard_reactions[P1_PORT].active) {
            ++combat_events_.guard_reactions[P1_PORT].age_frames;
            if (combat_events_.guard_reactions[P1_PORT].age_frames >
                GUARD_REACTION_TIMEOUT_FRAMES) {
                combat_events_.guard_reactions[P1_PORT].reset();
            }
        }

        combat_events_.block_contact_active[P1_PORT] = block_contact;

        // StrategyV4 對稱路徑：P1 攻擊、P2 防禦。這些新事件只寫入 V5，
        // 避免改變 V1-V4 原本「以 P1 為學習主體」的事件集合。
        const int32_t p2_reaction_kind_before =
            before.p2_reaction_e3_raw >= 0
                ? before.p2_reaction_e3_raw & 0x60
                : -1;
        const int32_t p2_reaction_kind_after =
            after.p2_reaction_e3_raw >= 0
                ? after.p2_reaction_e3_raw & 0x60
                : -1;
        const bool p2_guard_signature_after =
            p2_reaction_kind_after == 0x20;
        const bool p2_guard_value_spent =
            before.p2_guard_crush_value >= 0 &&
            after.p2_guard_crush_value >= 0 &&
            before.p2_guard_crush_value > after.p2_guard_crush_value;
        const bool p2_chip_block_fallback =
            p2_guard_signature_after && p2_hp_delta > 0 && !combo_rose;
        const bool p2_block_contact =
            p2_guard_signature_after &&
            (p2_guard_value_spent ||
             p2_chip_block_fallback ||
             before.p1_attack_guard_overlap ||
             after.p1_attack_guard_overlap);
        if (p2_block_contact) {
            // P2 削血會先被舊相容路徑看成 DAMAGE_ONLY。V1-V4 保留原 ABI，
            // 但 V5 必須讓 Hit/Block 互斥，避免同一幀同時推進
            // StarterHit 與 StarterBlocked。
            combat_events_.removeV5HitEventsForBlock(frame_offset, 1, 2);
            clearCombatEventState();
        }
        const bool p2_block_edge =
            p2_block_contact &&
            (!combat_events_.block_contact_active[P2_PORT] ||
             p2_guard_value_spent ||
             p2_hp_delta > 0);
        const bool p2_manual_guard_input =
            p2HoldingBack(before) ||
            p2HoldingBack(after) ||
            p2WasHoldingBack(before) ||
            p2WasHoldingBack(after);
        auto &p2_guard_reaction =
            combat_events_.guard_reactions[P2_PORT];
        const int32_t p2_y = before.has_p2_position
            ? before.p2_position.y
            : (after.has_p2_position ? after.p2_position.y : -1);
        const bool p2_airborne_before =
            before.has_p2_position &&
            before.p2_position.y < KOF98_AIRBORNE_Y_THRESHOLD;
        const bool p2_airborne_after =
            after.has_p2_position &&
            after.p2_position.y < KOF98_AIRBORNE_Y_THRESHOLD;

        if (p2_block_edge) {
            const int32_t hit_guard_stop = std::max(
                before.p2_hit_guard_stop_raw,
                after.p2_hit_guard_stop_raw);
            if (!p2_guard_reaction.active) {
                p2_guard_reaction.begin(
                    acting_action_id,
                    acting_action_serial,
                    combat_events_.allocateGuardReactionSerial(),
                    engine_frame_index_,
                    hit_guard_stop,
                    after.p2_reaction_counter,
                    p2_manual_guard_input);
            } else {
                p2_guard_reaction.refresh(
                    acting_action_id,
                    acting_action_serial,
                    engine_frame_index_,
                    hit_guard_stop,
                    after.p2_reaction_counter,
                    p2_manual_guard_input);
            }

            appendStepEventV5Only(
                frame_offset,
                KOF_ENV_STEP_EVENT_BLOCK_CONTACT,
                acting_action_id,
                acting_action_serial,
                before.p1_combo,
                after.p1_combo,
                0,
                0,
                p2_y,
                p2_airborne_before,
                p2_airborne_after,
                false,
                true,
                before.p1_hit_guard_stop_raw,
                after.p1_hit_guard_stop_raw,
                before.p2_hit_guard_stop_raw,
                after.p2_hit_guard_stop_raw,
                p1_action_.active_action_elapsed_frames,
                p2_guard_reaction.reaction_serial,
                1,
                2);

            if (p2_manual_guard_input &&
                !p2_guard_reaction.manual_success_emitted) {
                appendStepEventV5Only(
                    frame_offset,
                    KOF_ENV_STEP_EVENT_MANUAL_BLOCK_SUCCESS,
                    acting_action_id,
                    acting_action_serial,
                    before.p1_combo,
                    after.p1_combo,
                    0,
                    0,
                    p2_y,
                    p2_airborne_before,
                    p2_airborne_after,
                    false,
                    true,
                    before.p1_hit_guard_stop_raw,
                    after.p1_hit_guard_stop_raw,
                    before.p2_hit_guard_stop_raw,
                    after.p2_hit_guard_stop_raw,
                    p1_action_.active_action_elapsed_frames,
                    p2_guard_reaction.reaction_serial,
                    1,
                    2);
                p2_guard_reaction.manual_success_emitted = true;
            }
        }

        const bool p2_blockstun_ended =
            p2_guard_reaction.active &&
            p2_guard_reaction.confirmed_block_contact &&
            p2_guard_reaction.countdown_loaded &&
            !p2_guard_reaction.refresh_pending &&
            !p2_guard_reaction.end_emitted &&
            before.p2_reaction_counter == 0 &&
            after.p2_reaction_counter == -1 &&
            p2_reaction_kind_before == 0x20 &&
            p2_reaction_kind_after == 0x00;
        if (p2_blockstun_ended) {
            appendStepEventV5Only(
                frame_offset,
                KOF_ENV_STEP_EVENT_BLOCKSTUN_ENDED,
                p2_guard_reaction.attacker_action_id,
                p2_guard_reaction.attacker_action_serial,
                before.p1_combo,
                after.p1_combo,
                0,
                0,
                p2_y,
                p2_airborne_before,
                p2_airborne_after,
                false,
                false,
                before.p1_hit_guard_stop_raw,
                after.p1_hit_guard_stop_raw,
                before.p2_hit_guard_stop_raw,
                after.p2_hit_guard_stop_raw,
                p1_action_.active_action_elapsed_frames,
                p2_guard_reaction.reaction_serial,
                1,
                2);
            p2_guard_reaction.end_emitted = true;
            p2_guard_reaction.reset();
        } else if (p2_guard_reaction.active) {
            const bool gate_open =
                engine_frame_index_ >= p2_guard_reaction.not_before_frame;
            const bool waiting_for_countdown =
                p2_guard_reaction.phase ==
                    GuardReactionPhase::WaitingForCountdown ||
                p2_guard_reaction.phase ==
                    GuardReactionPhase::RefreshPending;
            const bool countdown_progressed =
                gate_open &&
                waiting_for_countdown &&
                p2_reaction_kind_before == 0x20 &&
                p2_reaction_kind_after == 0x20 &&
                before.p2_reaction_counter > 0 &&
                after.p2_reaction_counter ==
                    before.p2_reaction_counter - 1;
            if (countdown_progressed) {
                p2_guard_reaction.countdown_loaded = true;
                p2_guard_reaction.refresh_pending = false;
                p2_guard_reaction.phase =
                    GuardReactionPhase::CountdownActive;
                if (!p2_guard_reaction.start_emitted) {
                    appendStepEventV5Only(
                        frame_offset,
                        KOF_ENV_STEP_EVENT_BLOCKSTUN_STARTED,
                        p2_guard_reaction.attacker_action_id,
                        p2_guard_reaction.attacker_action_serial,
                        before.p1_combo,
                        after.p1_combo,
                        0,
                        0,
                        p2_y,
                        p2_airborne_before,
                        p2_airborne_after,
                        false,
                        false,
                        before.p1_hit_guard_stop_raw,
                        after.p1_hit_guard_stop_raw,
                        before.p2_hit_guard_stop_raw,
                        after.p2_hit_guard_stop_raw,
                        p1_action_.active_action_elapsed_frames,
                        p2_guard_reaction.reaction_serial,
                        1,
                        2);
                    p2_guard_reaction.start_emitted = true;
                }
            } else if (gate_open && p2_reaction_kind_after != 0x20) {
                p2_guard_reaction.reset();
            }
        }

        if (p2_chip_block_fallback) {
            appendStepEventV5Only(
                frame_offset,
                KOF_ENV_STEP_EVENT_CHIP_DAMAGE,
                acting_action_id,
                acting_action_serial,
                before.p1_combo,
                after.p1_combo,
                0,
                p2_hp_delta,
                p2_y,
                p2_airborne_before,
                p2_airborne_after,
                false,
                true,
                before.p1_hit_guard_stop_raw,
                after.p1_hit_guard_stop_raw,
                before.p2_hit_guard_stop_raw,
                after.p2_hit_guard_stop_raw,
                p1_action_.active_action_elapsed_frames,
                p2_guard_reaction.reaction_serial,
                1,
                2);
        }

        if (p2_guard_reaction.active) {
            ++p2_guard_reaction.age_frames;
            if (p2_guard_reaction.age_frames >
                GUARD_REACTION_TIMEOUT_FRAMES) {
                p2_guard_reaction.reset();
            }
        }
        combat_events_.block_contact_active[P2_PORT] = p2_block_contact;

        if (combat_events_.handoff.action_id >= 0) {
            ++combat_events_.handoff.age_frames;
            if (combat_events_.handoff.age_frames > ACTION_HANDOFF_HIT_GRACE_FRAMES)
                clearActionHandoffState();
        }

        updateReactionTiming(
            P1_PORT,
            before.p1_reaction_counter,
            after.p1_reaction_counter,
            before.p1_reaction_e3_raw,
            after.p1_reaction_e3_raw,
            block_edge,
            (p1_hp_delta > 0 || p2_combo_rose) &&
                !block_contact &&
                reaction_kind_after == 0x60);
        updateReactionTiming(
            P2_PORT,
            before.p2_reaction_counter,
            after.p2_reaction_counter,
            before.p2_reaction_e3_raw,
            after.p2_reaction_e3_raw,
            p2_block_edge,
            (p2_hp_delta > 0 || combo_rose) &&
                !p2_block_contact &&
                p2_reaction_kind_after == 0x60);
    }

    void clearActionScript() {
        p1_action_.clearScript();
    }

    int32_t actionScriptRemainingFrames() const {
        return p1_action_.scriptRemainingFrames();
    }

    int32_t p2ActionScriptRemainingFrames() const {
        return p2_controller_.scriptRemainingFrames();
    }

    void clearActionState() {
        clearActionHandoffState();
        p1_action_.reset();
    }

    bool startActionScript(int32_t action_id, std::vector<InputFrame> script) {
        if (script.empty())
            return fail("Action script is empty.");

        if (p1_action_.active_action_id > IDLE_ACTION_ID &&
            p1_action_.active_action_serial != 0 &&
            action_id != p1_action_.active_action_id) {
            // Cancel 發生時 child 已成為 active，但 parent 的攻擊框可能在
            // 接下來數幀才命中；暫存 parent serial 才不會把命中歸給 child。
            combat_events_.handoff.action_id = p1_action_.active_action_id;
            combat_events_.handoff.action_serial = p1_action_.active_action_serial;
            combat_events_.handoff.age_frames = 0;
        } else {
            clearActionHandoffState();
        }

        p1_action_.script = std::move(script);
        p1_action_.script_index = 0;
        p1_action_.script_remaining_frames = p1_action_.script[0].frames;
        p1_action_.active_action_id = action_id;
        p1_action_.active_action_elapsed_frames = 0;
        if (action_id != IDLE_ACTION_ID) {
            ++p1_action_.next_action_serial;
            if (p1_action_.next_action_serial == 0)
                ++p1_action_.next_action_serial;
            p1_action_.active_action_serial = p1_action_.next_action_serial;
            p1_action_.action_start_event_pending = true;
            p1_action_.last_started_action_id = action_id;
            p1_action_.last_started_action_serial = p1_action_.active_action_serial;
            p1_action_.last_started_action_age_frames = 0;
        } else {
            p1_action_.active_action_serial = 0;
            p1_action_.action_start_event_pending = false;
        }
        joypads_[P1_PORT] = p1_action_.script[0].state;
        return true;
    }

    void advanceActionScriptFrame() {
        if (p1_action_.script.empty())
            return;

        if (p1_action_.script_remaining_frames <= 0) {
            ++p1_action_.script_index;
            if (p1_action_.script_index >= p1_action_.script.size()) {
                clearActionScript();
                joypads_[P1_PORT] = {};
                return;
            }

            p1_action_.script_remaining_frames = p1_action_.script[p1_action_.script_index].frames;
        }

        if (p1_action_.script_remaining_frames <= 0)
            return;

        joypads_[P1_PORT] = p1_action_.script[p1_action_.script_index].state;
        --p1_action_.script_remaining_frames;
    }

    void finishActionScriptFrame() {
        if (p1_action_.script.empty())
            return;

        if (p1_action_.script_remaining_frames <= 0 &&
            p1_action_.script_index + 1 >= p1_action_.script.size()) {
            clearActionScript();
            joypads_[P1_PORT] = {};
        }
    }

    void advanceActionLifecycleFrame() {
        if (p1_action_.active_action_id < 0)
            return;

        ++p1_action_.active_action_elapsed_frames;
        if (p1_action_.queued_action_id >= 0) {
            const FollowUpRule *rule = action_lut_.findFollowUpRule(
                p1_action_.active_action_id,
                p1_action_.queued_action_id);
            if (!rule) {
                clearActionState();
                return;
            }

            if (p1_action_.active_action_elapsed_frames >= rule->execute_frame) {
                const int32_t queued_action_id = p1_action_.queued_action_id;
                p1_action_.queued_action_id = -1;
                if (!startActionById(queued_action_id, rule->script_action_id)) {
                    clearActionState();
                }
            }
            return;
        }

        if (p1_action_.script.empty() &&
            !action_lut_.hasPendingFollowUpWindow(
                p1_action_.active_action_id,
                p1_action_.active_action_elapsed_frames)) {
            p1_action_.active_action_id = -1;
            p1_action_.active_action_serial = 0;
            p1_action_.action_start_event_pending = false;
            p1_action_.active_action_elapsed_frames = 0;
        }
    }

    void clearP2RandomAiScript() {
        p2_controller_.clearScript();
    }

    void clearP2ActionState() {
        p2_controller_.resetActions();
    }

    bool startP2RandomAiScript(std::vector<InputFrame> script,
                               int32_t action_id = -1) {
        if (script.empty())
            return false;

        p2_controller_.script = std::move(script);
        p2_controller_.script_index = 0;
        p2_controller_.script_remaining_frames = p2_controller_.script[0].frames;
        p2_controller_.training_action_id = action_id;
        p2_controller_.training_action_elapsed_frames = 0;
        if (action_id > IDLE_ACTION_ID) {
            // scripted P2 沒有 p2_controller_.active_action_id，因此在腳本啟動時自行
            // 配發 serial，讓 Physical Fight 的傷害事件仍可精確歸因。
            ++p2_controller_.next_action_serial;
            if (p2_controller_.next_action_serial == 0)
                ++p2_controller_.next_action_serial;
            p2_controller_.last_started_action_id = action_id;
            p2_controller_.last_started_action_serial = p2_controller_.next_action_serial;
            p2_controller_.last_started_action_age_frames = 0;
            p2_controller_.action_start_event_pending = true;
        }
        joypads_[P2_PORT] = p2_controller_.script[0].state;
        return true;
    }

    bool startP2Action(const CharacterActionTable &actions,
                       int32_t action_id,
                       int32_t cooldown_frames = 0) {
        const auto action_it = actions.find(action_id);
        if (action_it == actions.cend())
            return false;

        std::vector<InputFrame> script = action_it->second;
        if (cooldown_frames > 0)
            script.push_back({ {}, cooldown_frames });
        return startP2RandomAiScript(std::move(script), action_id);
    }

    bool startP2ActionById(int32_t action_id, int32_t script_action_id = -1) {
        kof_env_observation observation {};
        const bool has_observation = getObservation(&observation);
        const bool forward_is_right =
            !has_observation || observation.p1_x >= observation.p2_x;
        const CharacterActionTable *actions = action_lut_.getAction(forward_is_right);
        if (!actions)
            return fail("P2 character action table is missing.");

        const int32_t resolved_script_action_id =
            script_action_id >= 0 ? script_action_id : action_id;
        const auto action_it = actions->find(resolved_script_action_id);
        if (action_it == actions->cend())
            return fail("P2 action id is out of range.");
        if (!startP2RandomAiScript(action_it->second, action_id))
            return fail("P2 action script is empty.");

        p2_controller_.active_action_id = action_id;
        p2_controller_.active_action_elapsed_frames = 0;
        return true;
    }

    bool setP2Action(int32_t action_id) {
        p2_controller_.last_action_accepted = false;
        if (!p2_controller_.actionApiEnabled())
            return fail("P2 action AI is disabled.");
        if (!isPublicActionId(action_id))
            return fail("P2 action id is outside the public action set.");

        if (p2_controller_.active_action_id >= 0) {
            if (action_id == IDLE_ACTION_ID) {
                p2_controller_.last_action_accepted = true;
                return true;
            }

            if (canQueueP2Action(action_id)) {
                p2_controller_.queued_action_id = action_id;
                p2_controller_.last_action_accepted = true;
                return true;
            }

            return true;
        }

        p2_controller_.last_action_accepted = startP2ActionById(action_id);
        return p2_controller_.last_action_accepted;
    }

    void advanceP2ActionLifecycleFrame() {
        if (p2_controller_.active_action_id < 0)
            return;

        ++p2_controller_.active_action_elapsed_frames;
        if (p2_controller_.queued_action_id >= 0) {
            const FollowUpRule *rule = action_lut_.findFollowUpRule(
                p2_controller_.active_action_id,
                p2_controller_.queued_action_id);
            if (!rule) {
                clearP2ActionState();
                joypads_[P2_PORT] = {};
                return;
            }

            if (p2_controller_.active_action_elapsed_frames >= rule->execute_frame) {
                const int32_t queued_action_id = p2_controller_.queued_action_id;
                p2_controller_.queued_action_id = -1;
                if (!startP2ActionById(queued_action_id, rule->script_action_id)) {
                    clearP2ActionState();
                    joypads_[P2_PORT] = {};
                }
            }
            return;
        }

        if (p2_controller_.script.empty() &&
            !action_lut_.hasPendingFollowUpWindow(
                p2_controller_.active_action_id,
                p2_controller_.active_action_elapsed_frames)) {
            p2_controller_.active_action_id = -1;
            p2_controller_.active_action_elapsed_frames = 0;
        }
    }

    void finishP2ActionScriptFrame() {
        if (!p2_controller_.actionApiEnabled() || p2_controller_.script.empty())
            return;

        if (p2_controller_.script_remaining_frames <= 0 &&
            p2_controller_.script_index + 1 >= p2_controller_.script.size()) {
            clearP2RandomAiScript();
            joypads_[P2_PORT] = {};
        }
    }

    bool startP2TrainingAction() {
        kof_env_observation observation {};
        const bool has_observation = getObservation(&observation);
        const bool forward_is_right = !has_observation || observation.p1_x >= observation.p2_x;
        const CharacterActionTable *actions = action_lut_.getAction(forward_is_right);
        if (!actions || actions->empty())
            return false;

        const uint32_t cycle = p2_controller_.training_cycle++;
        switch (p2_controller_.style) {
        case KOF_ENV_P2_STYLE_ONIYAKI:
            return startP2Action(*actions, KYO_ONIYAKI_ACTION_ID);

        case KOF_ENV_P2_STYLE_GUARD: {
            kof_env_joypad_state stand_guard {};
            setBackOn(stand_guard, forward_is_right);
            kof_env_joypad_state crouch_guard = stand_guard;
            crouch_guard.down = 1;

            switch (cycle % 4) {
            case 0:
                return startP2RandomAiScript({
                    { stand_guard, 42 },
                    { {}, 10 },
                }, WALK_BACK_ACTION_ID);
            case 1:
                return startP2RandomAiScript({
                    { crouch_guard, 36 },
                    { {}, 8 },
                }, 3);
            case 2:
                return startP2RandomAiScript({
                    { stand_guard, 18 },
                    { {}, 18 },
                }, WALK_BACK_ACTION_ID);
            default:
                return startP2RandomAiScript({
                    { crouch_guard, 24 },
                    { {}, 14 },
                }, 3);
            }
        }

        case KOF_ENV_P2_STYLE_JUMP_IN:
            return startP2Action(
                *actions,
                cycle % 2 == 0 ? KYO_JUMP_C_ACTION_ID : KYO_JUMP_D_ACTION_ID,
                cycle % 2 == 0 ? 10 : 14);

        case KOF_ENV_P2_STYLE_POKE: {
            const int32_t distance_x = has_observation
                ? std::abs(observation.distance_x)
                : 0;
            if (distance_x > 80)
                return startP2Action(*actions, WALK_FORWARD_ACTION_ID, 4);

            switch (cycle % 4) {
            case 0:
                return startP2Action(*actions, KYO_CROUCH_B_ACTION_ID, 10);
            case 1:
                return startP2Action(*actions, WALK_BACK_ACTION_ID, 8);
            case 2:
                return startP2Action(*actions, STAND_B_ACTION_ID, 12);
            default:
                return startP2Action(*actions, KYO_CROUCH_A_ACTION_ID, 10);
            }
        }

        default:
            return false;
        }
    }

    void advanceP2RandomAiFrame() {
        if (p2_controller_.actionApiEnabled()) {
            if (p2_controller_.script.empty()) {
                joypads_[P2_PORT] = {};
                return;
            }

            if (p2_controller_.script_remaining_frames <= 0) {
                ++p2_controller_.script_index;
                if (p2_controller_.script_index >= p2_controller_.script.size()) {
                    clearP2RandomAiScript();
                    joypads_[P2_PORT] = {};
                    return;
                }
                p2_controller_.script_remaining_frames =
                    p2_controller_.script[p2_controller_.script_index].frames;
            }

            if (p2_controller_.script_remaining_frames > 0) {
                joypads_[P2_PORT] = p2_controller_.script[p2_controller_.script_index].state;
                --p2_controller_.script_remaining_frames;
                ++p2_controller_.training_action_elapsed_frames;
            }
            return;
        }

        if (!p2_controller_.scriptedStyleEnabled()) {
            clearP2RandomAiScript();
            return;
        }

        if (p2_controller_.script.empty()) {
            if (!startP2TrainingAction())
                joypads_[P2_PORT] = {};
            return;
        }

        if (p2_controller_.script_remaining_frames <= 0) {
            ++p2_controller_.script_index;
            if (p2_controller_.script_index >= p2_controller_.script.size()) {
                clearP2RandomAiScript();
                joypads_[P2_PORT] = {};
                return;
            }

            p2_controller_.script_remaining_frames = p2_controller_.script[p2_controller_.script_index].frames;
        }

        if (p2_controller_.script_remaining_frames <= 0)
            return;

        joypads_[P2_PORT] = p2_controller_.script[p2_controller_.script_index].state;
        --p2_controller_.script_remaining_frames;
        ++p2_controller_.training_action_elapsed_frames;
    }

    bool startActionById(int32_t action_id, int32_t script_action_id = -1) {
        kof_env_observation observation {};
        const bool has_observation = getObservation(&observation);
        bool facing_left = false;
        bool has_facing = false;
        size_t ram_size = 0;
        const uint8_t *ram_ptr = systemRam(&ram_size);
        if (ram_ptr && ram_size > 0) {
            const game_memory::GameMemReaderCore mem_reader(ram_ptr, ram_size);
            has_facing = mem_reader.readP1FacingLeft(facing_left);
        }

        const bool p2_is_right = !has_observation || observation.p2_x >= observation.p1_x;
        const bool forward_is_right = has_facing ? !facing_left : p2_is_right;
        const CharacterActionTable *actions = action_lut_.getAction(forward_is_right);
        if (!actions)
            return fail("Character action table is missing.");

        const int32_t resolved_script_action_id =
            script_action_id >= 0 ? script_action_id : action_id;
        const auto action_it = actions->find(resolved_script_action_id);
        if (action_it == actions->cend())
            return fail("Action id is out of range.");

        return startActionScript(
            action_id,
            p1ActionScript(action_id, action_it->second));
    }

    bool setAction(int32_t action_id) {
        p1_action_.last_action_accepted = false;
        if (!isPublicActionId(action_id))
            return fail("Action id is outside the public action set.");

        if (p1_action_.active_action_id >= 0) {
            if (action_id == IDLE_ACTION_ID) {
                p1_action_.last_action_accepted = true;
                return true;
            }

            if (canQueueAction(action_id)) {
                p1_action_.queued_action_id = action_id;
                p1_action_.last_action_accepted = true;
                return true;
            }

            // Keep emulation running while reporting that this action was ignored.
            return true;
        }

        p1_action_.last_action_accepted = startActionById(action_id);
        return p1_action_.last_action_accepted;
    }

    bool runFrames(int32_t frame_count) {
        if (!game_loaded_ || !retro_run_)
            return fail("No loaded game to run.");
        if (frame_count < 0)
            return fail("Frame count cannot be negative.");

        clearStepEvents();
        combat_boundary_in_batch_ = false;
        g_active_runtime = this;
        for (int32_t frame = 0; frame < frame_count; ++frame) {
            if (combat_boundary_in_batch_) {
                // 換局後同一次 runFrames() 的剩餘幀只讓核心完成轉場。
                // 不重啟 P1/P2 腳本，也不產生新 epoch 的半套事件。
                joypads_[P1_PORT] = {};
                joypads_[P2_PORT] = {};
                previous_frame_joypads_ = last_frame_joypads_;
                last_frame_joypads_ = joypads_;
                retro_run_();
                ++engine_frame_index_;
                continue;
            }

            // 每個 emulator frame 的固定順序：先推進並套用輸入，再拍攝
            // before，執行 retro_run，最後以 after 產生事件。事件必須在
            // lifecycle 結束前完成，否則本幀命中會失去正確 action serial。
            if (p1_action_.last_started_action_id >= 0 &&
                p1_action_.last_started_action_age_frames < INT32_MAX) {
                ++p1_action_.last_started_action_age_frames;
            }
            if (p2_controller_.last_started_action_id >= 0 &&
                p2_controller_.last_started_action_age_frames < INT32_MAX) {
                ++p2_controller_.last_started_action_age_frames;
            }

            advanceActionScriptFrame();
            advanceP2RandomAiFrame();
            previous_frame_joypads_ = last_frame_joypads_;
            last_frame_joypads_ = joypads_;
            const bool has_recent_p1_action =
                p1_action_.last_started_action_id > IDLE_ACTION_ID &&
                p1_action_.last_started_action_age_frames <=
                    ACTION_ATTRIBUTION_FALLBACK_FRAMES;
            const int32_t acting_action_id =
                p1_action_.active_action_id > IDLE_ACTION_ID
                    ? p1_action_.active_action_id
                    : (has_recent_p1_action ? p1_action_.last_started_action_id : -1);
            const uint32_t acting_action_serial =
                p1_action_.active_action_id > IDLE_ACTION_ID
                    ? p1_action_.active_action_serial
                    : (has_recent_p1_action ? p1_action_.last_started_action_serial : 0);
            const bool p1_guard_action_active =
                p1_action_.active_action_id == WALK_BACK_ACTION_ID ||
                p1_action_.active_action_id == CROUCH_GUARD_ACTION_ID ||
                p1_action_.active_action_id == STAND_GUARD_ACTION_ID;
            const bool p1_guard_action_was_active =
                previous_p1_guard_action_active_;
            previous_p1_guard_action_active_ = p1_guard_action_active;
            const FrameCombatState before = readFrameCombatState();
            if (p1_action_.action_start_event_pending) {
                appendStepEvent(
                    frame,
                    KOF_ENV_STEP_EVENT_ACTION_STARTED,
                    acting_action_id,
                    acting_action_serial,
                    before.p1_combo,
                    before.p1_combo,
                    0,
                    0,
                    -1,
                    false,
                    false,
                    false,
                    false,
                    before.p1_hit_guard_stop_raw,
                    before.p1_hit_guard_stop_raw,
                    before.p2_hit_guard_stop_raw,
                    before.p2_hit_guard_stop_raw,
                    p1_action_.active_action_elapsed_frames);
                p1_action_.action_start_event_pending = false;
            }
            if (p2_controller_.action_start_event_pending) {
                appendStepEventV5Only(
                    frame,
                    KOF_ENV_STEP_EVENT_ACTION_STARTED,
                    currentP2ActionId(),
                    currentP2ActionSerial(),
                    before.p2_combo,
                    before.p2_combo,
                    0,
                    0,
                    -1,
                    false,
                    false,
                    false,
                    false,
                    before.p1_hit_guard_stop_raw,
                    before.p1_hit_guard_stop_raw,
                    before.p2_hit_guard_stop_raw,
                    before.p2_hit_guard_stop_raw,
                    currentP2ActionElapsedFrames(),
                    0,
                    2,
                    0);
                p2_controller_.action_start_event_pending = false;
            }
            retro_run_();
            const FrameCombatState after = readFrameCombatState();
            detectFrameCombatEvents(
                frame,
                acting_action_id,
                acting_action_serial,
                p1_guard_action_active,
                p1_guard_action_was_active,
                before,
                after);
            finishActionScriptFrame();
            finishP2ActionScriptFrame();
            advanceActionLifecycleFrame();
            if (p2_controller_.actionApiEnabled())
                advanceP2ActionLifecycleFrame();
            ++engine_frame_index_;
        }

        return true;
    }

    bool step(int32_t action_id, int32_t frame_count, kof_env_observation *observation) {
        if (!setAction(action_id))
            return false;
        if (!runFrames(frame_count))
            return false;

        return getObservation(observation);
    }

    bool getObservation(kof_env_observation *observation) const {
        if (!observation)
            return fail("Observation output pointer is null.");

        *observation = {};

        size_t ram_size = 0;
        const uint8_t *ram_ptr = systemRam(&ram_size);
        if (!ram_ptr || ram_size == 0)
            return fail("System RAM is not available.");

        const game_memory::GameMemReaderCore mem_reader(ram_ptr, ram_size);
        observation->round_time = mem_reader.readRoundTime();
        observation->p1_health = mem_reader.readP1Health();
        observation->p2_health = mem_reader.readP2Health();
        observation->p1_power = mem_reader.readP1Power();
        observation->p2_power = mem_reader.readP2Power();
        observation->p1_power_state = mem_reader.readP1PowerState();
        observation->p2_power_state = mem_reader.readP2PowerState();
        observation->p1_advanced_power_value = mem_reader.readP1AdvancedPowerValue();
        observation->p1_advanced_power_stocks = mem_reader.readP1AdvancedPowerStocks();
        observation->p2_advanced_power_value = mem_reader.readP2AdvancedPowerValue();
        observation->p2_advanced_power_stocks = mem_reader.readP2AdvancedPowerStocks();
        observation->p1_stun = mem_reader.readP1Stun();
        observation->p2_stun = mem_reader.readP2Stun();
        observation->p1_combo_count = mem_reader.readP1ComboCount();
        observation->p2_combo_count = mem_reader.readP2ComboCount();

        game_memory::Point p1_position;
        game_memory::Point p2_position;
        observation->p1_has_position = mem_reader.readP1Position(p1_position) ? 1 : 0;
        observation->p2_has_position = mem_reader.readP2Position(p2_position) ? 1 : 0;
        if (observation->p1_has_position) {
            observation->p1_x = p1_position.x;
            observation->p1_y = p1_position.y;
        }
        if (observation->p2_has_position) {
            observation->p2_x = p2_position.x;
            observation->p2_y = p2_position.y;
        }
        if (observation->p1_has_position && observation->p2_has_position) {
            observation->distance_x = observation->p2_x - observation->p1_x;
            observation->distance_y = observation->p2_y - observation->p1_y;
        }

        return true;
    }

    bool p1ReadyForAction() const {
        size_t ram_size = 0;
        const uint8_t *ram_ptr = systemRam(&ram_size);
        if (!ram_ptr || ram_size == 0)
            return false;

        return game_memory::GameMemReaderCore(ram_ptr, ram_size).p1ReadyForAction();
    }

    bool p2ReadyForAction() const {
        size_t ram_size = 0;
        const uint8_t *ram_ptr = systemRam(&ram_size);
        if (!ram_ptr || ram_size == 0)
            return false;

        return game_memory::GameMemReaderCore(ram_ptr, ram_size).p2ReadyForAction();
    }

    uint32_t systemRamSize() const {
        size_t ram_size = 0;
        systemRam(&ram_size);
        return static_cast<uint32_t>(ram_size);
    }

    bool copySystemRam(void *buffer, uint32_t buffer_size) const {
        if (!buffer)
            return fail("System RAM output buffer is null.");

        size_t ram_size = 0;
        const uint8_t *ram_ptr = systemRam(&ram_size);
        if (!ram_ptr || ram_size == 0)
            return fail("System RAM is not available.");
        if (buffer_size < ram_size)
            return fail("System RAM output buffer is too small.");

        std::memcpy(buffer, ram_ptr, ram_size);
        return true;
    }

    bool getHitboxOverlay(int32_t source_width,
                          int32_t,
                          kof_env_hitbox_rect *rects,
                          uint32_t rect_capacity,
                          uint32_t *rect_count,
                          kof_env_hitbox_axis *axes,
                          uint32_t axis_capacity,
                          uint32_t *axis_count) const {
        if (!rect_count || !axis_count)
            return fail("Hitbox count output pointer is null.");

        *rect_count = 0;
        *axis_count = 0;

        size_t ram_size = 0;
        const uint8_t *ram_ptr = systemRam(&ram_size);
        if (!ram_ptr || ram_size == 0)
            return fail("System RAM is not available.");

        const game_memory::HitboxOverlay overlay =
            game_memory::GameMemReaderCore(ram_ptr, ram_size, source_width).getHitboxOverlay();

        *rect_count = static_cast<uint32_t>(overlay.boxes.size());
        *axis_count = static_cast<uint32_t>(overlay.axes.size());

        if ((*rect_count > rect_capacity && rects) || (*axis_count > axis_capacity && axes))
            return fail("Hitbox output buffer is too small.");

        if (rects) {
            const uint32_t copy_count = std::min(*rect_count, rect_capacity);
            for (uint32_t index = 0; index < copy_count; ++index) {
                const game_memory::HitboxRect &box = overlay.boxes[index];
                rects[index] = { box.type, box.owner, box.left, box.top, box.width, box.height };
            }
        }

        if (axes) {
            const uint32_t copy_count = std::min(*axis_count, axis_capacity);
            for (uint32_t index = 0; index < copy_count; ++index) {
                const game_memory::HitboxAxis &axis = overlay.axes[index];
                axes[index] = { axis.x, axis.y };
            }
        }

        return true;
    }

    void captureVideoFrame(const void *data, unsigned width, unsigned height, size_t pitch) {
        if (!data || width == 0 || height == 0 || pitch == 0)
            return;

        if (video_refresh_callback_)
            video_refresh_callback_(data, width, height, pitch, video_refresh_user_data_);
    }

    const char *lastError() const {
        return last_error_.c_str();
    }

    bool environment(unsigned command, void *data) {
        switch (command) {
        case RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY:
            *static_cast<const char **>(data) = paths_.system_directory.c_str();
            return true;
        case RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY:
            *static_cast<const char **>(data) = paths_.save_directory.c_str();
            return true;
        case RETRO_ENVIRONMENT_GET_LOG_INTERFACE:
            static_cast<retro_log_callback *>(data)->log = logPrintf;
            return true;
        case RETRO_ENVIRONMENT_GET_CORE_OPTIONS_VERSION:
            *static_cast<unsigned *>(data) = 2;
            return true;
        case RETRO_ENVIRONMENT_SET_PIXEL_FORMAT:
        {
            const auto format = *static_cast<const retro_pixel_format *>(data);
            if (format != RETRO_PIXEL_FORMAT_RGB565)
                return false;

            return true;
        }
        case RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE:
            *static_cast<bool *>(data) = false;
            return true;
        case RETRO_ENVIRONMENT_GET_VARIABLE:
        {
            auto *variable = static_cast<retro_variable *>(data);
            if (!variable || !variable->key)
                return false;

            if (std::string(variable->key) == "fbneo-cpu-speed-adjust") {
                variable->value = "100%";
                return true;
            }
            if (std::string(variable->key) == "fbneo-frameskip") {
                variable->value = "0";
                return true;
            }
            if (std::string(variable->key) == "fbneo-lightgun-hide-crosshair") {
                variable->value = "enabled";
                return true;
            }
            if (std::string(variable->key) == "fbneo-neogeo-mode" ||
                std::string(variable->key) == "fbneo-neogeo-mode-switch") {
                variable->value = "MVS_JAP";
                return true;
            }
            if (std::string(variable->key) == "fbneo-memcard-mode") {
                variable->value = "disabled";
                return true;
            }
            return false;
        }
        case RETRO_ENVIRONMENT_SET_CORE_OPTIONS_V2:
        case RETRO_ENVIRONMENT_SET_VARIABLES:
        case RETRO_ENVIRONMENT_SET_INPUT_DESCRIPTORS:
        case RETRO_ENVIRONMENT_SET_CONTROLLER_INFO:
        case RETRO_ENVIRONMENT_SET_SUPPORT_ACHIEVEMENTS:
        case RETRO_ENVIRONMENT_SET_MEMORY_MAPS:
            return true;
        default:
            return false;
        }
    }

    int16_t inputState(unsigned port, unsigned device, unsigned, unsigned id) const {
        if (device != RETRO_DEVICE_JOYPAD)
            return 0;

        const kof_env_joypad_state *state = nullptr;
        if (port == P1_PORT)
            state = &joypads_[P1_PORT];
        else if (port == P2_PORT)
            state = &joypads_[P2_PORT];
        else
            return 0;

        switch (id) {
        case RETRO_DEVICE_ID_JOYPAD_UP:
            return state->up ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_DOWN:
            return state->down ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_LEFT:
            return state->left ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_RIGHT:
            return state->right ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_B:
            return state->a ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_A:
            return state->b ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_Y:
            return state->c ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_X:
            return state->d ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_START:
            return state->start ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_SELECT:
            return state->coin ? 1 : 0;
        default:
            return 0;
        }
    }

private:
    bool fail(const char *message) const {
        last_error_ = message ? message : "Unknown error.";
        return false;
    }

    bool cacheStateFile(const std::wstring &path) {
        if (path == cached_state_.path && !cached_state_.data.empty())
            return true;

        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
            return fail("Could not open state file for reading.");

        const std::streampos end = file.tellg();
        if (end <= 0)
            return fail("State file is empty.");

        std::vector<uint8_t> data(static_cast<size_t>(end));
        file.seekg(0, std::ios::beg);
        file.read(
            reinterpret_cast<char *>(data.data()),
            static_cast<std::streamsize>(data.size()));
        if (!file)
            return fail("Could not read complete state file.");

        cached_state_.path = path;
        cached_state_.data = std::move(data);
        return true;
    }

    void clearStateCache() {
        cached_state_.clear();
    }

    void close() {
        unloadGame();

        if (initialized_ && retro_deinit_)
            retro_deinit_();
        initialized_ = false;

        if (library_)
            FreeLibrary(library_);
        library_ = nullptr;
        resetSymbols();

        if (g_active_runtime == this)
            g_active_runtime = nullptr;
    }

    void unloadGame() {
        if (game_loaded_ && retro_unload_game_)
            retro_unload_game_();
        game_loaded_ = false;
        clearStateCache();
        joypads_[P1_PORT] = {};
        joypads_[P2_PORT] = {};
        last_frame_joypads_ = {};
        previous_frame_joypads_ = {};
        previous_p1_guard_action_active_ = false;
        engine_frame_index_ = 0;
        combat_boundary_in_batch_ = false;
        clearActionState();
        clearP2ActionState();
        clearStepEvents();
        combat_events_.advanceEpoch();
    }

    void installCallbacks() {
        retro_set_environment_(environmentCallback);
        retro_set_video_refresh_(videoCallback);
        retro_set_audio_sample_(audioSampleCallback);
        retro_set_audio_sample_batch_(audioBatchCallback);
        retro_set_input_poll_(inputPollCallback);
        retro_set_input_state_(inputStateCallback);
    }

    template <typename T>
    T resolveSymbol(const char *name) {
        static_assert(
            std::is_pointer_v<T> &&
            std::is_function_v<std::remove_pointer_t<T>>,
            "T must be a function pointer type"
        );

        auto *symbol = GetProcAddress(library_, name);
        if (!symbol)
            throw std::runtime_error(name);

        return reinterpret_cast<T>(symbol);
    }

    void loadSymbols() {
        retro_set_environment_ = resolveSymbol<retro_set_environment_t>("retro_set_environment");
        retro_set_video_refresh_ = resolveSymbol<retro_set_video_refresh_t>("retro_set_video_refresh");
        retro_set_audio_sample_ = resolveSymbol<retro_set_audio_sample_t>("retro_set_audio_sample");
        retro_set_audio_sample_batch_ = resolveSymbol<retro_set_audio_sample_batch_t>("retro_set_audio_sample_batch");
        retro_set_input_poll_ = resolveSymbol<retro_set_input_poll_t>("retro_set_input_poll");
        retro_set_input_state_ = resolveSymbol<retro_set_input_state_t>("retro_set_input_state");
        retro_set_controller_port_device_ =
            resolveSymbol<retro_set_controller_port_device_t>("retro_set_controller_port_device");
        retro_init_ = resolveSymbol<retro_init_t>("retro_init");
        retro_deinit_ = resolveSymbol<retro_deinit_t>("retro_deinit");
        retro_reset_ = resolveSymbol<retro_reset_t>("retro_reset");
        retro_load_game_ = resolveSymbol<retro_load_game_t>("retro_load_game");
        retro_unload_game_ = resolveSymbol<retro_unload_game_t>("retro_unload_game");
        retro_run_ = resolveSymbol<retro_run_t>("retro_run");
        retro_get_system_info_ = resolveSymbol<retro_get_system_info_t>("retro_get_system_info");
        retro_get_system_av_info_ = resolveSymbol<retro_get_system_av_info_t>("retro_get_system_av_info");
        retro_serialize_size_ = resolveSymbol<retro_serialize_size_t>("retro_serialize_size");
        retro_serialize_ = resolveSymbol<retro_serialize_t>("retro_serialize");
        retro_unserialize_ = resolveSymbol<retro_unserialize_t>("retro_unserialize");
        retro_get_memory_data_ = resolveSymbol<retro_get_memory_data_t>("retro_get_memory_data");
        retro_get_memory_size_ = resolveSymbol<retro_get_memory_size_t>("retro_get_memory_size");
    }

    void resetSymbols() {
        retro_set_environment_ = nullptr;
        retro_set_video_refresh_ = nullptr;
        retro_set_audio_sample_ = nullptr;
        retro_set_audio_sample_batch_ = nullptr;
        retro_set_input_poll_ = nullptr;
        retro_set_input_state_ = nullptr;
        retro_set_controller_port_device_ = nullptr;
        retro_init_ = nullptr;
        retro_deinit_ = nullptr;
        retro_reset_ = nullptr;
        retro_load_game_ = nullptr;
        retro_unload_game_ = nullptr;
        retro_run_ = nullptr;
        retro_get_system_info_ = nullptr;
        retro_get_system_av_info_ = nullptr;
        retro_serialize_size_ = nullptr;
        retro_serialize_ = nullptr;
        retro_unserialize_ = nullptr;
        retro_get_memory_data_ = nullptr;
        retro_get_memory_size_ = nullptr;
    }

    const uint8_t *systemRam(size_t *size) const {
        if (size)
            *size = 0;
        if (!game_loaded_ || !retro_get_memory_data_ || !retro_get_memory_size_)
            return nullptr;

        auto *ram = static_cast<const uint8_t *>(retro_get_memory_data_(RETRO_MEMORY_SYSTEM_RAM));
        const size_t ram_size = retro_get_memory_size_(RETRO_MEMORY_SYSTEM_RAM);
        if (size)
            *size = ram_size;
        return ram;
    }

    static bool environmentCallback(unsigned command, void *data) {
        return g_active_runtime ? g_active_runtime->environment(command, data) : false;
    }

    static void videoCallback(const void *data, unsigned width, unsigned height, size_t pitch) {
        if (g_active_runtime)
            g_active_runtime->captureVideoFrame(data, width, height, pitch);
    }

    static void audioSampleCallback(int16_t, int16_t) {
    }

    static size_t audioBatchCallback(const int16_t *, size_t frames) {
        return frames;
    }

    static void inputPollCallback() {
    }

    static int16_t inputStateCallback(unsigned port, unsigned device, unsigned index, unsigned id) {
        return g_active_runtime ? g_active_runtime->inputState(port, device, index, id) : 0;
    }

    HMODULE library_ = nullptr;
    bool initialized_ = false;
    bool game_loaded_ = false;
    std::array<kof_env_joypad_state, PLAYER_PORT_COUNT> joypads_ {};
    std::array<kof_env_joypad_state, PLAYER_PORT_COUNT> last_frame_joypads_ {};
    std::array<kof_env_joypad_state, PLAYER_PORT_COUNT> previous_frame_joypads_ {};
    bool previous_p1_guard_action_active_ = false;
    uint64_t engine_frame_index_ = 0;
    bool combat_boundary_in_batch_ = false;
    ActionRuntimeState p1_action_;
    CombatEventState combat_events_;
    P2ControllerState p2_controller_;
    CharacterActionMapLut action_lut_;
    RuntimePaths paths_;
    CachedState cached_state_;
    mutable std::string last_error_;
    kof_env_video_refresh_t video_refresh_callback_ = nullptr;
    void *video_refresh_user_data_ = nullptr;

    retro_set_environment_t retro_set_environment_ = nullptr;
    retro_set_video_refresh_t retro_set_video_refresh_ = nullptr;
    retro_set_audio_sample_t retro_set_audio_sample_ = nullptr;
    retro_set_audio_sample_batch_t retro_set_audio_sample_batch_ = nullptr;
    retro_set_input_poll_t retro_set_input_poll_ = nullptr;
    retro_set_input_state_t retro_set_input_state_ = nullptr;
    retro_set_controller_port_device_t retro_set_controller_port_device_ = nullptr;
    retro_init_t retro_init_ = nullptr;
    retro_deinit_t retro_deinit_ = nullptr;
    retro_reset_t retro_reset_ = nullptr;
    retro_load_game_t retro_load_game_ = nullptr;
    retro_unload_game_t retro_unload_game_ = nullptr;
    retro_run_t retro_run_ = nullptr;
    retro_get_system_info_t retro_get_system_info_ = nullptr;
    retro_get_system_av_info_t retro_get_system_av_info_ = nullptr;
    retro_serialize_size_t retro_serialize_size_ = nullptr;
    retro_serialize_t retro_serialize_ = nullptr;
    retro_unserialize_t retro_unserialize_ = nullptr;
    retro_get_memory_data_t retro_get_memory_data_ = nullptr;
    retro_get_memory_size_t retro_get_memory_size_ = nullptr;
};

FbneoTrainingRuntime *runtimeFromHandle(kof_env_handle handle) {
    return static_cast<FbneoTrainingRuntime *>(handle);
}

} // namespace

extern "C" {

uint32_t kof_env_api_version(void) {
    return KOF_ENV_API_VERSION;
}

uint32_t kof_env_public_action_count(void) {
    return KOF_ENV_PUBLIC_ACTION_COUNT;
}

uint32_t kof_env_action_set_version(void) {
    return KOF_ENV_ACTION_SET_VERSION;
}

uint32_t kof_env_p1_hold_chunk_frames(void) {
    return KOF_ENV_P1_HOLD_CHUNK_FRAMES;
}

kof_env_handle kof_env_create(void) {
    return new FbneoTrainingRuntime();
}

void kof_env_destroy(kof_env_handle handle) {
    delete runtimeFromHandle(handle);
}

int kof_env_load_core(kof_env_handle handle, const wchar_t *core_path) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->loadCore(core_path) ? 1 : 0;
}

int kof_env_load_game(kof_env_handle handle,
                      const wchar_t *game_path,
                      const wchar_t *system_directory,
                      const wchar_t *save_directory) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->loadGame(game_path, system_directory, save_directory) ? 1 : 0;
}

int kof_env_reset(kof_env_handle handle) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->reset() ? 1 : 0;
}

int kof_env_load_state(kof_env_handle handle, const wchar_t *state_path) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->loadState(state_path) ? 1 : 0;
}

int kof_env_save_state(kof_env_handle handle, const wchar_t *state_path) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->saveState(state_path) ? 1 : 0;
}

int kof_env_snapshot_safe(kof_env_handle handle) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->snapshotSafe() ? 1 : 0;
}

void kof_env_set_joypad(kof_env_handle handle, const kof_env_joypad_state *state) {
    if (auto *runtime = runtimeFromHandle(handle))
        runtime->setJoypad(state);
}

void kof_env_set_joypad_for_port(kof_env_handle handle,
                                 unsigned port,
                                 const kof_env_joypad_state *state) {
    if (auto *runtime = runtimeFromHandle(handle))
        runtime->setJoypadForPort(port, state);
}

int kof_env_get_last_joypad_for_port(kof_env_handle handle,
                                     unsigned port,
                                     kof_env_joypad_state *state) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->getLastJoypadForPort(port, state) ? 1 : 0;
}

void kof_env_set_video_refresh(kof_env_handle handle,
                               kof_env_video_refresh_t callback,
                               void *user_data) {
    if (auto *runtime = runtimeFromHandle(handle))
        runtime->setVideoRefresh(callback, user_data);
}

void kof_env_set_p2_random_ai(kof_env_handle handle, int enabled) {
    if (auto *runtime = runtimeFromHandle(handle))
        runtime->setP2RandomAiEnabled(enabled != 0);
}

int kof_env_set_p2_style(kof_env_handle handle, int32_t style) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->setP2Style(style) ? 1 : 0;
}

void kof_env_set_p2_action_ai(kof_env_handle handle, int enabled) {
    if (auto *runtime = runtimeFromHandle(handle))
        runtime->setP2ActionAiEnabled(enabled != 0);
}

int kof_env_set_p2_action(kof_env_handle handle, int32_t action_id) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->setP2Action(action_id) ? 1 : 0;
}

int kof_env_can_queue_p2_action(kof_env_handle handle, int32_t action_id) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->canQueueP2Action(action_id) ? 1 : 0;
}

int kof_env_p2_input_ready(kof_env_handle handle) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->p2InputReady() ? 1 : 0;
}

int kof_env_p2_ready_for_action(kof_env_handle handle) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->p2ReadyForAction() ? 1 : 0;
}

int kof_env_set_action(kof_env_handle handle, int32_t action_id) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->setAction(action_id) ? 1 : 0;
}

int kof_env_can_queue_action(kof_env_handle handle, int32_t action_id) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->canQueueAction(action_id) ? 1 : 0;
}

int kof_env_get_action_status(kof_env_handle handle,
                              kof_env_action_status *status) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->getActionStatus(status) ? 1 : 0;
}

int kof_env_get_strategy_state_v1(kof_env_handle handle,
                                  kof_env_strategy_state_v1 *state) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->getStrategyStateV1(state) ? 1 : 0;
}

int kof_env_get_combat_timing_state_v1(
    kof_env_handle handle,
    kof_env_combat_timing_state_v1 *state) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->getCombatTimingStateV1(state) ? 1 : 0;
}

int kof_env_get_step_events_v1(kof_env_handle handle,
                               kof_env_step_events_v1 *events) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->getStepEventsV1(events) ? 1 : 0;
}

int kof_env_get_step_events_v2(kof_env_handle handle,
                               kof_env_step_events_v2 *events) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->getStepEventsV2(events) ? 1 : 0;
}

int kof_env_get_step_events_v3(kof_env_handle handle,
                               kof_env_step_events_v3 *events) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->getStepEventsV3(events) ? 1 : 0;
}

int kof_env_get_step_events_v4(kof_env_handle handle,
                               kof_env_step_events_v4 *events) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->getStepEventsV4(events) ? 1 : 0;
}

int kof_env_get_step_events_v5(kof_env_handle handle,
                               kof_env_step_events_v5 *events) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->getStepEventsV5(events) ? 1 : 0;
}

int kof_env_get_kyo_move_data_v1(int32_t action_id,
                                 int32_t variant,
                                 kof_env_move_data_v1 *move_data) {
    if (!move_data ||
        move_data->struct_size != sizeof(kof_env_move_data_v1) ||
        move_data->version != KOF_ENV_MOVE_DATA_VERSION_1) {
        return 0;
    }

    const kof98::MoveData *source = kof98::findKyoMoveData(
        action_id,
        static_cast<kof98::MoveVariant>(variant));
    if (!source)
        return 0;

    kof_env_move_data_v1 result {};
    result.struct_size = sizeof(kof_env_move_data_v1);
    result.version = KOF_ENV_MOVE_DATA_VERSION_1;
    result.action_id = source->action_id;
    result.variant = static_cast<int32_t>(source->variant);
    result.move_class = static_cast<int32_t>(source->move_class);
    result.startup_frames = source->startup_frames;
    result.active_frames = source->active_frames;
    result.recovery_frames = source->recovery_frames;
    result.reach_front = source->reach_front;
    result.reach_back = source->reach_back;
    result.movement_forward = source->movement_forward;
    result.attack_y_min = source->attack_y_min;
    result.attack_y_max = source->attack_y_max;
    result.anti_ground_small_jump_y = source->anti_ground_small_jump_y;
    result.anti_ground_normal_jump_y = source->anti_ground_normal_jump_y;
    result.ground_blockstun_frames = source->ground_blockstun_frames;
    result.air_blockstun_frames = source->air_blockstun_frames;
    result.flags = source->flags;
    result.source = static_cast<int32_t>(source->source);
    *move_data = result;
    return 1;
}

int kof_env_run_frames(kof_env_handle handle, int32_t frame_count) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->runFrames(frame_count) ? 1 : 0;
}

int kof_env_step(kof_env_handle handle,
                 int32_t action_id,
                 int32_t frame_count,
                 kof_env_observation *observation) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->step(action_id, frame_count, observation) ? 1 : 0;
}

int kof_env_get_observation(kof_env_handle handle, kof_env_observation *observation) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->getObservation(observation) ? 1 : 0;
}

int kof_env_input_ready(kof_env_handle handle) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->inputReady() ? 1 : 0;
}

int kof_env_p1_ready_for_action(kof_env_handle handle) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->p1ReadyForAction() ? 1 : 0;
}

uint32_t kof_env_system_ram_size(kof_env_handle handle) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime ? runtime->systemRamSize() : 0;
}

int kof_env_copy_system_ram(kof_env_handle handle,
                            void *buffer,
                            uint32_t buffer_size) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->copySystemRam(buffer, buffer_size) ? 1 : 0;
}

int kof_env_get_hitbox_overlay(kof_env_handle handle,
                               int32_t source_width,
                               int32_t source_height,
                               kof_env_hitbox_rect *rects,
                               uint32_t rect_capacity,
                               uint32_t *rect_count,
                               kof_env_hitbox_axis *axes,
                               uint32_t axis_capacity,
                               uint32_t *axis_count) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->getHitboxOverlay(source_width,
                                                source_height,
                                                rects,
                                                rect_capacity,
                                                rect_count,
                                                axes,
                                                axis_capacity,
                                                axis_count) ? 1 : 0;
}

const char *kof_env_last_error(kof_env_handle handle) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime ? runtime->lastError() : "Invalid kof_env handle.";
}

} // extern "C"
