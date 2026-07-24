#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#ifdef KOF_ENV_BUILD
#define KOF_ENV_API __declspec(dllexport)
#else
#define KOF_ENV_API __declspec(dllimport)
#endif
#else
#define KOF_ENV_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void *kof_env_handle;

enum {
    /* Version 3 adds CombatTimingState V1 and Step Events V5. Existing
     * Strategy/Step Events ABI versions remain available unchanged. */
    KOF_ENV_API_VERSION = 3,
    KOF_ENV_PUBLIC_ACTION_COUNT = 29,
    /* Version 2 replaces P1 movement/guard 4+2 scripts with interruptible
     * 4-frame chunks. Public action ids remain unchanged. */
    KOF_ENV_ACTION_SET_VERSION = 2,
    KOF_ENV_P1_HOLD_CHUNK_FRAMES = 4,
};

typedef void (*kof_env_video_refresh_t)(const void *data,
                                        unsigned width,
                                        unsigned height,
                                        size_t pitch,
                                        void *user_data);

typedef struct kof_env_joypad_state {
    uint8_t up;
    uint8_t down;
    uint8_t left;
    uint8_t right;
    uint8_t a;
    uint8_t b;
    uint8_t c;
    uint8_t d;
    uint8_t start;
    uint8_t coin;
} kof_env_joypad_state;

typedef enum kof_env_p2_style {
    KOF_ENV_P2_STYLE_ONIYAKI = 0,
    KOF_ENV_P2_STYLE_GUARD = 1,
    KOF_ENV_P2_STYLE_JUMP_IN = 2,
    KOF_ENV_P2_STYLE_POKE = 3,
    KOF_ENV_P2_STYLE_COUNT = 4,
} kof_env_p2_style;

typedef struct kof_env_observation {
    int32_t round_time;
    int32_t p1_health;
    int32_t p2_health;
    int32_t p1_power;
    int32_t p2_power;
    int32_t p1_power_state;
    int32_t p2_power_state;
    int32_t p1_advanced_power_value;
    int32_t p1_advanced_power_stocks;
    int32_t p2_advanced_power_value;
    int32_t p2_advanced_power_stocks;
    int32_t p1_stun;
    int32_t p2_stun;
    int32_t p1_combo_count;
    int32_t p2_combo_count;
    int32_t p1_x;
    int32_t p1_y;
    int32_t p2_x;
    int32_t p2_y;
    int32_t distance_x;
    int32_t distance_y;
    uint8_t p1_has_position;
    uint8_t p2_has_position;
} kof_env_observation;

typedef struct kof_env_action_status {
    int32_t active_action_id;
    int32_t queued_action_id;
    int32_t last_started_action_id;
    uint8_t action_accepted;
} kof_env_action_status;

enum {
    KOF_ENV_STRATEGY_STATE_VERSION_1 = 1,
    KOF_ENV_COMBAT_TIMING_STATE_VERSION_1 = 1,
};

/* Versioned action/state snapshot used by StrategyV2 observations.
 * p1_status/p2_status are retained as ABI field names, but KOF98 stores the
 * hitbox-slot active mask at base+0x7c. They are not animation, hitstun, or
 * blockstun state enums. */
typedef struct kof_env_strategy_state_v1 {
    uint32_t struct_size;
    uint32_t version;
    int32_t p1_status;
    int32_t p2_status;
    int32_t p1_active_action_id;
    int32_t p1_queued_action_id;
    int32_t p1_last_started_action_id;
    int32_t p1_action_elapsed_frames;
    int32_t p1_action_remaining_frames;
    int32_t p2_active_action_id;
    int32_t p2_queued_action_id;
    int32_t p2_action_elapsed_frames;
    int32_t p2_action_remaining_frames;
    uint8_t p1_input_ready;
    uint8_t p1_ready;
    uint8_t p2_input_ready;
    uint8_t p2_ready;
    uint8_t p1_facing_left;
    uint8_t p2_facing_left;
    uint8_t p2_scripted;
    uint8_t reserved;
} kof_env_strategy_state_v1;

typedef enum kof_env_reaction_kind {
    KOF_ENV_REACTION_NONE = 0,
    KOF_ENV_REACTION_GUARD = 1,
    KOF_ENV_REACTION_HIT = 2,
} kof_env_reaction_kind;

/* StrategyV4 的單一玩家時序快照。
 *
 * input_script_* 只描述 DLL 輸入腳本，不能解讀為角色招式 recovery。
 * reaction_* 來自經接觸證據、E3 特徵與 signed D2:D3 倒數保守追蹤的
 * 受擊/防禦反應。reaction_valid 表示已確認反應類型；
 * reaction_remaining_valid 只有在觀察到 N->N-1 後才成立。
 * actionable/recovery 尚未找到可靠 RAM 或探針 Ground Truth，因此目前
 * valid 固定為 0。用戶端必須先檢查 valid，不能把 0 當成可行動。 */
typedef struct kof_env_player_timing_state_v1 {
    int32_t input_script_remaining;
    int32_t reaction_kind;
    int32_t reaction_remaining;
    int32_t recovery_remaining;
    uint8_t input_script_ready;
    uint8_t reaction_valid;
    uint8_t actionable_valid;
    uint8_t actionable;
    uint8_t recovery_valid;
    uint8_t reaction_remaining_valid;
    uint8_t reserved[2];
} kof_env_player_timing_state_v1;

/* 不含課程、Oracle 或 scenario 身分的通用戰鬥時序。Targeted、
 * Physical、scripted P2 與真人輸入都使用相同資料來源。 */
typedef struct kof_env_combat_timing_state_v1 {
    uint32_t struct_size;
    uint32_t version;
    uint64_t engine_frame;
    uint32_t event_epoch;
    int32_t frame_advantage;
    uint8_t frame_advantage_valid;
    uint8_t reserved[3];
    kof_env_player_timing_state_v1 p1;
    kof_env_player_timing_state_v1 p2;
} kof_env_combat_timing_state_v1;

enum {
    KOF_ENV_STEP_EVENTS_VERSION_1 = 1,
    KOF_ENV_STEP_EVENT_CAPACITY_V1 = 16,
    KOF_ENV_STEP_EVENTS_VERSION_2 = 2,
    KOF_ENV_STEP_EVENT_CAPACITY_V2 = 16,
    KOF_ENV_STEP_EVENTS_VERSION_3 = 3,
    KOF_ENV_STEP_EVENT_CAPACITY_V3 = 16,
    KOF_ENV_STEP_EVENTS_VERSION_4 = 4,
    KOF_ENV_STEP_EVENT_CAPACITY_V4 = 16,
    KOF_ENV_STEP_EVENTS_VERSION_5 = 5,
    KOF_ENV_STEP_EVENT_CAPACITY_V5 = 32,
};

typedef enum kof_env_step_event_type {
    /* P1 的高階 Action 腳本在本幀真正開始執行。 */
    KOF_ENV_STEP_EVENT_ACTION_STARTED = 1,
    /* P1 命中且遊戲的 Combo counter 在本幀增加。 */
    KOF_ENV_STEP_EVENT_COMBO_HIT = 2,
    /* P1 造成傷害，但 Combo counter 尚未增加或不屬於連段。 */
    KOF_ENV_STEP_EVENT_DAMAGE_ONLY = 3,
    /* 遊戲物理上發生格擋；由 Guard Crush delta 或 Guard box 接觸確認。 */
    KOF_ENV_STEP_EVENT_BLOCK_CONTACT = 4,
    /* 舊版 V2 的泛用 P1 受傷事件，保留數值供既有客戶端相容。 */
    KOF_ENV_STEP_EVENT_P1_DAMAGE = 5,
    /* P1 防禦成功但必殺技造成削血。 */
    KOF_ENV_STEP_EVENT_CHIP_DAMAGE = 6,
    /* P1 未成功防禦而受到的正常傷害。 */
    KOF_ENV_STEP_EVENT_CLEAN_HIT = 7,
    /* BLOCK_CONTACT 發生，而且 P1 正以 Action 2/3/4 輸入後或下後。 */
    KOF_ENV_STEP_EVENT_MANUAL_BLOCK_SUCCESS = 8,
    /* 已由額外 RAM 證據確認為 autoguard。沒有手動證據時不會自動歸類
     * 成 AUTO，而是只保留 BLOCK_CONTACT，代表來源 Unknown。 */
    KOF_ENV_STEP_EVENT_AUTO_GUARD = 9,
    /* 已確認格擋後，完整 signed D2:D3 倒數出現 N→N-1。這是比實際
     * 倒數載入晚一幀的確認 telemetry；確反邏輯不可依賴 START。 */
    KOF_ENV_STEP_EVENT_BLOCKSTUN_STARTED = 10,
    /* 已確認格擋與倒數後，signed D2:D3 由 0→-1，且 E3 reaction kind
     * 同幀由 0x20→0x00；P1 在下一個 emulator frame 可再次行動。 */
    KOF_ENV_STEP_EVENT_BLOCKSTUN_ENDED = 11,
} kof_env_step_event_type;

typedef struct kof_env_step_event_v1 {
    int32_t frame_offset;
    int32_t event_type;
    int32_t action_id;
    uint32_t action_serial;
    int32_t combo_before;
    int32_t combo_after;
    int32_t p2_hp_delta;
} kof_env_step_event_v1;

typedef struct kof_env_step_events_v1 {
    uint32_t struct_size;
    uint32_t version;
    uint32_t event_count;
    uint32_t dropped_event_count;
    kof_env_step_event_v1 events[KOF_ENV_STEP_EVENT_CAPACITY_V1];
} kof_env_step_events_v1;

/* V2 逐 emulator frame 記錄接觸當下的戰鬥狀態。V1 繼續保留給舊客戶端。
 *
 * target_airborne_at_event 採用「受擊前」狀態，避免地面角色被打飛後
 * 誤算成對空；target_airborne_after_event 則保留受擊後狀態，供落地
 * 邊界與擊飛分析。原本的 reserved byte 改作 after flag，因此結構尺寸
 * 仍維持 40 bytes，V2 ABI 不變。 */
typedef struct kof_env_step_event_v2 {
    int32_t frame_offset;
    int32_t event_type;
    int32_t action_id;
    uint32_t action_serial;
    int32_t combo_before;
    int32_t combo_after;
    int32_t p1_hp_delta;
    int32_t p2_hp_delta;
    int32_t target_y_at_event;
    uint8_t target_airborne_at_event;
    uint8_t hit_contact;
    uint8_t block_contact;
    uint8_t target_airborne_after_event;
} kof_env_step_event_v2;

typedef struct kof_env_step_events_v2 {
    uint32_t struct_size;
    uint32_t version;
    uint32_t event_count;
    uint32_t dropped_event_count;
    kof_env_step_event_v2 events[KOF_ENV_STEP_EVENT_CAPACITY_V2];
} kof_env_step_events_v2;

/* V3 保留 V2 的全部欄位，另外記錄事件同一 emulator frame 前後的
 * Hit/Guard Stop 原始值。逐幀 trace 顯示它不是剩餘幀數倒數，客戶端
 * 不可直接把它們解讀成 blockstun。action_elapsed_frames_at_event 可用來
 * 對照靜態招式表的 startup/active/recovery 時序。 */
typedef struct kof_env_step_event_v3 {
    int32_t frame_offset;
    int32_t event_type;
    int32_t action_id;
    uint32_t action_serial;
    int32_t combo_before;
    int32_t combo_after;
    int32_t p1_hp_delta;
    int32_t p2_hp_delta;
    int32_t target_y_at_event;
    uint8_t target_airborne_at_event;
    uint8_t hit_contact;
    uint8_t block_contact;
    uint8_t target_airborne_after_event;
    int32_t p1_hit_guard_stop_before;
    int32_t p1_hit_guard_stop_after;
    int32_t p2_hit_guard_stop_before;
    int32_t p2_hit_guard_stop_after;
    int32_t action_elapsed_frames_at_event;
    int32_t expected_blockstun_frames;
    int32_t expected_blockstun_source;
} kof_env_step_event_v3;

typedef struct kof_env_step_events_v3 {
    uint32_t struct_size;
    uint32_t version;
    uint32_t event_count;
    uint32_t dropped_event_count;
    kof_env_step_event_v3 events[KOF_ENV_STEP_EVENT_CAPACITY_V3];
} kof_env_step_events_v3;

/* V4 保留 V3 的完整 telemetry，並加入兩個 transaction 欄位：
 *
 * event_epoch
 *     reset/load state/換局時遞增。不同 epoch 的事件不可互相配對。
 *
 * guard_reaction_serial
 *     每條 guard string 的流水號。同一條防禦的 BLOCK_CONTACT、
 *     MANUAL_BLOCK_SUCCESS、BLOCKSTUN_STARTED/ENDED 都使用相同值；
 *     非防禦事件為 0。attacker action id/serial 只負責招式歸因，未知
 *     raw input 或遊戲內 CPU 可以合法回傳 -1/0。 */
typedef struct kof_env_step_event_v4 {
    int32_t frame_offset;
    int32_t event_type;
    int32_t action_id;
    uint32_t action_serial;
    int32_t combo_before;
    int32_t combo_after;
    int32_t p1_hp_delta;
    int32_t p2_hp_delta;
    int32_t target_y_at_event;
    uint8_t target_airborne_at_event;
    uint8_t hit_contact;
    uint8_t block_contact;
    uint8_t target_airborne_after_event;
    int32_t p1_hit_guard_stop_before;
    int32_t p1_hit_guard_stop_after;
    int32_t p2_hit_guard_stop_before;
    int32_t p2_hit_guard_stop_after;
    int32_t action_elapsed_frames_at_event;
    int32_t expected_blockstun_frames;
    int32_t expected_blockstun_source;
    uint32_t event_epoch;
    uint32_t guard_reaction_serial;
} kof_env_step_event_v4;

typedef struct kof_env_step_events_v4 {
    uint32_t struct_size;
    uint32_t version;
    uint32_t event_count;
    uint32_t dropped_event_count;
    kof_env_step_event_v4 events[KOF_ENV_STEP_EVENT_CAPACITY_V4];
} kof_env_step_events_v4;

/* V5 保留 V4 的事件內容，新增事件的攻擊方/受影響方與絕對引擎幀。
 * source_player/target_player 使用 1、2；0 代表不適用或未知。
 * batch_event_epoch 是取得這批事件時的最新 epoch。事件本身仍保留
 * event_epoch，讓換局前已寫入的合法事件不會被誤接到新回合。 */
typedef struct kof_env_step_event_v5 {
    int32_t frame_offset;
    int32_t event_type;
    int32_t action_id;
    uint32_t action_serial;
    int32_t combo_before;
    int32_t combo_after;
    int32_t p1_hp_delta;
    int32_t p2_hp_delta;
    int32_t target_y_at_event;
    uint8_t target_airborne_at_event;
    uint8_t hit_contact;
    uint8_t block_contact;
    uint8_t target_airborne_after_event;
    int32_t p1_hit_guard_stop_before;
    int32_t p1_hit_guard_stop_after;
    int32_t p2_hit_guard_stop_before;
    int32_t p2_hit_guard_stop_after;
    int32_t action_elapsed_frames_at_event;
    int32_t expected_blockstun_frames;
    int32_t expected_blockstun_source;
    uint32_t event_epoch;
    uint32_t guard_reaction_serial;
    int32_t source_player;
    int32_t target_player;
    uint64_t absolute_engine_frame;
} kof_env_step_event_v5;

typedef struct kof_env_step_events_v5 {
    uint32_t struct_size;
    uint32_t version;
    uint32_t event_count;
    uint32_t dropped_event_count;
    uint32_t batch_event_epoch;
    uint32_t reserved;
    kof_env_step_event_v5 events[KOF_ENV_STEP_EVENT_CAPACITY_V5];
} kof_env_step_events_v5;

enum {
    KOF_ENV_MOVE_DATA_VERSION_1 = 1,
};

/* 唯讀的京招式資料。來源沒有提供或尚未逐幀量測的欄位一律為 -1；
 * 這份資料只供分析與事件標註，不會直接修改輸入、Action Mask 或 Reward。 */
typedef struct kof_env_move_data_v1 {
    uint32_t struct_size;
    uint32_t version;
    int32_t action_id;
    int32_t variant;
    int32_t move_class;
    int32_t startup_frames;
    int32_t active_frames;
    int32_t recovery_frames;
    int32_t reach_front;
    int32_t reach_back;
    int32_t movement_forward;
    int32_t attack_y_min;
    int32_t attack_y_max;
    int32_t anti_ground_small_jump_y;
    int32_t anti_ground_normal_jump_y;
    int32_t ground_blockstun_frames;
    int32_t air_blockstun_frames;
    uint32_t flags;
    int32_t source;
} kof_env_move_data_v1;

typedef enum kof_env_hitbox_type {
    KOF_ENV_HITBOX_UNDEFINED = 0,
    KOF_ENV_HITBOX_ATTACK = 1,
    KOF_ENV_HITBOX_VULNERABILITY = 2,
    KOF_ENV_HITBOX_PROJECTILE_VULNERABILITY = 3,
    KOF_ENV_HITBOX_PROJECTILE_ATTACK = 4,
    KOF_ENV_HITBOX_PUSH = 5,
    KOF_ENV_HITBOX_GUARD = 6,
} kof_env_hitbox_type;

typedef struct kof_env_hitbox_rect {
    int32_t type;
    int32_t owner;
    int32_t left;
    int32_t top;
    int32_t width;
    int32_t height;
} kof_env_hitbox_rect;

typedef struct kof_env_hitbox_axis {
    int32_t x;
    int32_t y;
} kof_env_hitbox_axis;

KOF_ENV_API kof_env_handle kof_env_create(void);
KOF_ENV_API void kof_env_destroy(kof_env_handle handle);

KOF_ENV_API uint32_t kof_env_api_version(void);
KOF_ENV_API uint32_t kof_env_public_action_count(void);
KOF_ENV_API uint32_t kof_env_action_set_version(void);
KOF_ENV_API uint32_t kof_env_p1_hold_chunk_frames(void);

KOF_ENV_API int kof_env_load_core(kof_env_handle handle, const wchar_t *core_path);
KOF_ENV_API int kof_env_load_game(kof_env_handle handle,
                                  const wchar_t *game_path,
                                  const wchar_t *system_directory,
                                  const wchar_t *save_directory);
KOF_ENV_API int kof_env_reset(kof_env_handle handle);
KOF_ENV_API int kof_env_load_state(kof_env_handle handle, const wchar_t *state_path);
KOF_ENV_API int kof_env_save_state(kof_env_handle handle, const wchar_t *state_path);
KOF_ENV_API int kof_env_snapshot_safe(kof_env_handle handle);

KOF_ENV_API void kof_env_set_joypad(kof_env_handle handle, const kof_env_joypad_state *state);
KOF_ENV_API void kof_env_set_joypad_for_port(kof_env_handle handle,
                                             unsigned port,
                                             const kof_env_joypad_state *state);
KOF_ENV_API int kof_env_get_last_joypad_for_port(kof_env_handle handle,
                                                 unsigned port,
                                                 kof_env_joypad_state *state);
KOF_ENV_API void kof_env_set_video_refresh(kof_env_handle handle,
                                           kof_env_video_refresh_t callback,
                                           void *user_data);
KOF_ENV_API void kof_env_set_p2_random_ai(kof_env_handle handle, int enabled);
KOF_ENV_API int kof_env_set_p2_style(kof_env_handle handle, int32_t style);
KOF_ENV_API void kof_env_set_p2_action_ai(kof_env_handle handle, int enabled);
KOF_ENV_API int kof_env_set_p2_action(kof_env_handle handle, int32_t action_id);
KOF_ENV_API int kof_env_can_queue_p2_action(kof_env_handle handle, int32_t action_id);
KOF_ENV_API int kof_env_p2_input_ready(kof_env_handle handle);
KOF_ENV_API int kof_env_p2_ready_for_action(kof_env_handle handle);
KOF_ENV_API int kof_env_set_action(kof_env_handle handle, int32_t action_id);
KOF_ENV_API int kof_env_can_queue_action(kof_env_handle handle, int32_t action_id);
KOF_ENV_API int kof_env_get_action_status(kof_env_handle handle,
                                          kof_env_action_status *status);
KOF_ENV_API int kof_env_get_strategy_state_v1(kof_env_handle handle,
                                              kof_env_strategy_state_v1 *state);
KOF_ENV_API int kof_env_get_combat_timing_state_v1(
    kof_env_handle handle,
    kof_env_combat_timing_state_v1 *state);
KOF_ENV_API int kof_env_get_step_events_v1(kof_env_handle handle,
                                           kof_env_step_events_v1 *events);
KOF_ENV_API int kof_env_get_step_events_v2(kof_env_handle handle,
                                           kof_env_step_events_v2 *events);
KOF_ENV_API int kof_env_get_step_events_v3(kof_env_handle handle,
                                           kof_env_step_events_v3 *events);
KOF_ENV_API int kof_env_get_step_events_v4(kof_env_handle handle,
                                           kof_env_step_events_v4 *events);
KOF_ENV_API int kof_env_get_step_events_v5(kof_env_handle handle,
                                           kof_env_step_events_v5 *events);
KOF_ENV_API int kof_env_get_kyo_move_data_v1(int32_t action_id,
                                             int32_t variant,
                                             kof_env_move_data_v1 *move_data);
KOF_ENV_API int kof_env_run_frames(kof_env_handle handle, int32_t frame_count);
KOF_ENV_API int kof_env_step(kof_env_handle handle,
                             int32_t action_id,
                             int32_t frame_count,
                             kof_env_observation *observation);

KOF_ENV_API int kof_env_get_observation(kof_env_handle handle,
                                        kof_env_observation *observation);
KOF_ENV_API int kof_env_input_ready(kof_env_handle handle);
KOF_ENV_API int kof_env_p1_ready_for_action(kof_env_handle handle);
KOF_ENV_API uint32_t kof_env_system_ram_size(kof_env_handle handle);
KOF_ENV_API int kof_env_copy_system_ram(kof_env_handle handle,
                                        void *buffer,
                                        uint32_t buffer_size);
KOF_ENV_API int kof_env_get_hitbox_overlay(kof_env_handle handle,
                                           int32_t source_width,
                                           int32_t source_height,
                                           kof_env_hitbox_rect *rects,
                                           uint32_t rect_capacity,
                                           uint32_t *rect_count,
                                           kof_env_hitbox_axis *axes,
                                           uint32_t axis_capacity,
                                           uint32_t *axis_count);
KOF_ENV_API const char *kof_env_last_error(kof_env_handle handle);

#ifdef __cplusplus
}
#endif
