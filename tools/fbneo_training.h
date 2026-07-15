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

KOF_ENV_API int kof_env_load_core(kof_env_handle handle, const wchar_t *core_path);
KOF_ENV_API int kof_env_load_game(kof_env_handle handle,
                                  const wchar_t *game_path,
                                  const wchar_t *system_directory,
                                  const wchar_t *save_directory);
KOF_ENV_API int kof_env_reset(kof_env_handle handle);
KOF_ENV_API int kof_env_load_state(kof_env_handle handle, const wchar_t *state_path);
KOF_ENV_API int kof_env_save_state(kof_env_handle handle, const wchar_t *state_path);

KOF_ENV_API void kof_env_set_joypad(kof_env_handle handle, const kof_env_joypad_state *state);
KOF_ENV_API void kof_env_set_joypad_for_port(kof_env_handle handle,
                                             unsigned port,
                                             const kof_env_joypad_state *state);
KOF_ENV_API void kof_env_set_video_refresh(kof_env_handle handle,
                                           kof_env_video_refresh_t callback,
                                           void *user_data);
KOF_ENV_API void kof_env_set_p2_random_ai(kof_env_handle handle, int enabled);
KOF_ENV_API int kof_env_set_action(kof_env_handle handle, int32_t action_id);
KOF_ENV_API int kof_env_run_frames(kof_env_handle handle, int32_t frame_count);
KOF_ENV_API int kof_env_step(kof_env_handle handle,
                             int32_t action_id,
                             int32_t frame_count,
                             kof_env_observation *observation);

KOF_ENV_API int kof_env_get_observation(kof_env_handle handle,
                                        kof_env_observation *observation);
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
