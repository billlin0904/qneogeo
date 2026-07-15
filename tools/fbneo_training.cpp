#define KOF_ENV_BUILD
#include "fbneo_training.h"

#include <windows.h>

#include <algorithm>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../gamememreadercore.h"
#include "libretro.h"

namespace {

using retro_set_environment_t = void (*)(retro_environment_t);
using retro_set_video_refresh_t = void (*)(retro_video_refresh_t);
using retro_set_audio_sample_t = void (*)(retro_audio_sample_t);
using retro_set_audio_sample_batch_t = void (*)(retro_audio_sample_batch_t);
using retro_set_input_poll_t = void (*)(retro_input_poll_t);
using retro_set_input_state_t = void (*)(retro_input_state_t);
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

enum class CharacterID {
    Kyo,
};

using CharacterActionTable = std::map<int32_t, std::vector<InputFrame>>;

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
    kof_env_joypad_state crouch_a = down();
    crouch_a.a = 1;
    kof_env_joypad_state crouch_b = down();
    crouch_b.b = 1;
    kof_env_joypad_state crouch_c = down();
    crouch_c.c = 1;
    kof_env_joypad_state crouch_d = down();
    crouch_d.d = 1;

    kof_env_joypad_state qcf_a = down_forward();
    qcf_a.a = 1;
    kof_env_joypad_state qcf_c = down_forward();
    qcf_c.c = 1;
    kof_env_joypad_state hcb_c = back();
    hcb_c.c = 1;
    kof_env_joypad_state hcb_d = back();
    hcb_d.d = 1;
    kof_env_joypad_state dp_a = down_forward();
    dp_a.a = 1;
    kof_env_joypad_state red_kick_b = down_back();
    red_kick_b.b = 1;
    kof_env_joypad_state super_a = forward();
    super_a.a = 1;
    kof_env_joypad_state jump_forward_c = forward();
    jump_forward_c.c = 1;
    kof_env_joypad_state jump_forward_d = forward();
    jump_forward_d.d = 1;
    kof_env_joypad_state forward_b = forward();
    forward_b.b = 1;

    CharacterActionTable kyo {
        { 0, { { {}, 6 } } },
        { 1, simpleAction(forward()) },
        { 2, simpleAction(back()) },
        { 3, simpleAction(crouch_back) },
        { 4, simpleAction(back()) },
        { 5, simpleAction(up_forward()) },
        { 6, simpleAction(neutral_a) },
        { 7, simpleAction(neutral_b) },
        { 8, simpleAction(neutral_c) },
        { 9, simpleAction(neutral_d) },
        { 10, simpleAction(crouch_a) },
        { 11, simpleAction(crouch_b) },
        { 12, simpleAction(crouch_c) },
        { 13, simpleAction(crouch_d) },
        { 14, {
            { down(), 2 },
            { down_forward(), 2 },
            { qcf_a, 2 },
            { {}, 4 },
        } },
        { 15, {
            { {}, 9 },
            { down(), 2 },
            { down_forward(), 2 },
            { qcf_c, 4 },
            { {}, 18 },
            { forward(), 2 },
            { down_forward(), 2 },
            { down(), 2 },
            { down_back(), 2 },
            { hcb_c, 4 },
            { {}, 16 },
            { qcf_c, 4 },
            { {}, 10 },
        } },
        { 16, {
            { {}, 9 },
            { forward(), 2 },
            { down_forward(), 2 },
            { down(), 2 },
            { down_back(), 2 },
            { hcb_d, 4 },
            { {}, 12 },
        } },
        { 17, {
            { forward(), 2 },
            { down(), 2 },
            { dp_a, 4 },
            { {}, 8 },
        } },
        { 18, {
            { back(), 2 },
            { down(), 2 },
            { red_kick_b, 4 },
            { {}, 8 },
        } },
        { 19, {
            { down(), 2 },
            { down_back(), 2 },
            { back(), 2 },
            { down_back(), 2 },
            { down(), 2 },
            { down_forward(), 2 },
            { super_a, 5 },
            { {}, 12 },
        } },
        { 20, {
            { down(), 2 },
            { down_forward(), 2 },
            { forward(), 2 },
            { down(), 2 },
            { down_forward(), 2 },
            { super_a, 5 },
            { {}, 12 },
        } },
        { 21, {
            { up_forward(), 2 },
            { forward(), 12 },
            { jump_forward_c, 5 },
            { {}, 18 },
        } },
        { 22, {
            { up_forward(), 2 },
            { forward(), 12 },
            { jump_forward_d, 5 },
            { {}, 10 },
        } },
        { 23, {
            { forward_b, 5 },
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

private:
	CharacterID current_character_ = CharacterID::Kyo;
	std::map<CharacterID, std::pair<CharacterActionTable, CharacterActionTable>> lut_;
};

class FbneoTrainingRuntime {
public:
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

        game_path_utf8_ = absoluteUtf8Path(game_path);
        system_directory_utf8_ = absoluteUtf8Path(system_directory);
        save_directory_utf8_ = absoluteUtf8Path(save_directory);
        if (game_path_utf8_.empty())
            return fail("Game path is empty.");

        if (!initialized_) {
            g_active_runtime = this;
            installCallbacks();
            retro_init_();
            initialized_ = true;
        }

        retro_game_info info {};
        info.path = game_path_utf8_.c_str();
        info.data = nullptr;
        info.size = 0;
        info.meta = nullptr;

        if (!retro_load_game_(&info))
            return fail("FBNeo could not load game content.");

        game_loaded_ = true;
        return true;
    }

    bool reset() {
        if (!game_loaded_ || !retro_reset_)
            return fail("No loaded game to reset.");

        joypad_ = {};
        clearActionScript();
        retro_reset_();
        return true;
    }

    bool loadState(const wchar_t *state_path) {
        if (!game_loaded_ || !retro_unserialize_)
            return fail("No loaded game for state load.");

        const std::wstring path = absoluteWidePath(state_path);
        std::ifstream file(path, std::ios::binary);
        if (!file)
            return fail("Could not open state file for reading.");

        std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (data.empty())
            return fail("State file is empty.");

        if (!retro_unserialize_(data.data(), data.size()))
            return fail("Core rejected state data.");

        joypad_ = {};
        clearActionScript();
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
        return static_cast<bool>(file);
    }

    void setJoypad(const kof_env_joypad_state *state) {
        joypad_ = state ? *state : kof_env_joypad_state {};
    }

    void setVideoRefresh(kof_env_video_refresh_t callback, void *user_data) {
        video_refresh_callback_ = callback;
        video_refresh_user_data_ = user_data;
    }

    void clearActionScript() {
        active_script_.clear();
        active_script_index_ = 0;
        active_script_remaining_frames_ = 0;
    }

    bool startActionScript(std::vector<InputFrame> script) {
        if (script.empty())
            return fail("Action script is empty.");

        active_script_ = std::move(script);
        active_script_index_ = 0;
        active_script_remaining_frames_ = active_script_[0].frames;
        joypad_ = active_script_[0].state;
        return true;
    }

    void advanceActionScriptFrame() {
        if (active_script_.empty())
            return;

        if (active_script_remaining_frames_ <= 0) {
            ++active_script_index_;
            if (active_script_index_ >= active_script_.size()) {
                clearActionScript();
                joypad_ = {};
                return;
            }

            active_script_remaining_frames_ = active_script_[active_script_index_].frames;
        }

        if (active_script_remaining_frames_ <= 0)
            return;

        joypad_ = active_script_[active_script_index_].state;
        --active_script_remaining_frames_;
    }

    void finishActionScriptFrame() {
        if (active_script_.empty())
            return;

        if (active_script_remaining_frames_ <= 0 &&
            active_script_index_ + 1 >= active_script_.size()) {
            clearActionScript();
            joypad_ = {};
        }
    }

    bool setAction(int32_t action_id) {
        if (!active_script_.empty())
            return true;

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

        const auto action_it = actions->find(action_id);
        if (action_it == actions->cend())
            return fail("Action id is out of range.");

        return startActionScript(action_it->second);
    }

    bool runFrames(int32_t frame_count) {
        if (!game_loaded_ || !retro_run_)
            return fail("No loaded game to run.");
        if (frame_count < 0)
            return fail("Frame count cannot be negative.");

        g_active_runtime = this;
        for (int32_t frame = 0; frame < frame_count; ++frame) {
            advanceActionScriptFrame();
            retro_run_();
            finishActionScriptFrame();
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
            *static_cast<const char **>(data) = system_directory_utf8_.c_str();
            return true;
        case RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY:
            *static_cast<const char **>(data) = save_directory_utf8_.c_str();
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
                variable->value = "shared";
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
        if (port != 0 || device != RETRO_DEVICE_JOYPAD)
            return 0;

        switch (id) {
        case RETRO_DEVICE_ID_JOYPAD_UP:
            return joypad_.up ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_DOWN:
            return joypad_.down ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_LEFT:
            return joypad_.left ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_RIGHT:
            return joypad_.right ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_B:
            return joypad_.a ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_A:
            return joypad_.b ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_Y:
            return joypad_.c ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_X:
            return joypad_.d ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_START:
            return joypad_.start ? 1 : 0;
        case RETRO_DEVICE_ID_JOYPAD_SELECT:
            return joypad_.coin ? 1 : 0;
        default:
            return 0;
        }
    }

private:
    bool fail(const char *message) const {
        last_error_ = message ? message : "Unknown error.";
        return false;
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
        joypad_ = {};
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
    kof_env_joypad_state joypad_ {};
    std::vector<InputFrame> active_script_;
    size_t active_script_index_ = 0;
    int32_t active_script_remaining_frames_ = 0;
    CharacterActionMapLut action_lut_;
    std::string game_path_utf8_;
    std::string system_directory_utf8_;
    std::string save_directory_utf8_;
    mutable std::string last_error_;
    kof_env_video_refresh_t video_refresh_callback_ = nullptr;
    void *video_refresh_user_data_ = nullptr;

    retro_set_environment_t retro_set_environment_ = nullptr;
    retro_set_video_refresh_t retro_set_video_refresh_ = nullptr;
    retro_set_audio_sample_t retro_set_audio_sample_ = nullptr;
    retro_set_audio_sample_batch_t retro_set_audio_sample_batch_ = nullptr;
    retro_set_input_poll_t retro_set_input_poll_ = nullptr;
    retro_set_input_state_t retro_set_input_state_ = nullptr;
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

void kof_env_set_joypad(kof_env_handle handle, const kof_env_joypad_state *state) {
    if (auto *runtime = runtimeFromHandle(handle))
        runtime->setJoypad(state);
}

void kof_env_set_video_refresh(kof_env_handle handle,
                               kof_env_video_refresh_t callback,
                               void *user_data) {
    if (auto *runtime = runtimeFromHandle(handle))
        runtime->setVideoRefresh(callback, user_data);
}

int kof_env_set_action(kof_env_handle handle, int32_t action_id) {
    auto *runtime = runtimeFromHandle(handle);
    return runtime && runtime->setAction(action_id) ? 1 : 0;
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
