#include <windows.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "libretro.h"

namespace
{
std::filesystem::path g_systemDirectory;
std::filesystem::path g_saveDirectory;
std::vector<std::string> g_biosChoices;
bool g_sawBiosOption = false;
bool g_sawCoreOptions = false;

using retro_set_environment_t = void (*)(retro_environment_t);
using retro_set_video_refresh_t = void (*)(retro_video_refresh_t);
using retro_set_audio_sample_t = void (*)(retro_audio_sample_t);
using retro_set_audio_sample_batch_t = void (*)(retro_audio_sample_batch_t);
using retro_set_input_poll_t = void (*)(retro_input_poll_t);
using retro_set_input_state_t = void (*)(retro_input_state_t);
using retro_init_t = void (*)();
using retro_deinit_t = void (*)();
using retro_get_system_info_t = void (*)(retro_system_info*);
using retro_load_game_t = bool (*)(const retro_game_info*);
using retro_unload_game_t = void (*)();

std::string narrow(const std::filesystem::path& path) {
    return path.u8string();
}

void log_printf(enum retro_log_level level, const char* fmt, ...) {
    const char* prefix = "info";
    switch (level)
    {
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

    std::fprintf(stderr, "[core:%s] ", prefix);

    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
}

bool capture_core_options_v2(const retro_core_options_v2* options) {
    g_sawCoreOptions = true;

    if (!options || !options->definitions)
        return true;

    for (const retro_core_option_v2_definition* def = options->definitions; def->key; ++def)
    {
        if (std::string(def->key) != "neocd_bios")
            continue;

        g_sawBiosOption = true;
        for (size_t i = 0; i < RETRO_NUM_CORE_OPTION_VALUES_MAX && def->values[i].value; ++i)
            g_biosChoices.emplace_back(def->values[i].value);
    }

    return true;
}

bool capture_legacy_variables(const retro_variable* variables) {
    if (!variables)
        return true;

    for (const retro_variable* var = variables; var->key; ++var)
    {
        if (std::string(var->key) != "neocd_bios")
            continue;

        g_sawBiosOption = true;
        if (!var->value)
            return true;

        std::string value = var->value;
        const std::string prefix = "BIOS Select; ";
        if (value.rfind(prefix, 0) == 0)
            value.erase(0, prefix.size());

        size_t start = 0;
        while (start <= value.size())
        {
            const size_t separator = value.find('|', start);
            g_biosChoices.emplace_back(value.substr(start, separator - start));
            if (separator == std::string::npos)
                break;
            start = separator + 1;
        }
    }

    return true;
}

bool environment(unsigned cmd, void* data) {
    switch (cmd)
    {
    case RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY:
    {
        static std::string systemDirectory;
        systemDirectory = narrow(g_systemDirectory);
        *static_cast<const char**>(data) = systemDirectory.c_str();
        return true;
    }
    case RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY:
    {
        static std::string saveDirectory;
        saveDirectory = narrow(g_saveDirectory);
        *static_cast<const char**>(data) = saveDirectory.c_str();
        return true;
    }
    case RETRO_ENVIRONMENT_GET_LOG_INTERFACE:
    {
        auto* callback = static_cast<retro_log_callback*>(data);
        callback->log = log_printf;
        return true;
    }
    case RETRO_ENVIRONMENT_GET_CORE_OPTIONS_VERSION:
        *static_cast<unsigned*>(data) = 2;
        return true;
    case RETRO_ENVIRONMENT_SET_PIXEL_FORMAT:
    {
        const auto format = *static_cast<const retro_pixel_format*>(data);
        return format == RETRO_PIXEL_FORMAT_RGB565;
    }
    case RETRO_ENVIRONMENT_SET_CORE_OPTIONS_V2:
        return capture_core_options_v2(static_cast<const retro_core_options_v2*>(data));
    case RETRO_ENVIRONMENT_SET_VARIABLES:
        return capture_legacy_variables(static_cast<const retro_variable*>(data));
    default:
        return false;
    }
}

void video_refresh(const void*, unsigned, unsigned, size_t) {
}

void audio_sample(int16_t, int16_t) {
}

size_t audio_sample_batch(const int16_t*, size_t frames) {
    return frames;
}

void input_poll() {
}

int16_t input_state(unsigned, unsigned, unsigned, unsigned) {
    return 0;
}

template <typename T>
T load_symbol(HMODULE module, const char* name) {
    auto* symbol = reinterpret_cast<T>(GetProcAddress(module, name));
    if (!symbol)
        std::cerr << "Missing export: " << name << "\n";
    return symbol;
}

std::filesystem::path absolute_or_current(const char* arg) {
    std::filesystem::path path(arg);
    if (path.is_relative())
        path = std::filesystem::current_path() / path;
    return std::filesystem::weakly_canonical(path);
}
}

int main(int argc, char** argv) {
    const auto cwd = std::filesystem::current_path();
    const auto corePath = argc > 1 ? absolute_or_current(argv[1]) : absolute_or_current("build-vs2026-x64/Debug/neocd_libretro.dll");
    g_systemDirectory = argc > 2 ? absolute_or_current(argv[2]) : absolute_or_current("system");
    g_saveDirectory = argc > 3 ? absolute_or_current(argv[3]) : absolute_or_current("saves");
    const auto contentPath = argc > 4 ? absolute_or_current(argv[4]) : absolute_or_current("roms/neocd/_bios_probe_missing.cue");

    std::filesystem::create_directories(g_systemDirectory / "neocd");
    std::filesystem::create_directories(g_saveDirectory);

    std::wcout << L"Core: " << corePath.wstring() << L"\n";
    std::wcout << L"System directory: " << g_systemDirectory.wstring() << L"\n";
    std::wcout << L"NeoCD BIOS directory: " << (g_systemDirectory / "neocd").wstring() << L"\n";
    std::wcout << L"Load-game probe path: " << contentPath.wstring() << L"\n";

    HMODULE core = LoadLibraryW(corePath.c_str());
    if (!core)
    {
        std::cerr << "LoadLibrary failed. Win32 error: " << GetLastError() << "\n";
        return EXIT_FAILURE;
    }

    auto retro_set_environment = load_symbol<retro_set_environment_t>(core, "retro_set_environment");
    auto retro_set_video_refresh = load_symbol<retro_set_video_refresh_t>(core, "retro_set_video_refresh");
    auto retro_set_audio_sample = load_symbol<retro_set_audio_sample_t>(core, "retro_set_audio_sample");
    auto retro_set_audio_sample_batch = load_symbol<retro_set_audio_sample_batch_t>(core, "retro_set_audio_sample_batch");
    auto retro_set_input_poll = load_symbol<retro_set_input_poll_t>(core, "retro_set_input_poll");
    auto retro_set_input_state = load_symbol<retro_set_input_state_t>(core, "retro_set_input_state");
    auto retro_get_system_info = load_symbol<retro_get_system_info_t>(core, "retro_get_system_info");
    auto retro_init = load_symbol<retro_init_t>(core, "retro_init");
    auto retro_deinit = load_symbol<retro_deinit_t>(core, "retro_deinit");
    auto retro_load_game = load_symbol<retro_load_game_t>(core, "retro_load_game");
    auto retro_unload_game = load_symbol<retro_unload_game_t>(core, "retro_unload_game");

    if (!retro_set_environment || !retro_set_video_refresh || !retro_set_audio_sample ||
        !retro_set_audio_sample_batch || !retro_set_input_poll || !retro_set_input_state ||
        !retro_get_system_info || !retro_init || !retro_deinit || !retro_load_game || !retro_unload_game)
    {
        FreeLibrary(core);
        return EXIT_FAILURE;
    }

    retro_system_info info{};
    retro_get_system_info(&info);
    std::cout << "Core name: " << (info.library_name ? info.library_name : "(unknown)") << "\n";
    std::cout << "Core version: " << (info.library_version ? info.library_version : "(unknown)") << "\n";
    std::cout << "Valid extensions: " << (info.valid_extensions ? info.valid_extensions : "(unknown)") << "\n";

    retro_set_environment(environment);
    retro_set_video_refresh(video_refresh);
    retro_set_audio_sample(audio_sample);
    retro_set_audio_sample_batch(audio_sample_batch);
    retro_set_input_poll(input_poll);
    retro_set_input_state(input_state);

    retro_init();

    const std::string contentPathString = narrow(contentPath);
    retro_game_info game{};
    game.path = contentPathString.c_str();
    const bool loaded = retro_load_game(&game);
    if (loaded)
        retro_unload_game();

    retro_deinit();
    FreeLibrary(core);

    if (!g_sawCoreOptions)
        std::cout << "Core options v2 were not set by the core.\n";

    if (!g_sawBiosOption || g_biosChoices.empty())
    {
        std::cout << "BIOS status: NOT FOUND\n";
        std::cout << "Load-game result: FAILED before BIOS could load\n";
        std::cout << "Put a supported BIOS file in: " << narrow(g_systemDirectory / "neocd") << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "BIOS status: FOUND\n";
    for (const auto& choice : g_biosChoices)
        std::cout << "  - " << choice << "\n";

    if (loaded)
        std::cout << "Load-game result: BIOS and CD image loaded successfully\n";
    else
        std::cout << "Load-game result: BIOS passed; CD image did not load at probe path\n";

    return EXIT_SUCCESS;
}
