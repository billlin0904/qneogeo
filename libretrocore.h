#pragma once

#include "emulatorview.h"
#include "wasapiaudio.h"

#include <QByteArray>
#include <QLibrary>
#include <QObject>
#include <QTimer>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "libretro.h"
#include "samplerate.h"

class LibretroCore final : public QObject
{
    Q_OBJECT

public:
    enum XInputControl {
        XInputNone = 0,
        XInputDpadUp,
        XInputDpadDown,
        XInputDpadLeft,
        XInputDpadRight,
        XInputStart,
        XInputBack,
        XInputA,
        XInputB,
        XInputX,
        XInputY,
        XInputLeftShoulder,
        XInputRightShoulder,
        XInputLeftTrigger,
        XInputRightTrigger,
        XInputLeftThumbUp,
        XInputLeftThumbDown,
        XInputLeftThumbLeft,
        XInputLeftThumbRight
    };

    explicit LibretroCore(EmulatorView *videoOutput, QObject *parent = nullptr);
    ~LibretroCore() override;

    bool loadCore(const QString &corePath);
    bool startGame(const QString &contentPath, const QString &systemDirectory, const QString &saveDirectory);
    void stop();
    void setPaused(bool paused);
    bool isPaused() const;
    bool isGameLoaded() const;
    bool saveState(const QString &statePath);
    bool loadState(const QString &statePath);

    int keyBinding(unsigned retroButtonId) const;
    void setKeyBinding(unsigned retroButtonId, int key);
    std::array<int, 16> keyBindings() const;
    void setKeyBindings(const std::array<int, 16> &bindings);
    int xinputBinding(unsigned retroButtonId) const;
    void setXInputBinding(unsigned retroButtonId, int control);
    std::array<int, 16> xinputBindings() const;
    void setXInputBindings(const std::array<int, 16> &bindings);
    void loadInputConfiguration(const QString &iniPath);
    void saveInputConfiguration(const QString &iniPath) const;
    static QString xinputControlName(int control);
    static int firstPressedXInputControl(unsigned userIndex = 0);
    QString lastError() const;

signals:
    void pausedChanged(bool paused);

private slots:
    void runFrame();

protected:
    bool eventFilter(QObject *watched, QEvent *event) override;

private:
    using retro_set_environment_t = void (*)(retro_environment_t);
    using retro_set_video_refresh_t = void (*)(retro_video_refresh_t);
    using retro_set_audio_sample_t = void (*)(retro_audio_sample_t);
    using retro_set_audio_sample_batch_t = void (*)(retro_audio_sample_batch_t);
    using retro_set_input_poll_t = void (*)(retro_input_poll_t);
    using retro_set_input_state_t = void (*)(retro_input_state_t);
    using retro_init_t = void (*)();
    using retro_deinit_t = void (*)();
    using retro_load_game_t = bool (*)(const retro_game_info *);
    using retro_unload_game_t = void (*)();
    using retro_run_t = void (*)();
    using retro_get_system_info_t = void (*)(retro_system_info *);
    using retro_get_system_av_info_t = void (*)(retro_system_av_info *);
    using retro_serialize_size_t = size_t (*)();
    using retro_serialize_t = bool (*)(void *, size_t);
    using retro_unserialize_t = bool (*)(const void *, size_t);

    static bool environmentCallback(unsigned command, void *data);
    static void videoCallback(const void *data, unsigned width, unsigned height, size_t pitch);
    static void audioSampleCallback(int16_t left, int16_t right);
    static size_t audioBatchCallback(const int16_t *data, size_t frames);
    static void inputPollCallback();
    static int16_t inputStateCallback(unsigned port, unsigned device, unsigned index, unsigned id);
    static void logCallback(retro_log_level level, const char *format, ...);

    bool environment(unsigned command, void *data);
    void video(const void *data, unsigned width, unsigned height, size_t pitch);
    size_t audioBatch(const int16_t *data, size_t frames);
    int16_t inputState(unsigned port, unsigned device, unsigned index, unsigned id) const;
    void setError(const QString &message);
    bool resolveSymbols();
    void resetSymbols();
    void installCallbacks();
    QByteArray pathToUtf8(const QString &path) const;
    int buttonForKey(int key) const;
    static bool isXInputControlPressed(int control, const void *state);
    void pollXInput();
    bool initializeResampler(int sourceSampleRate);
    void releaseResampler();

    static LibretroCore *active_core_;
    static constexpr int output_sample_rate_ = 48000;

    EmulatorView *video_output_ = nullptr;
    QLibrary library_;
    QTimer frame_timer_;
    WasapiAudio audio_;
    QString last_error_;
    SRC_STATE *resampler_ = nullptr;
    int source_sample_rate_ = 0;
    std::vector<float> input_float_buffer_;
    std::vector<float> output_float_buffer_;

    QByteArray system_directory_utf8_;
    QByteArray save_directory_utf8_;
    QByteArray content_path_utf8_;
    std::string selected_bios_;

    bool initialized_ = false;
    bool game_loaded_ = false;
    bool paused_ = false;
    std::array<bool, 16> keyboard_joypad_state_ {};
    std::array<bool, 16> xinput_joypad_state_ {};
    std::array<int, 16> key_bindings_ {};
    std::array<int, 16> xinput_bindings_ {};
    unsigned xinput_user_index_ = 0;

    retro_set_environment_t retro_set_environment_ = nullptr;
    retro_set_video_refresh_t retro_set_video_refresh_ = nullptr;
    retro_set_audio_sample_t retro_set_audio_sample_ = nullptr;
    retro_set_audio_sample_batch_t retro_set_audio_sample_batch_ = nullptr;
    retro_set_input_poll_t retro_set_input_poll_ = nullptr;
    retro_set_input_state_t retro_set_input_state_ = nullptr;
    retro_init_t retro_init_ = nullptr;
    retro_deinit_t retro_deinit_ = nullptr;
    retro_load_game_t retro_load_game_ = nullptr;
    retro_unload_game_t retro_unload_game_ = nullptr;
    retro_run_t retro_run_ = nullptr;
    retro_get_system_info_t retro_get_system_info_ = nullptr;
    retro_get_system_av_info_t retro_get_system_av_info_ = nullptr;
    retro_serialize_size_t retro_serialize_size_ = nullptr;
    retro_serialize_t retro_serialize_ = nullptr;
    retro_unserialize_t retro_unserialize_ = nullptr;
};
