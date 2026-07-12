#pragma once

#include "emulatorview.h"
#include "iemulatorcore.h"
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

class LibretroCore : public IEmulatorCore {
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

    bool loadCore(const QString &corePath) override;
    bool startGame(const QString &contentPath, const QString &systemDirectory, const QString &saveDirectory) override;
    void stop() override;
    bool reset() override;
    void setPaused(bool paused) override;
    bool isPaused() const override;
    bool isGameLoaded() const override;
    bool saveState(const QString &statePath) override;
    bool loadState(const QString &statePath) override;
    virtual bool readSystemRam(QByteArray &ram) const;
    virtual bool readSystemRamByte(uint32_t address, uint8_t &value) const;

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
    bool arcadeSocdClean() const;
    void setArcadeSocdClean(bool enabled);
    bool keyboardMotionAssist() const;
    void setKeyboardMotionAssist(bool enabled);
    QString systemRegionOption() const;
    void setSystemRegionOption(const QString &region);
    QString systemModeOption() const;
    void setSystemModeOption(const QString &mode);
    QString fbneoCpuClockOption() const;
    void setFbneoCpuClockOption(const QString &cpuClock);
    static QString xinputControlName(int control);
    static int firstPressedXInputControl(unsigned userIndex = 0);
    QString lastError() const override;
    QString displayName() const override;
    QString coreFileName() const override;
    QString romDirectoryName() const override;
    QStringList supportedExtensions() const override;

signals:
    void frameAdvanced();

private slots:
    void runFrame();

protected:
    bool eventFilter(QObject *watched, QEvent *event) override;

    virtual bool coreOptionValue(const QByteArray &key, const char *&value) const;
    virtual void coreOptionsUpdated(const retro_core_options_v2 *options);
    EmulatorView *videoOutput() const;
    void setError(const QString &message);
    void pollXInput();
    int16_t inputState(unsigned port, unsigned device, unsigned index, unsigned id) const;
    void finishKeyboardMotionAssist();
    void finishXInputMotionAssist();

    std::string selected_bios_;
    std::string system_region_option_ = "Japan";
    std::string system_mode_option_ = "MVS";
    std::string fbneo_cpu_clock_option_ = "100%";

private:
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
    bool resolveSymbols();
    void resetSymbols();
    void installCallbacks();
    QByteArray pathToUtf8(const QString &path) const;
    int buttonForKey(int key) const;
    static bool isXInputControlPressed(int control, const void *state);
    bool initializeResampler(int sourceSampleRate);
    void releaseResampler();
    void applyKeyboardInputState();
    void applyXInputInputState();
    uint8_t rawKeyboardDirectionBits() const;
    uint8_t rawXInputDirectionBits() const;
    uint8_t cleanedDirectionBits(uint8_t directionBits) const;
    void setKeyboardDirectionBits(uint8_t directionBits);
    void setXInputDirectionBits(uint8_t directionBits);

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
    bool initialized_ = false;
    bool game_loaded_ = false;
    bool paused_ = false;
    bool arcade_socd_clean_ = true;
    bool keyboard_motion_assist_ = true;
    uint8_t current_keyboard_direction_bits_ = 0;
    uint8_t pending_keyboard_direction_bits_ = 0;
    uint8_t current_xinput_direction_bits_ = 0;
    uint8_t pending_xinput_direction_bits_ = 0;
    int motion_assist_polls_remaining_ = 0;
    int xinput_motion_assist_polls_remaining_ = 0;
    std::array<bool, 16> raw_keyboard_joypad_state_ {};
    std::array<bool, 16> keyboard_joypad_state_ {};
    std::array<bool, 16> raw_xinput_joypad_state_ {};
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
