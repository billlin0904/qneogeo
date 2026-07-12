#pragma once

#include "libretrocore.h"
#include "tools/fbneo_training.h"

#include <QLibrary>
#include <QTimer>

class FbneoTrainingCore final : public LibretroCore {
    Q_OBJECT

public:
    explicit FbneoTrainingCore(EmulatorView *videoOutput, QObject *parent = nullptr);
    ~FbneoTrainingCore() override;

    bool loadCore(const QString &corePath) override;
    bool startGame(const QString &contentPath, const QString &systemDirectory, const QString &saveDirectory) override;
    void stop() override;
    bool reset() override;
    void setPaused(bool paused) override;
    bool isPaused() const override;
    bool isGameLoaded() const override;
    bool saveState(const QString &statePath) override;
    bool loadState(const QString &statePath) override;
    bool readSystemRam(QByteArray &ram) const override;
    bool readSystemRamByte(uint32_t address, uint8_t &value) const override;

    QString displayName() const override;
    QString coreFileName() const override;
    QString romDirectoryName() const override;
    QStringList supportedExtensions() const override;

private:
    using kof_env_create_t = kof_env_handle (*)();
    using kof_env_destroy_t = void (*)(kof_env_handle);
    using kof_env_load_core_t = int (*)(kof_env_handle, const wchar_t *);
    using kof_env_load_game_t = int (*)(kof_env_handle, const wchar_t *, const wchar_t *, const wchar_t *);
    using kof_env_reset_t = int (*)(kof_env_handle);
    using kof_env_load_state_t = int (*)(kof_env_handle, const wchar_t *);
    using kof_env_save_state_t = int (*)(kof_env_handle, const wchar_t *);
    using kof_env_set_joypad_t = void (*)(kof_env_handle, const kof_env_joypad_state *);
    using kof_env_set_video_refresh_t = void (*)(kof_env_handle, kof_env_video_refresh_t, void *);
    using kof_env_run_frames_t = int (*)(kof_env_handle, int32_t);
    using kof_env_system_ram_size_t = uint32_t (*)(kof_env_handle);
    using kof_env_copy_system_ram_t = int (*)(kof_env_handle, void *, uint32_t);
    using kof_env_last_error_t = const char *(*)(kof_env_handle);

    template <typename T>
    T resolveSymbol(const char *name);

    bool resolveSymbols();
    bool recreateRuntime();
    bool fail(const QString &message);
    bool failFromRuntime(const QString &action);
    void destroyRuntime();
    void unloadLibrary();
    void advanceFrame();
    void updateJoypad();
    void handleVideoFrame(const void *data, unsigned width, unsigned height, size_t pitch);

    static void videoRefreshCallback(const void *data,
                                     unsigned width,
                                     unsigned height,
                                     size_t pitch,
                                     void *userData);

    QLibrary library_;
    QTimer frame_timer_;
    QString library_path_;
    kof_env_handle handle_ = nullptr;
    bool game_loaded_ = false;
    bool paused_ = false;

    kof_env_create_t kof_env_create_ = nullptr;
    kof_env_destroy_t kof_env_destroy_ = nullptr;
    kof_env_load_core_t kof_env_load_core_ = nullptr;
    kof_env_load_game_t kof_env_load_game_ = nullptr;
    kof_env_reset_t kof_env_reset_ = nullptr;
    kof_env_load_state_t kof_env_load_state_ = nullptr;
    kof_env_save_state_t kof_env_save_state_ = nullptr;
    kof_env_set_joypad_t kof_env_set_joypad_ = nullptr;
    kof_env_set_video_refresh_t kof_env_set_video_refresh_ = nullptr;
    kof_env_run_frames_t kof_env_run_frames_ = nullptr;
    kof_env_system_ram_size_t kof_env_system_ram_size_ = nullptr;
    kof_env_copy_system_ram_t kof_env_copy_system_ram_ = nullptr;
    kof_env_last_error_t kof_env_last_error_ = nullptr;
};
