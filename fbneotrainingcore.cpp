#include "fbneotrainingcore.h"
#include "ppoagentbridge.h"

#include <QApplication>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QKeyEvent>

#include <stdexcept>
#include <type_traits>

FbneoTrainingCore::FbneoTrainingCore(EmulatorView *videoOutput, QObject *parent)
    : LibretroCore(videoOutput, parent) {
    ppo_agent_bridge_ = new PpoAgentBridge(this);
    connect(ppo_agent_bridge_, &PpoAgentBridge::actionReady, this, [this](int32_t action_id) {
        if (p2_ppo_ai_enabled_ && handle_ && kof_env_set_p2_action_)
            kof_env_set_p2_action_(handle_, action_id);
    });
    connect(ppo_agent_bridge_, &PpoAgentBridge::failed, this, [this](const QString &message) {
        p2_ppo_ai_enabled_ = false;
        if (handle_ && kof_env_set_p2_action_ai_)
            kof_env_set_p2_action_ai_(handle_, 0);
        setError(QStringLiteral("P2 PPO AI: %1").arg(message));
        emit p2PpoAiFailed(message);
    });
    frame_timer_.setTimerType(Qt::PreciseTimer);
    connect(&frame_timer_, &QTimer::timeout, this, &FbneoTrainingCore::advanceFrame);
}

FbneoTrainingCore::~FbneoTrainingCore() {
    stop();
    unloadLibrary();
}

bool FbneoTrainingCore::loadCore(const QString &corePath) {
    stop();
    unloadLibrary();

    library_path_ = QFileInfo(corePath).absoluteFilePath();
    library_.setFileName(library_path_);
    if (!library_.load())
        return fail(QStringLiteral("Could not load fbneo_training: %1").arg(library_.errorString()));

    if (!resolveSymbols()) {
        unloadLibrary();
        return false;
    }

    return true;
}

bool FbneoTrainingCore::startGame(const QString &contentPath,
                                  const QString &systemDirectory,
                                  const QString &saveDirectory) {
    if (!library_.isLoaded()) {
        const QString path = QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(coreFileName());
        if (!loadCore(path))
            return false;
    }

    if (!recreateRuntime())
        return false;

    QDir().mkpath(systemDirectory);
    QDir().mkpath(saveDirectory);

    const QString fbneo_core_path = QFileInfo(library_path_).dir().absoluteFilePath(QStringLiteral("fbneo_libretro.dll"));
    const std::wstring fbneo_core_path_w = QFileInfo(fbneo_core_path).absoluteFilePath().toStdWString();
    if (!kof_env_load_core_(handle_, fbneo_core_path_w.c_str()))
        return failFromRuntime(QStringLiteral("Could not load fbneo_libretro.dll through fbneo_training"));

    const std::wstring content_path_w = QFileInfo(contentPath).absoluteFilePath().toStdWString();
    const std::wstring system_directory_w = QFileInfo(systemDirectory).absoluteFilePath().toStdWString();
    const std::wstring save_directory_w = QFileInfo(saveDirectory).absoluteFilePath().toStdWString();

    if (!kof_env_load_game_(handle_,
                            content_path_w.c_str(),
                            system_directory_w.c_str(),
                            save_directory_w.c_str())) {
        return failFromRuntime(QStringLiteral("Could not load game through fbneo_training"));
    }

    game_loaded_ = true;
    paused_ = false;
    resetInputFrameTracking();
    if (p2_ppo_ai_enabled_ &&
        !ppo_agent_bridge_->start(
            p2_ppo_python_path_,
            p2_ppo_script_path_,
            p2_ppo_model_path_,
            p2_ppo_combo_model_path_)) {
        game_loaded_ = false;
        return fail(QStringLiteral("Could not start P2 PPO AI: %1")
                        .arg(ppo_agent_bridge_->lastError()));
    }
    frame_timer_.start(16);
    emit pausedChanged(false);
    return true;
}

void FbneoTrainingCore::stop() {
    frame_timer_.stop();
    if (ppo_agent_bridge_)
        ppo_agent_bridge_->stop();
    destroyRuntime();
    game_loaded_ = false;
    resetInputFrameTracking();

    if (videoOutput())
        videoOutput()->clearFrame();

    if (paused_) {
        paused_ = false;
        emit pausedChanged(false);
    }
}

bool FbneoTrainingCore::reset() {
    if (!game_loaded_ || !handle_ || !kof_env_reset_)
        return fail(QStringLiteral("No running fbneo_training game is available to reset."));

    const bool resume_after_reset = !paused_;
    frame_timer_.stop();

    if (videoOutput())
        videoOutput()->clearFrame();

    if (!kof_env_reset_(handle_))
        return failFromRuntime(QStringLiteral("fbneo_training reset failed"));

    resetInputFrameTracking();

    if (resume_after_reset)
        frame_timer_.start(16);

    return true;
}

void FbneoTrainingCore::setPaused(bool paused) {
    if (paused_ == paused)
        return;

    paused_ = paused;
    if (paused_)
        frame_timer_.stop();
    else if (game_loaded_)
        frame_timer_.start(16);

    emit pausedChanged(paused_);
}

bool FbneoTrainingCore::isPaused() const {
    return paused_;
}

bool FbneoTrainingCore::isGameLoaded() const {
    return game_loaded_;
}

bool FbneoTrainingCore::saveState(const QString &statePath) {
    if (!game_loaded_ || !handle_ || !kof_env_save_state_)
        return fail(QStringLiteral("No running fbneo_training game is available for save state."));

    const bool timer_was_active = frame_timer_.isActive();
    frame_timer_.stop();

    const QFileInfo file_info(statePath);
    QDir().mkpath(file_info.absolutePath());
    const std::wstring state_path_w = file_info.absoluteFilePath().toStdWString();
    const bool saved = kof_env_save_state_(handle_, state_path_w.c_str()) != 0;

    if (timer_was_active && !paused_)
        frame_timer_.start(16);

    return saved ? true : failFromRuntime(QStringLiteral("fbneo_training save state failed"));
}

bool FbneoTrainingCore::loadState(const QString &statePath) {
    if (!game_loaded_ || !handle_ || !kof_env_load_state_)
        return fail(QStringLiteral("No running fbneo_training game is available for load state."));

    const bool timer_was_active = frame_timer_.isActive();
    frame_timer_.stop();

    const std::wstring state_path_w = QFileInfo(statePath).absoluteFilePath().toStdWString();
    const bool loaded = kof_env_load_state_(handle_, state_path_w.c_str()) != 0;
    if (loaded)
        resetInputFrameTracking();

    if (timer_was_active && !paused_)
        frame_timer_.start(16);

    return loaded ? true : failFromRuntime(QStringLiteral("fbneo_training load state failed"));
}

bool FbneoTrainingCore::readSystemRam(QByteArray &ram) const {
    ram.clear();

    if (!game_loaded_ || !handle_ || !kof_env_system_ram_size_ || !kof_env_copy_system_ram_)
        return false;

    const uint32_t ram_size = kof_env_system_ram_size_(handle_);
    if (ram_size == 0)
        return false;

    ram.resize(static_cast<qsizetype>(ram_size));
    if (!kof_env_copy_system_ram_(handle_, ram.data(), ram_size)) {
        ram.clear();
        return false;
    }

    return true;
}

bool FbneoTrainingCore::readSystemRamByte(uint32_t address, uint8_t &value) const {
    QByteArray ram;
    if (!readSystemRam(ram) || address >= static_cast<uint32_t>(ram.size()))
        return false;

    value = static_cast<uint8_t>(ram[static_cast<int>(address)]);
    return true;
}

void FbneoTrainingCore::setP2RandomAiEnabled(bool enabled) {
    p2_random_ai_enabled_ = enabled;
    if (enabled)
        p2_ppo_ai_enabled_ = false;
    if (ppo_agent_bridge_ && enabled)
        ppo_agent_bridge_->stop();
    if (handle_ && kof_env_set_p2_random_ai_)
        kof_env_set_p2_random_ai_(handle_, enabled ? 1 : 0);
    if (!enabled)
        updateJoypad();
}

bool FbneoTrainingCore::setP2PpoAiEnabled(bool enabled,
                                          const QString &pythonPath,
                                          const QString &scriptPath,
                                          const QString &modelPath,
                                          const QString &comboModelPath) {
    p2_ppo_python_path_ = pythonPath;
    p2_ppo_script_path_ = scriptPath;
    p2_ppo_model_path_ = modelPath;
    p2_ppo_combo_model_path_ = comboModelPath;
    p2_ppo_ai_enabled_ = enabled;
    p2_random_ai_enabled_ = false;
    p2_ppo_frame_counter_ = 0;

    if (handle_ && kof_env_set_p2_random_ai_)
        kof_env_set_p2_random_ai_(handle_, 0);
    if (handle_ && kof_env_set_p2_action_ai_)
        kof_env_set_p2_action_ai_(handle_, enabled ? 1 : 0);

    if (!enabled) {
        if (ppo_agent_bridge_)
            ppo_agent_bridge_->stop();
        updateJoypad();
        return true;
    }

    if (!game_loaded_)
        return true;

    if (!ppo_agent_bridge_ ||
        !ppo_agent_bridge_->start(
            pythonPath,
            scriptPath,
            modelPath,
            comboModelPath)) {
        p2_ppo_ai_enabled_ = false;
        if (handle_ && kof_env_set_p2_action_ai_)
            kof_env_set_p2_action_ai_(handle_, 0);
        return fail(QStringLiteral("Could not start P2 PPO AI: %1")
                        .arg(ppo_agent_bridge_ ? ppo_agent_bridge_->lastError()
                                             : QStringLiteral("bridge unavailable")));
    }
    return true;
}

QString FbneoTrainingCore::displayName() const {
    return QStringLiteral("FBNeo Training DLL");
}

QString FbneoTrainingCore::coreFileName() const {
    return QStringLiteral("fbneo_training.dll");
}

QString FbneoTrainingCore::romDirectoryName() const {
    return QStringLiteral("fbneo");
}

QStringList FbneoTrainingCore::supportedExtensions() const {
    return {
        QStringLiteral("zip"),
        QStringLiteral("7z")
    };
}

template <typename T>
T FbneoTrainingCore::resolveSymbol(const char *name) {
    static_assert(
        std::is_pointer_v<T> &&
        std::is_function_v<std::remove_pointer_t<T>>,
        "T must be a function pointer type"
    );

    auto symbol = library_.resolve(name);
    if (!symbol)
        throw std::runtime_error(name);

    return reinterpret_cast<T>(symbol);
}

bool FbneoTrainingCore::resolveSymbols() {
    try {
        kof_env_create_ = resolveSymbol<kof_env_create_t>("kof_env_create");
        kof_env_destroy_ = resolveSymbol<kof_env_destroy_t>("kof_env_destroy");
        kof_env_load_core_ = resolveSymbol<kof_env_load_core_t>("kof_env_load_core");
        kof_env_load_game_ = resolveSymbol<kof_env_load_game_t>("kof_env_load_game");
        kof_env_reset_ = resolveSymbol<kof_env_reset_t>("kof_env_reset");
        kof_env_load_state_ = resolveSymbol<kof_env_load_state_t>("kof_env_load_state");
        kof_env_save_state_ = resolveSymbol<kof_env_save_state_t>("kof_env_save_state");
        kof_env_set_joypad_ = resolveSymbol<kof_env_set_joypad_t>("kof_env_set_joypad");
        kof_env_set_joypad_for_port_ =
            resolveSymbol<kof_env_set_joypad_for_port_t>("kof_env_set_joypad_for_port");
        kof_env_get_last_joypad_for_port_ =
            resolveSymbol<kof_env_get_last_joypad_for_port_t>("kof_env_get_last_joypad_for_port");
        kof_env_set_video_refresh_ = resolveSymbol<kof_env_set_video_refresh_t>("kof_env_set_video_refresh");
        kof_env_set_p2_random_ai_ = resolveSymbol<kof_env_set_p2_random_ai_t>("kof_env_set_p2_random_ai");
        kof_env_set_p2_action_ai_ = resolveSymbol<kof_env_set_p2_action_ai_t>("kof_env_set_p2_action_ai");
        kof_env_set_p2_action_ = resolveSymbol<kof_env_set_p2_action_t>("kof_env_set_p2_action");
        kof_env_can_queue_p2_action_ =
            resolveSymbol<kof_env_can_queue_p2_action_t>("kof_env_can_queue_p2_action");
        kof_env_p2_input_ready_ =
            resolveSymbol<kof_env_p2_input_ready_t>("kof_env_p2_input_ready");
        kof_env_p2_ready_for_action_ =
            resolveSymbol<kof_env_p2_ready_for_action_t>("kof_env_p2_ready_for_action");
        kof_env_get_observation_ =
            resolveSymbol<kof_env_get_observation_t>("kof_env_get_observation");
        kof_env_run_frames_ = resolveSymbol<kof_env_run_frames_t>("kof_env_run_frames");
        kof_env_system_ram_size_ = resolveSymbol<kof_env_system_ram_size_t>("kof_env_system_ram_size");
        kof_env_copy_system_ram_ = resolveSymbol<kof_env_copy_system_ram_t>("kof_env_copy_system_ram");
        kof_env_last_error_ = resolveSymbol<kof_env_last_error_t>("kof_env_last_error");
    } catch (const std::exception &exception) {
        return fail(QStringLiteral("%1 is missing required export: %2")
                        .arg(coreFileName(), QString::fromLatin1(exception.what())));
    }

    return true;
}

bool FbneoTrainingCore::recreateRuntime() {
    destroyRuntime();

    handle_ = kof_env_create_ ? kof_env_create_() : nullptr;
    if (!handle_)
        return fail(QStringLiteral("kof_env_create failed."));

    kof_env_set_video_refresh_(handle_, videoRefreshCallback, this);
    if (kof_env_set_p2_random_ai_)
        kof_env_set_p2_random_ai_(handle_, p2_random_ai_enabled_ ? 1 : 0);
    if (kof_env_set_p2_action_ai_)
        kof_env_set_p2_action_ai_(handle_, p2_ppo_ai_enabled_ ? 1 : 0);
    return true;
}

bool FbneoTrainingCore::fail(const QString &message) {
    setError(message);
    return false;
}

bool FbneoTrainingCore::failFromRuntime(const QString &action) {
    QString detail;
    if (kof_env_last_error_ && handle_) {
        if (const char *message = kof_env_last_error_(handle_))
            detail = QString::fromUtf8(message);
    }

    if (detail.isEmpty())
        detail = QStringLiteral("Unknown fbneo_training error.");

    return fail(QStringLiteral("%1: %2").arg(action, detail));
}

void FbneoTrainingCore::destroyRuntime() {
    if (handle_ && kof_env_destroy_)
        kof_env_destroy_(handle_);
    handle_ = nullptr;
}

void FbneoTrainingCore::unloadLibrary() {
    destroyRuntime();
    if (library_.isLoaded())
        library_.unload();

    kof_env_create_ = nullptr;
    kof_env_destroy_ = nullptr;
    kof_env_load_core_ = nullptr;
    kof_env_load_game_ = nullptr;
    kof_env_reset_ = nullptr;
    kof_env_load_state_ = nullptr;
    kof_env_save_state_ = nullptr;
    kof_env_set_joypad_ = nullptr;
    kof_env_set_joypad_for_port_ = nullptr;
    kof_env_get_last_joypad_for_port_ = nullptr;
    kof_env_set_video_refresh_ = nullptr;
    kof_env_set_p2_random_ai_ = nullptr;
    kof_env_set_p2_action_ai_ = nullptr;
    kof_env_set_p2_action_ = nullptr;
    kof_env_can_queue_p2_action_ = nullptr;
    kof_env_p2_input_ready_ = nullptr;
    kof_env_p2_ready_for_action_ = nullptr;
    kof_env_get_observation_ = nullptr;
    kof_env_run_frames_ = nullptr;
    kof_env_system_ram_size_ = nullptr;
    kof_env_copy_system_ram_ = nullptr;
    kof_env_last_error_ = nullptr;
}

void FbneoTrainingCore::advanceFrame() {
    if (!game_loaded_ || paused_ || !handle_ || !kof_env_run_frames_)
        return;

    pollXInput();
    updateJoypad();
    const EmulatorView::JoypadInput p1_input = currentP1Input();

    if (!kof_env_run_frames_(handle_, 1)) {
        failFromRuntime(QStringLiteral("fbneo_training frame failed"));
        setPaused(true);
        return;
    }

    if (p2_ppo_ai_enabled_) {
        ++p2_ppo_frame_counter_;
        if (p2_ppo_frame_counter_ >= 4) {
            p2_ppo_frame_counter_ = 0;
            requestP2PpoAction();
        }
    }

    auto toJoypadInput = [](const kof_env_joypad_state &state) {
        EmulatorView::JoypadInput input;
        input.up = state.up != 0;
        input.down = state.down != 0;
        input.left = state.left != 0;
        input.right = state.right != 0;
        input.a = state.a != 0;
        input.b = state.b != 0;
        input.c = state.c != 0;
        input.d = state.d != 0;
        return input;
    };

    EmulatorView::JoypadInput recorded_p1_input = p1_input;
    EmulatorView::JoypadInput recorded_p2_input;
    if (kof_env_get_last_joypad_for_port_) {
        kof_env_joypad_state state {};
        if (kof_env_get_last_joypad_for_port_(handle_, 0, &state))
            recorded_p1_input = toJoypadInput(state);
        if (kof_env_get_last_joypad_for_port_(handle_, 1, &state))
            recorded_p2_input = toJoypadInput(state);
    }

    recordInputFrame(recorded_p1_input, recorded_p2_input);
    finishKeyboardMotionAssist();
    finishXInputMotionAssist();
    emit frameAdvanced();
}

void FbneoTrainingCore::updateJoypad() {
    if (!handle_ || !kof_env_set_joypad_)
        return;

    auto pressed = [this](unsigned id) -> uint8_t {
        return inputState(0, RETRO_DEVICE_JOYPAD, 0, id) ? 1 : 0;
    };

    kof_env_joypad_state state {};
    state.up = pressed(RETRO_DEVICE_ID_JOYPAD_UP);
    state.down = pressed(RETRO_DEVICE_ID_JOYPAD_DOWN);
    state.left = pressed(RETRO_DEVICE_ID_JOYPAD_LEFT);
    state.right = pressed(RETRO_DEVICE_ID_JOYPAD_RIGHT);
    state.start = pressed(RETRO_DEVICE_ID_JOYPAD_START);
    state.coin = pressed(RETRO_DEVICE_ID_JOYPAD_SELECT);
    state.a = pressed(RETRO_DEVICE_ID_JOYPAD_B);
    state.b = pressed(RETRO_DEVICE_ID_JOYPAD_A);
    state.c = pressed(RETRO_DEVICE_ID_JOYPAD_Y);
    state.d = pressed(RETRO_DEVICE_ID_JOYPAD_X);

    if (pressed(RETRO_DEVICE_ID_JOYPAD_L)) {
        state.b = 1;
        state.c = 1;
    }
    if (pressed(RETRO_DEVICE_ID_JOYPAD_L2)) {
        state.a = 1;
        state.b = 1;
    }
    if (pressed(RETRO_DEVICE_ID_JOYPAD_R2)) {
        state.a = 1;
        state.b = 1;
        state.c = 1;
    }

    kof_env_set_joypad_(handle_, &state);

    if (!p2_random_ai_enabled_ &&
        !p2_ppo_ai_enabled_ &&
        kof_env_set_joypad_for_port_) {
        auto p2Pressed = [this](unsigned id) -> uint8_t {
            return id < p2_keyboard_joypad_state_.size() && p2_keyboard_joypad_state_[id] ? 1 : 0;
        };

        kof_env_joypad_state p2_state {};
        p2_state.up = p2Pressed(RETRO_DEVICE_ID_JOYPAD_UP);
        p2_state.down = p2Pressed(RETRO_DEVICE_ID_JOYPAD_DOWN);
        p2_state.left = p2Pressed(RETRO_DEVICE_ID_JOYPAD_LEFT);
        p2_state.right = p2Pressed(RETRO_DEVICE_ID_JOYPAD_RIGHT);
        p2_state.start = p2Pressed(RETRO_DEVICE_ID_JOYPAD_START);
        p2_state.coin = p2Pressed(RETRO_DEVICE_ID_JOYPAD_SELECT);
        p2_state.a = p2Pressed(RETRO_DEVICE_ID_JOYPAD_B);
        p2_state.b = p2Pressed(RETRO_DEVICE_ID_JOYPAD_A);
        p2_state.c = p2Pressed(RETRO_DEVICE_ID_JOYPAD_Y);
        p2_state.d = p2Pressed(RETRO_DEVICE_ID_JOYPAD_X);

        kof_env_set_joypad_for_port_(handle_, 1, &p2_state);
    }
}

void FbneoTrainingCore::requestP2PpoAction() {
    if (!p2_ppo_ai_enabled_ ||
        !ppo_agent_bridge_ ||
        !ppo_agent_bridge_->isReady() ||
        ppo_agent_bridge_->hasPendingRequest() ||
        !handle_ ||
        !kof_env_get_observation_) {
        return;
    }

    kof_env_observation observation {};
    if (!kof_env_get_observation_(handle_, &observation))
        return;

    ppo_agent_bridge_->requestAction(
        p2ObservationVector(observation),
        p2ActionMask(observation));
}

QVector<float> FbneoTrainingCore::p2ObservationVector(
    const kof_env_observation &observation) const {
    // Mirror the screen after swapping player roles so the deployed P2 sees
    // the same left-side coordinate distribution used to train the P1 policy.
    const int32_t p2_x =
        observation.p2_has_position ? 320 - observation.p2_x : 0;
    const int32_t p2_y = observation.p2_has_position ? observation.p2_y : 0;
    const int32_t p1_x =
        observation.p1_has_position ? 320 - observation.p1_x : 0;
    const int32_t p1_y = observation.p1_has_position ? observation.p1_y : 0;
    const int32_t p2_combo = qMax(0, observation.p2_combo_count);
    const int32_t p1_combo = qMax(0, observation.p1_combo_count);
    const int32_t p2_power_value = qMax(0, observation.p2_advanced_power_value);
    const int32_t p2_power_stocks = qMax(0, observation.p2_advanced_power_stocks);
    const int32_t p1_power_value = qMax(0, observation.p1_advanced_power_value);
    const int32_t p1_power_stocks = qMax(0, observation.p1_advanced_power_stocks);
    const bool input_ready =
        kof_env_p2_input_ready_ &&
        kof_env_p2_ready_for_action_ &&
        kof_env_p2_input_ready_(handle_) != 0 &&
        kof_env_p2_ready_for_action_(handle_) != 0;

    return {
        observation.round_time / 99.0f,
        observation.p2_health / 103.0f,
        observation.p1_health / 103.0f,
        observation.p2_power / 128.0f,
        observation.p1_power / 128.0f,
        observation.p2_power_state / 128.0f,
        observation.p1_power_state / 128.0f,
        p2_power_value / 128.0f,
        p2_power_stocks / 5.0f,
        p1_power_value / 128.0f,
        p1_power_stocks / 5.0f,
        observation.p2_stun / 255.0f,
        observation.p1_stun / 255.0f,
        p2_combo / 99.0f,
        p1_combo / 99.0f,
        p2_x / 320.0f,
        p2_y / 224.0f,
        p1_x / 320.0f,
        p1_y / 224.0f,
        observation.distance_x / 320.0f,
        -observation.distance_y / 224.0f,
        static_cast<float>(observation.p2_has_position != 0),
        static_cast<float>(observation.p1_has_position != 0),
        static_cast<float>(input_ready),
        0.0f,
        0.0f,
    };
}

QVector<bool> FbneoTrainingCore::p2ActionMask(
    const kof_env_observation &observation) const {
    constexpr int32_t ACTION_COUNT = 29;
    constexpr int32_t OROCHINAGI_ACTION_ID = 18;
    constexpr int32_t MUSHIKI_ACTION_ID = 19;
    QVector<bool> mask(ACTION_COUNT, false);
    mask[0] = true;

    const bool super_available = observation.p2_advanced_power_stocks >= 1;
    const bool input_ready =
        kof_env_p2_input_ready_ &&
        kof_env_p2_ready_for_action_ &&
        kof_env_p2_input_ready_(handle_) != 0 &&
        kof_env_p2_ready_for_action_(handle_) != 0;
    if (input_ready) {
        mask.fill(true);
        if (!super_available) {
            mask[OROCHINAGI_ACTION_ID] = false;
            mask[MUSHIKI_ACTION_ID] = false;
        }
        return mask;
    }

    if (!kof_env_can_queue_p2_action_)
        return mask;
    for (int32_t action_id = 1; action_id < ACTION_COUNT; ++action_id) {
        if ((action_id == OROCHINAGI_ACTION_ID ||
             action_id == MUSHIKI_ACTION_ID) &&
            !super_available) {
            continue;
        }
        mask[action_id] =
            kof_env_can_queue_p2_action_(handle_, action_id) != 0;
    }
    return mask;
}

int FbneoTrainingCore::p2ButtonForKey(int key) const {
    switch (key) {
    case Qt::Key_I:
        return RETRO_DEVICE_ID_JOYPAD_UP;
    case Qt::Key_K:
        return RETRO_DEVICE_ID_JOYPAD_DOWN;
    case Qt::Key_J:
        return RETRO_DEVICE_ID_JOYPAD_LEFT;
    case Qt::Key_L:
        return RETRO_DEVICE_ID_JOYPAD_RIGHT;
    case Qt::Key_8:
        return RETRO_DEVICE_ID_JOYPAD_START;
    case Qt::Key_7:
        return RETRO_DEVICE_ID_JOYPAD_SELECT;
    case Qt::Key_U:
        return RETRO_DEVICE_ID_JOYPAD_B;
    case Qt::Key_O:
        return RETRO_DEVICE_ID_JOYPAD_A;
    case Qt::Key_H:
        return RETRO_DEVICE_ID_JOYPAD_Y;
    case Qt::Key_Semicolon:
        return RETRO_DEVICE_ID_JOYPAD_X;
    default:
        return -1;
    }
}

bool FbneoTrainingCore::eventFilter(QObject *watched, QEvent *event) {
    if (QApplication::activeModalWidget())
        return LibretroCore::eventFilter(watched, event);

    if (event->type() == QEvent::KeyPress || event->type() == QEvent::KeyRelease) {
        auto *key_event = static_cast<QKeyEvent *>(event);
        const int button = p2ButtonForKey(key_event->key());
        if (button >= 0) {
            if (!key_event->isAutoRepeat()) {
                p2_keyboard_joypad_state_[static_cast<size_t>(button)] = event->type() == QEvent::KeyPress;
                if (!p2_random_ai_enabled_ && !p2_ppo_ai_enabled_)
                    updateJoypad();
            }
            return true;
        }
    }

    return LibretroCore::eventFilter(watched, event);
}

void FbneoTrainingCore::handleVideoFrame(const void *data, unsigned width, unsigned height, size_t pitch) {
    if (!videoOutput() || !data || width == 0 || height == 0)
        return;

    videoOutput()->submitFrame(data,
                               static_cast<int>(width),
                               static_cast<int>(height),
                               static_cast<int>(pitch),
                               EmulatorView::PixelFormat::Rgb565);
}

void FbneoTrainingCore::videoRefreshCallback(const void *data,
                                             unsigned width,
                                             unsigned height,
                                             size_t pitch,
                                             void *userData) {
    if (auto *core = static_cast<FbneoTrainingCore *>(userData))
        core->handleVideoFrame(data, width, height, pitch);
}
