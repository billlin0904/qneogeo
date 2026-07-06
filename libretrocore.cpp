#include "libretrocore.h"

#include <QApplication>
#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QEvent>
#include <QFile>
#include <QFileInfo>
#include <QKeyEvent>
#include <QSaveFile>
#include <QSettings>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <Xinput.h>

#include <cstdarg>
#include <cstdio>
#include <cmath>

LibretroCore *LibretroCore::active_core_ = nullptr;

namespace {
constexpr uint8_t DirectionUp = 0x01;
constexpr uint8_t DirectionDown = 0x02;
constexpr uint8_t DirectionLeft = 0x04;
constexpr uint8_t DirectionRight = 0x08;

QString bindingSettingName(unsigned retroButtonId) {
    switch (retroButtonId) {
    case RETRO_DEVICE_ID_JOYPAD_UP:
        return QStringLiteral("Up");
    case RETRO_DEVICE_ID_JOYPAD_DOWN:
        return QStringLiteral("Down");
    case RETRO_DEVICE_ID_JOYPAD_LEFT:
        return QStringLiteral("Left");
    case RETRO_DEVICE_ID_JOYPAD_RIGHT:
        return QStringLiteral("Right");
    case RETRO_DEVICE_ID_JOYPAD_START:
        return QStringLiteral("Start");
    case RETRO_DEVICE_ID_JOYPAD_SELECT:
        return QStringLiteral("Select");
    case RETRO_DEVICE_ID_JOYPAD_B:
        return QStringLiteral("A");
    case RETRO_DEVICE_ID_JOYPAD_A:
        return QStringLiteral("B");
    case RETRO_DEVICE_ID_JOYPAD_Y:
        return QStringLiteral("C");
    case RETRO_DEVICE_ID_JOYPAD_X:
        return QStringLiteral("D");
    case RETRO_DEVICE_ID_JOYPAD_L:
        return QStringLiteral("BPlusC");
    case RETRO_DEVICE_ID_JOYPAD_L2:
        return QStringLiteral("APlusB");
    case RETRO_DEVICE_ID_JOYPAD_R2:
        return QStringLiteral("APlusBPlusC");
    default:
        return QStringLiteral("Button%1").arg(retroButtonId);
    }
}
}

LibretroCore::LibretroCore(EmulatorView *videoOutput, QObject *parent)
    : IEmulatorCore(parent)
    , video_output_(videoOutput)
    , audio_(this) {
    key_bindings_.fill(0);
    key_bindings_[RETRO_DEVICE_ID_JOYPAD_LEFT] = Qt::Key_Left;
    key_bindings_[RETRO_DEVICE_ID_JOYPAD_RIGHT] = Qt::Key_Right;
    key_bindings_[RETRO_DEVICE_ID_JOYPAD_UP] = Qt::Key_Up;
    key_bindings_[RETRO_DEVICE_ID_JOYPAD_DOWN] = Qt::Key_Down;
    key_bindings_[RETRO_DEVICE_ID_JOYPAD_START] = Qt::Key_Return;
    key_bindings_[RETRO_DEVICE_ID_JOYPAD_SELECT] = Qt::Key_Shift;
    key_bindings_[RETRO_DEVICE_ID_JOYPAD_B] = Qt::Key_Z;
    key_bindings_[RETRO_DEVICE_ID_JOYPAD_A] = Qt::Key_X;
    key_bindings_[RETRO_DEVICE_ID_JOYPAD_Y] = Qt::Key_A;
    key_bindings_[RETRO_DEVICE_ID_JOYPAD_X] = Qt::Key_S;

    xinput_bindings_.fill(XInputNone);
    xinput_bindings_[RETRO_DEVICE_ID_JOYPAD_UP] = XInputDpadUp;
    xinput_bindings_[RETRO_DEVICE_ID_JOYPAD_DOWN] = XInputDpadDown;
    xinput_bindings_[RETRO_DEVICE_ID_JOYPAD_LEFT] = XInputDpadLeft;
    xinput_bindings_[RETRO_DEVICE_ID_JOYPAD_RIGHT] = XInputDpadRight;
    xinput_bindings_[RETRO_DEVICE_ID_JOYPAD_START] = XInputStart;
    xinput_bindings_[RETRO_DEVICE_ID_JOYPAD_SELECT] = XInputBack;
    xinput_bindings_[RETRO_DEVICE_ID_JOYPAD_B] = XInputA;
    xinput_bindings_[RETRO_DEVICE_ID_JOYPAD_A] = XInputB;
    xinput_bindings_[RETRO_DEVICE_ID_JOYPAD_Y] = XInputX;
    xinput_bindings_[RETRO_DEVICE_ID_JOYPAD_X] = XInputY;
    xinput_bindings_[RETRO_DEVICE_ID_JOYPAD_L] = XInputLeftShoulder;
    xinput_bindings_[RETRO_DEVICE_ID_JOYPAD_R] = XInputRightShoulder;
    xinput_bindings_[RETRO_DEVICE_ID_JOYPAD_L2] = XInputLeftTrigger;
    xinput_bindings_[RETRO_DEVICE_ID_JOYPAD_R2] = XInputRightTrigger;

    frame_timer_.setTimerType(Qt::PreciseTimer);
    connect(&frame_timer_, &QTimer::timeout, this, &LibretroCore::runFrame);

    if (QCoreApplication::instance())
        QCoreApplication::instance()->installEventFilter(this);
}

LibretroCore::~LibretroCore() {
    if (QCoreApplication::instance())
        QCoreApplication::instance()->removeEventFilter(this);

    stop();
    if (active_core_ == this)
        active_core_ = nullptr;
}

bool LibretroCore::loadCore(const QString &corePath) {
    stop();

    if (library_.isLoaded()) {
        resetSymbols();
        library_.unload();
    }

    library_.setFileName(corePath);
    if (!library_.load()) {
        setError(QStringLiteral("Could not load core: %1").arg(library_.errorString()));
        return false;
    }

    if (!resolveSymbols()) {
        library_.unload();
        return false;
    }

    retro_system_info info {};
    retro_get_system_info_(&info);
    qInfo() << "Loaded libretro core"
            << (info.library_name ? info.library_name : "(unknown)")
            << (info.library_version ? info.library_version : "(unknown)")
            << "extensions"
            << (info.valid_extensions ? info.valid_extensions : "(unknown)");

    return true;
}

bool LibretroCore::startGame(const QString &contentPath, const QString &systemDirectory, const QString &saveDirectory) {
    if (!library_.isLoaded() && !loadCore(coreFileName()))
        return false;

    stop();

    QDir().mkpath(systemDirectory);
    QDir().mkpath(saveDirectory);

    system_directory_utf8_ = pathToUtf8(systemDirectory);
    save_directory_utf8_ = pathToUtf8(saveDirectory);
    content_path_utf8_ = pathToUtf8(contentPath);
    selected_bios_.clear();

    active_core_ = this;
    installCallbacks();

    retro_init_();
    initialized_ = true;

    retro_game_info game {};
    game.path = content_path_utf8_.constData();

    if (!retro_load_game_(&game)) {
        setError(QStringLiteral("Could not load game: %1").arg(contentPath));
        stop();
        return false;
    }

    game_loaded_ = true;

    retro_system_av_info av_info {};
    retro_get_system_av_info_(&av_info);
    if (!initializeResampler(static_cast<int>(std::lround(av_info.timing.sample_rate)))) {
        stop();
        return false;
    }

    if (!audio_.start(output_sample_rate_))
        qWarning().noquote() << "WASAPI shared audio disabled:" << audio_.lastError();

    paused_ = false;
    raw_keyboard_joypad_state_.fill(false);
    keyboard_joypad_state_.fill(false);
    raw_xinput_joypad_state_.fill(false);
    xinput_joypad_state_.fill(false);
    current_keyboard_direction_bits_ = 0;
    pending_keyboard_direction_bits_ = 0;
    current_xinput_direction_bits_ = 0;
    pending_xinput_direction_bits_ = 0;
    motion_assist_polls_remaining_ = 0;
    xinput_motion_assist_polls_remaining_ = 0;
    frame_timer_.start(16);
    emit pausedChanged(false);
    return true;
}

void LibretroCore::stop() {
    frame_timer_.stop();
    audio_.stop();
    releaseResampler();

    if (game_loaded_ && retro_unload_game_)
        retro_unload_game_();
    game_loaded_ = false;

    if (initialized_ && retro_deinit_)
        retro_deinit_();
    initialized_ = false;

    if (video_output_)
        video_output_->clearFrame();

    raw_keyboard_joypad_state_.fill(false);
    keyboard_joypad_state_.fill(false);
    raw_xinput_joypad_state_.fill(false);
    xinput_joypad_state_.fill(false);
    current_keyboard_direction_bits_ = 0;
    pending_keyboard_direction_bits_ = 0;
    current_xinput_direction_bits_ = 0;
    pending_xinput_direction_bits_ = 0;
    motion_assist_polls_remaining_ = 0;
    xinput_motion_assist_polls_remaining_ = 0;

    if (paused_) {
        paused_ = false;
        emit pausedChanged(false);
    }
}

bool LibretroCore::reset() {
    if (!game_loaded_ || !retro_reset_) {
        setError(QStringLiteral("No running game is available to reset."));
        return false;
    }

    const bool resume_after_reset = !paused_;
    frame_timer_.stop();
    audio_.stop();

    raw_keyboard_joypad_state_.fill(false);
    keyboard_joypad_state_.fill(false);
    raw_xinput_joypad_state_.fill(false);
    xinput_joypad_state_.fill(false);
    current_keyboard_direction_bits_ = 0;
    pending_keyboard_direction_bits_ = 0;
    current_xinput_direction_bits_ = 0;
    pending_xinput_direction_bits_ = 0;
    motion_assist_polls_remaining_ = 0;
    xinput_motion_assist_polls_remaining_ = 0;

    if (video_output_)
        video_output_->clearFrame();

    retro_reset_();

    if (resume_after_reset) {
        if (!audio_.start(output_sample_rate_))
            qWarning().noquote() << "WASAPI shared audio disabled:" << audio_.lastError();
        frame_timer_.start(16);
    }

    return true;
}

QString LibretroCore::lastError() const {
    return last_error_;
}

QString LibretroCore::displayName() const {
    return QStringLiteral("Libretro");
}

QString LibretroCore::coreFileName() const {
    return QString();
}

QString LibretroCore::romDirectoryName() const {
    return QString();
}

QStringList LibretroCore::supportedExtensions() const {
    return {};
}

bool LibretroCore::coreOptionValue(const QByteArray &, const char *&) const {
    return false;
}

void LibretroCore::coreOptionsUpdated(const retro_core_options_v2 *) {
}

void LibretroCore::setPaused(bool paused) {
    if (paused_ == paused)
        return;

    paused_ = paused;
    raw_keyboard_joypad_state_.fill(false);
    keyboard_joypad_state_.fill(false);
    raw_xinput_joypad_state_.fill(false);
    xinput_joypad_state_.fill(false);
    current_keyboard_direction_bits_ = 0;
    pending_keyboard_direction_bits_ = 0;
    current_xinput_direction_bits_ = 0;
    pending_xinput_direction_bits_ = 0;
    motion_assist_polls_remaining_ = 0;
    xinput_motion_assist_polls_remaining_ = 0;

    if (paused_) {
        frame_timer_.stop();
        audio_.stop();
    } else if (game_loaded_) {
        if (!audio_.start(output_sample_rate_))
            qWarning().noquote() << "WASAPI shared audio disabled:" << audio_.lastError();
        frame_timer_.start(16);
    }

    emit pausedChanged(paused_);
}

bool LibretroCore::isPaused() const {
    return paused_;
}

bool LibretroCore::isGameLoaded() const {
    return game_loaded_;
}

bool LibretroCore::saveState(const QString &statePath) {
    if (!game_loaded_ || !retro_serialize_size_ || !retro_serialize_) {
        setError(QStringLiteral("No running game is available for save state."));
        return false;
    }

    const size_t state_size = retro_serialize_size_();
    if (state_size == 0) {
        setError(QStringLiteral("Core reported an empty save state."));
        return false;
    }

    QByteArray state;
    state.resize(static_cast<qsizetype>(state_size));

    const bool timer_was_active = frame_timer_.isActive();
    frame_timer_.stop();

    const bool serialized = retro_serialize_(state.data(), state_size);

    if (timer_was_active && !paused_)
        frame_timer_.start(16);

    if (!serialized) {
        setError(QStringLiteral("Core failed to serialize save state."));
        return false;
    }

    const QFileInfo file_info(statePath);
    QDir().mkpath(file_info.absolutePath());

    QSaveFile file(statePath);
    if (!file.open(QIODevice::WriteOnly)) {
        setError(QStringLiteral("Could not open save state file: %1").arg(file.errorString()));
        return false;
    }

    if (file.write(state) != state.size()) {
        setError(QStringLiteral("Could not write save state file: %1").arg(file.errorString()));
        return false;
    }

    if (!file.commit()) {
        setError(QStringLiteral("Could not commit save state file: %1").arg(file.errorString()));
        return false;
    }

    return true;
}

bool LibretroCore::loadState(const QString &statePath) {
    if (!game_loaded_ || !retro_unserialize_) {
        setError(QStringLiteral("No running game is available for load state."));
        return false;
    }

    QFile file(statePath);
    if (!file.open(QIODevice::ReadOnly)) {
        setError(QStringLiteral("Could not open save state file: %1").arg(file.errorString()));
        return false;
    }

    const QByteArray state = file.readAll();
    if (state.isEmpty()) {
        setError(QStringLiteral("Save state file is empty."));
        return false;
    }

    const bool timer_was_active = frame_timer_.isActive();
    frame_timer_.stop();

    const bool unserialized = retro_unserialize_(state.constData(), static_cast<size_t>(state.size()));

    if (timer_was_active && !paused_)
        frame_timer_.start(16);

    if (!unserialized) {
        setError(QStringLiteral("Core failed to restore save state."));
        return false;
    }

    return true;
}

bool LibretroCore::readSystemRam(QByteArray &ram) const {
    ram.clear();

    if (!game_loaded_ || !retro_get_memory_data_ || !retro_get_memory_size_)
        return false;

    auto *memory = static_cast<const char *>(retro_get_memory_data_(RETRO_MEMORY_SYSTEM_RAM));
    const size_t memory_size = retro_get_memory_size_(RETRO_MEMORY_SYSTEM_RAM);
    if (!memory || memory_size == 0)
        return false;

    ram = QByteArray(memory, static_cast<qsizetype>(memory_size));
    return true;
}

bool LibretroCore::readSystemRamByte(uint32_t address, uint8_t &value) const {
    if (!game_loaded_ || !retro_get_memory_data_ || !retro_get_memory_size_)
        return false;

    auto *memory = static_cast<const uint8_t *>(retro_get_memory_data_(RETRO_MEMORY_SYSTEM_RAM));
    const size_t memory_size = retro_get_memory_size_(RETRO_MEMORY_SYSTEM_RAM);
    if (!memory || address >= memory_size)
        return false;

    value = memory[address];
    return true;
}

int LibretroCore::keyBinding(unsigned retroButtonId) const {
    if (retroButtonId >= key_bindings_.size())
        return 0;

    return key_bindings_[retroButtonId];
}

void LibretroCore::setKeyBinding(unsigned retroButtonId, int key) {
    if (retroButtonId >= key_bindings_.size())
        return;

    key_bindings_[retroButtonId] = key;
    raw_keyboard_joypad_state_[retroButtonId] = false;
    keyboard_joypad_state_[retroButtonId] = false;
}

std::array<int, 16> LibretroCore::keyBindings() const {
    return key_bindings_;
}

void LibretroCore::setKeyBindings(const std::array<int, 16> &bindings) {
    key_bindings_ = bindings;
    raw_keyboard_joypad_state_.fill(false);
    keyboard_joypad_state_.fill(false);
    current_keyboard_direction_bits_ = 0;
    pending_keyboard_direction_bits_ = 0;
    motion_assist_polls_remaining_ = 0;
}

int LibretroCore::xinputBinding(unsigned retroButtonId) const {
    if (retroButtonId >= xinput_bindings_.size())
        return XInputNone;

    return xinput_bindings_[retroButtonId];
}

void LibretroCore::setXInputBinding(unsigned retroButtonId, int control) {
    if (retroButtonId >= xinput_bindings_.size())
        return;

    xinput_bindings_[retroButtonId] = control;
    raw_xinput_joypad_state_[retroButtonId] = false;
    xinput_joypad_state_[retroButtonId] = false;
}

std::array<int, 16> LibretroCore::xinputBindings() const {
    return xinput_bindings_;
}

void LibretroCore::setXInputBindings(const std::array<int, 16> &bindings) {
    xinput_bindings_ = bindings;
    raw_xinput_joypad_state_.fill(false);
    xinput_joypad_state_.fill(false);
    current_xinput_direction_bits_ = 0;
    pending_xinput_direction_bits_ = 0;
    xinput_motion_assist_polls_remaining_ = 0;
}

void LibretroCore::loadInputConfiguration(const QString &iniPath) {
    QSettings settings(iniPath, QSettings::IniFormat);

    settings.beginGroup(QStringLiteral("Input"));
    arcade_socd_clean_ = settings.value(QStringLiteral("ArcadeSocdClean"), arcade_socd_clean_).toBool();
    keyboard_motion_assist_ = settings.value(QStringLiteral("KeyboardMotionAssist"), keyboard_motion_assist_).toBool();
    settings.endGroup();

    for (unsigned id = 0; id < key_bindings_.size(); ++id) {
        const QString name = bindingSettingName(id);

        settings.beginGroup(QStringLiteral("Keyboard"));
        key_bindings_[id] = settings.value(name, key_bindings_[id]).toInt();
        settings.endGroup();

        settings.beginGroup(QStringLiteral("XInput"));
        xinput_bindings_[id] = settings.value(name, xinput_bindings_[id]).toInt();
        settings.endGroup();
    }

    raw_keyboard_joypad_state_.fill(false);
    keyboard_joypad_state_.fill(false);
    raw_xinput_joypad_state_.fill(false);
    xinput_joypad_state_.fill(false);
    current_keyboard_direction_bits_ = 0;
    pending_keyboard_direction_bits_ = 0;
    current_xinput_direction_bits_ = 0;
    pending_xinput_direction_bits_ = 0;
    motion_assist_polls_remaining_ = 0;
    xinput_motion_assist_polls_remaining_ = 0;
}

void LibretroCore::saveInputConfiguration(const QString &iniPath) const {
    QFileInfo file_info(iniPath);
    QDir().mkpath(file_info.absolutePath());

    QSettings settings(iniPath, QSettings::IniFormat);

    settings.beginGroup(QStringLiteral("Input"));
    settings.setValue(QStringLiteral("ArcadeSocdClean"), arcade_socd_clean_);
    settings.setValue(QStringLiteral("KeyboardMotionAssist"), keyboard_motion_assist_);
    settings.endGroup();

    settings.beginGroup(QStringLiteral("Keyboard"));
    for (unsigned id = 0; id < key_bindings_.size(); ++id)
        settings.setValue(bindingSettingName(id), key_bindings_[id]);
    settings.endGroup();

    settings.beginGroup(QStringLiteral("XInput"));
    for (unsigned id = 0; id < xinput_bindings_.size(); ++id)
        settings.setValue(bindingSettingName(id), xinput_bindings_[id]);
    settings.endGroup();

    settings.sync();
}

bool LibretroCore::arcadeSocdClean() const {
    return arcade_socd_clean_;
}

void LibretroCore::setArcadeSocdClean(bool enabled) {
    if (arcade_socd_clean_ == enabled)
        return;

    arcade_socd_clean_ = enabled;
    applyKeyboardInputState();
    applyXInputInputState();
}

bool LibretroCore::keyboardMotionAssist() const {
    return keyboard_motion_assist_;
}

void LibretroCore::setKeyboardMotionAssist(bool enabled) {
    if (keyboard_motion_assist_ == enabled)
        return;

    keyboard_motion_assist_ = enabled;
    motion_assist_polls_remaining_ = 0;
    xinput_motion_assist_polls_remaining_ = 0;
    applyKeyboardInputState();
    applyXInputInputState();
}

QString LibretroCore::systemRegionOption() const {
    return QString::fromStdString(system_region_option_);
}

void LibretroCore::setSystemRegionOption(const QString &region) {
    system_region_option_ = region.toStdString();
}

QString LibretroCore::systemModeOption() const {
    return QString::fromStdString(system_mode_option_);
}

void LibretroCore::setSystemModeOption(const QString &mode) {
    system_mode_option_ = mode.toStdString();
}

QString LibretroCore::fbneoCpuClockOption() const {
    return QString::fromStdString(fbneo_cpu_clock_option_);
}

void LibretroCore::setFbneoCpuClockOption(const QString &cpuClock) {
    fbneo_cpu_clock_option_ = cpuClock.toStdString();
}

QString LibretroCore::xinputControlName(int control) {
    switch (control) {
    case XInputNone:
        return QStringLiteral("Unbound");
    case XInputDpadUp:
        return QStringLiteral("D-Pad Up");
    case XInputDpadDown:
        return QStringLiteral("D-Pad Down");
    case XInputDpadLeft:
        return QStringLiteral("D-Pad Left");
    case XInputDpadRight:
        return QStringLiteral("D-Pad Right");
    case XInputStart:
        return QStringLiteral("Start");
    case XInputBack:
        return QStringLiteral("Back");
    case XInputA:
        return QStringLiteral("A");
    case XInputB:
        return QStringLiteral("B");
    case XInputX:
        return QStringLiteral("X");
    case XInputY:
        return QStringLiteral("Y");
    case XInputLeftShoulder:
        return QStringLiteral("LB");
    case XInputRightShoulder:
        return QStringLiteral("RB");
    case XInputLeftTrigger:
        return QStringLiteral("LT");
    case XInputRightTrigger:
        return QStringLiteral("RT");
    case XInputLeftThumbUp:
        return QStringLiteral("Left Stick Up");
    case XInputLeftThumbDown:
        return QStringLiteral("Left Stick Down");
    case XInputLeftThumbLeft:
        return QStringLiteral("Left Stick Left");
    case XInputLeftThumbRight:
        return QStringLiteral("Left Stick Right");
    default:
        return QStringLiteral("Unknown");
    }
}

int LibretroCore::firstPressedXInputControl(unsigned userIndex) {
    XINPUT_STATE state {};
    if (XInputGetState(userIndex, &state) != ERROR_SUCCESS)
        return XInputNone;

    constexpr int controls[] = {
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

    for (int control : controls) {
        if (isXInputControlPressed(control, &state))
            return control;
    }

    return XInputNone;
}

bool LibretroCore::eventFilter(QObject *watched, QEvent *event) {
    if (QApplication::activeModalWidget())
        return QObject::eventFilter(watched, event);

    if (event->type() != QEvent::KeyPress && event->type() != QEvent::KeyRelease)
        return QObject::eventFilter(watched, event);

    auto *keyEvent = static_cast<QKeyEvent *>(event);
    if (keyEvent->isAutoRepeat())
        return true;

    const int button = buttonForKey(keyEvent->key());
    if (button < 0)
        return QObject::eventFilter(watched, event);

    raw_keyboard_joypad_state_[static_cast<size_t>(button)] = event->type() == QEvent::KeyPress;
    applyKeyboardInputState();
    return true;
}

void LibretroCore::runFrame() {
    if (game_loaded_ && !paused_ && retro_run_) {
        retro_run_();
        finishKeyboardMotionAssist();
        finishXInputMotionAssist();
        emit frameAdvanced();
    }
}

bool LibretroCore::environmentCallback(unsigned command, void *data) {
    return active_core_ ? active_core_->environment(command, data) : false;
}

void LibretroCore::videoCallback(const void *data, unsigned width, unsigned height, size_t pitch) {
    if (active_core_)
        active_core_->video(data, width, height, pitch);
}

void LibretroCore::audioSampleCallback(int16_t, int16_t) {
}

size_t LibretroCore::audioBatchCallback(const int16_t *data, size_t frames) {
    return active_core_ ? active_core_->audioBatch(data, frames) : 0;
}

void LibretroCore::inputPollCallback() {
    if (active_core_)
        active_core_->pollXInput();
}

int16_t LibretroCore::inputStateCallback(unsigned port, unsigned device, unsigned index, unsigned id) {
    return active_core_ ? active_core_->inputState(port, device, index, id) : 0;
}

void LibretroCore::logCallback(retro_log_level level, const char *format, ...) {
    char buffer[2048] {};

    va_list args;
    va_start(args, format);
    std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    switch (level) {
    case RETRO_LOG_ERROR:
        qWarning().noquote() << "[neocd]" << buffer;
        break;
    case RETRO_LOG_WARN:
        qWarning().noquote() << "[neocd]" << buffer;
        break;
    default:
        qInfo().noquote() << "[neocd]" << buffer;
        break;
    }
}

bool LibretroCore::environment(unsigned command, void *data) {
    switch (command) {
    case RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY:
        *static_cast<const char **>(data) = system_directory_utf8_.constData();
        return true;
    case RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY:
        *static_cast<const char **>(data) = save_directory_utf8_.constData();
        return true;
    case RETRO_ENVIRONMENT_GET_LOG_INTERFACE:
        static_cast<retro_log_callback *>(data)->log = logCallback;
        return true;
    case RETRO_ENVIRONMENT_GET_CORE_OPTIONS_VERSION:
        *static_cast<unsigned *>(data) = 2;
        return true;
    case RETRO_ENVIRONMENT_SET_PIXEL_FORMAT:
        return *static_cast<const retro_pixel_format *>(data) == RETRO_PIXEL_FORMAT_RGB565;
    case RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE:
        *static_cast<bool *>(data) = false;
        return true;
    case RETRO_ENVIRONMENT_GET_VARIABLE:
    {
        auto *variable = static_cast<retro_variable *>(data);
        if (!variable || !variable->key)
            return false;

        const QByteArray key(variable->key);
        const char *value = nullptr;
        if (coreOptionValue(key, value)) {
            variable->value = value;
            if (key.startsWith("fbneo-neogeo") || key == "fbneo-cpu-speed-adjust" || key == "neocd_region")
                qInfo().noquote() << "Libretro option" << key << "=" << value;
            return true;
        }
        if (key.startsWith("fbneo-neogeo") || key == "fbneo-cpu-speed-adjust" || key == "neocd_region")
            qInfo().noquote() << "Unsupported libretro option request" << key;
        return false;
    }
    case RETRO_ENVIRONMENT_SET_CORE_OPTIONS_V2:
    {
        auto *options = static_cast<const retro_core_options_v2 *>(data);
        if (!options || !options->definitions)
            return true;

        coreOptionsUpdated(options);
        return true;
    }
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

void LibretroCore::video(const void *data, unsigned width, unsigned height, size_t pitch) {
    if (!video_output_ || !data || width == 0 || height == 0)
        return;

    video_output_->submitFrame(data,
                              static_cast<int>(width),
                              static_cast<int>(height),
                              static_cast<int>(pitch),
                              EmulatorView::PixelFormat::Rgb565);
}

size_t LibretroCore::audioBatch(const int16_t *data, size_t frames) {
    if (!resampler_ || !data || frames == 0)
        return 0;

    input_float_buffer_.resize(frames * 2);
    constexpr float int16_scale = 1.0f / 32768.0f;
    for (size_t i = 0; i < frames * 2; ++i)
        input_float_buffer_[i] = static_cast<float>(data[i]) * int16_scale;

    const double ratio = static_cast<double>(output_sample_rate_) / static_cast<double>(source_sample_rate_);
    const auto output_frames_capacity = static_cast<size_t>(std::ceil(static_cast<double>(frames) * ratio)) + 32;
    output_float_buffer_.resize(output_frames_capacity * 2);

    SRC_DATA src_data {};
    src_data.data_in = input_float_buffer_.data();
    src_data.input_frames = static_cast<long>(frames);
    src_data.data_out = output_float_buffer_.data();
    src_data.output_frames = static_cast<long>(output_frames_capacity);
    src_data.src_ratio = ratio;

    const int result = src_process(resampler_, &src_data);
    if (result != 0) {
        qWarning().noquote() << "libsamplerate failed:" << src_strerror(result);
        return 0;
    }

    audio_.writeSamples(output_float_buffer_.data(), static_cast<size_t>(src_data.output_frames_gen));
    return frames;
}

int16_t LibretroCore::inputState(unsigned port, unsigned device, unsigned, unsigned id) const {
    if (port != 0 || device != RETRO_DEVICE_JOYPAD || id >= keyboard_joypad_state_.size())
        return 0;

    const auto index = static_cast<size_t>(id);
    return (keyboard_joypad_state_[index] || xinput_joypad_state_[index]) ? 1 : 0;
}

uint8_t LibretroCore::rawKeyboardDirectionBits() const {
    uint8_t bits = 0;
    if (raw_keyboard_joypad_state_[RETRO_DEVICE_ID_JOYPAD_UP])
        bits |= DirectionUp;
    if (raw_keyboard_joypad_state_[RETRO_DEVICE_ID_JOYPAD_DOWN])
        bits |= DirectionDown;
    if (raw_keyboard_joypad_state_[RETRO_DEVICE_ID_JOYPAD_LEFT])
        bits |= DirectionLeft;
    if (raw_keyboard_joypad_state_[RETRO_DEVICE_ID_JOYPAD_RIGHT])
        bits |= DirectionRight;
    return bits;
}

uint8_t LibretroCore::rawXInputDirectionBits() const {
    uint8_t bits = 0;
    if (raw_xinput_joypad_state_[RETRO_DEVICE_ID_JOYPAD_UP])
        bits |= DirectionUp;
    if (raw_xinput_joypad_state_[RETRO_DEVICE_ID_JOYPAD_DOWN])
        bits |= DirectionDown;
    if (raw_xinput_joypad_state_[RETRO_DEVICE_ID_JOYPAD_LEFT])
        bits |= DirectionLeft;
    if (raw_xinput_joypad_state_[RETRO_DEVICE_ID_JOYPAD_RIGHT])
        bits |= DirectionRight;
    return bits;
}

uint8_t LibretroCore::cleanedDirectionBits(uint8_t directionBits) const {
    if (!arcade_socd_clean_)
        return directionBits;

    if ((directionBits & (DirectionUp | DirectionDown)) == (DirectionUp | DirectionDown))
        directionBits &= static_cast<uint8_t>(~(DirectionUp | DirectionDown));
    if ((directionBits & (DirectionLeft | DirectionRight)) == (DirectionLeft | DirectionRight))
        directionBits &= static_cast<uint8_t>(~(DirectionLeft | DirectionRight));

    return directionBits;
}

void LibretroCore::setKeyboardDirectionBits(uint8_t directionBits) {
    keyboard_joypad_state_[RETRO_DEVICE_ID_JOYPAD_UP] = (directionBits & DirectionUp) != 0;
    keyboard_joypad_state_[RETRO_DEVICE_ID_JOYPAD_DOWN] = (directionBits & DirectionDown) != 0;
    keyboard_joypad_state_[RETRO_DEVICE_ID_JOYPAD_LEFT] = (directionBits & DirectionLeft) != 0;
    keyboard_joypad_state_[RETRO_DEVICE_ID_JOYPAD_RIGHT] = (directionBits & DirectionRight) != 0;
}

void LibretroCore::setXInputDirectionBits(uint8_t directionBits) {
    xinput_joypad_state_[RETRO_DEVICE_ID_JOYPAD_UP] = (directionBits & DirectionUp) != 0;
    xinput_joypad_state_[RETRO_DEVICE_ID_JOYPAD_DOWN] = (directionBits & DirectionDown) != 0;
    xinput_joypad_state_[RETRO_DEVICE_ID_JOYPAD_LEFT] = (directionBits & DirectionLeft) != 0;
    xinput_joypad_state_[RETRO_DEVICE_ID_JOYPAD_RIGHT] = (directionBits & DirectionRight) != 0;
}

void LibretroCore::applyKeyboardInputState() {
    keyboard_joypad_state_ = raw_keyboard_joypad_state_;

    const uint8_t previous_bits = current_keyboard_direction_bits_;
    const uint8_t next_bits = cleanedDirectionBits(rawKeyboardDirectionBits());
    pending_keyboard_direction_bits_ = next_bits;

    uint8_t effective_bits = next_bits;
    motion_assist_polls_remaining_ = 0;

    if (keyboard_motion_assist_) {
        const bool previous_down_diagonal = (previous_bits & DirectionDown) != 0 &&
                                            ((previous_bits & DirectionLeft) != 0 || (previous_bits & DirectionRight) != 0);
        const bool next_down_diagonal = (next_bits & DirectionDown) != 0 &&
                                        ((next_bits & DirectionLeft) != 0 || (next_bits & DirectionRight) != 0);
        const bool previous_up_diagonal = (previous_bits & DirectionUp) != 0 &&
                                          ((previous_bits & DirectionLeft) != 0 || (previous_bits & DirectionRight) != 0);
        const bool next_up_diagonal = (next_bits & DirectionUp) != 0 &&
                                      ((next_bits & DirectionLeft) != 0 || (next_bits & DirectionRight) != 0);
        const bool previous_left_diagonal = (previous_bits & DirectionLeft) != 0 &&
                                            ((previous_bits & DirectionUp) != 0 || (previous_bits & DirectionDown) != 0);
        const bool next_left_diagonal = (next_bits & DirectionLeft) != 0 &&
                                        ((next_bits & DirectionUp) != 0 || (next_bits & DirectionDown) != 0);
        const bool previous_right_diagonal = (previous_bits & DirectionRight) != 0 &&
                                             ((previous_bits & DirectionUp) != 0 || (previous_bits & DirectionDown) != 0);
        const bool next_right_diagonal = (next_bits & DirectionRight) != 0 &&
                                         ((next_bits & DirectionUp) != 0 || (next_bits & DirectionDown) != 0);

        if (previous_down_diagonal && next_down_diagonal &&
            ((previous_bits ^ next_bits) & (DirectionLeft | DirectionRight)) == (DirectionLeft | DirectionRight)) {
            effective_bits = DirectionDown;
        } else if (previous_up_diagonal && next_up_diagonal &&
                   ((previous_bits ^ next_bits) & (DirectionLeft | DirectionRight)) == (DirectionLeft | DirectionRight)) {
            effective_bits = DirectionUp;
        } else if (previous_left_diagonal && next_left_diagonal &&
                   ((previous_bits ^ next_bits) & (DirectionUp | DirectionDown)) == (DirectionUp | DirectionDown)) {
            effective_bits = DirectionLeft;
        } else if (previous_right_diagonal && next_right_diagonal &&
                   ((previous_bits ^ next_bits) & (DirectionUp | DirectionDown)) == (DirectionUp | DirectionDown)) {
            effective_bits = DirectionRight;
        }

        if (effective_bits != next_bits)
            motion_assist_polls_remaining_ = 1;
    }

    current_keyboard_direction_bits_ = effective_bits;
    setKeyboardDirectionBits(effective_bits);
}

void LibretroCore::applyXInputInputState() {
    xinput_joypad_state_ = raw_xinput_joypad_state_;

    const uint8_t previous_bits = current_xinput_direction_bits_;
    const uint8_t next_bits = cleanedDirectionBits(rawXInputDirectionBits());
    pending_xinput_direction_bits_ = next_bits;

    uint8_t effective_bits = next_bits;
    xinput_motion_assist_polls_remaining_ = 0;

    if (keyboard_motion_assist_) {
        const bool previous_down_diagonal = (previous_bits & DirectionDown) != 0 &&
                                            ((previous_bits & DirectionLeft) != 0 || (previous_bits & DirectionRight) != 0);
        const bool next_down_diagonal = (next_bits & DirectionDown) != 0 &&
                                        ((next_bits & DirectionLeft) != 0 || (next_bits & DirectionRight) != 0);
        const bool previous_up_diagonal = (previous_bits & DirectionUp) != 0 &&
                                          ((previous_bits & DirectionLeft) != 0 || (previous_bits & DirectionRight) != 0);
        const bool next_up_diagonal = (next_bits & DirectionUp) != 0 &&
                                      ((next_bits & DirectionLeft) != 0 || (next_bits & DirectionRight) != 0);
        const bool previous_left_diagonal = (previous_bits & DirectionLeft) != 0 &&
                                            ((previous_bits & DirectionUp) != 0 || (previous_bits & DirectionDown) != 0);
        const bool next_left_diagonal = (next_bits & DirectionLeft) != 0 &&
                                        ((next_bits & DirectionUp) != 0 || (next_bits & DirectionDown) != 0);
        const bool previous_right_diagonal = (previous_bits & DirectionRight) != 0 &&
                                             ((previous_bits & DirectionUp) != 0 || (previous_bits & DirectionDown) != 0);
        const bool next_right_diagonal = (next_bits & DirectionRight) != 0 &&
                                         ((next_bits & DirectionUp) != 0 || (next_bits & DirectionDown) != 0);

        if (previous_down_diagonal && next_down_diagonal &&
            ((previous_bits ^ next_bits) & (DirectionLeft | DirectionRight)) == (DirectionLeft | DirectionRight)) {
            effective_bits = DirectionDown;
        } else if (previous_up_diagonal && next_up_diagonal &&
                   ((previous_bits ^ next_bits) & (DirectionLeft | DirectionRight)) == (DirectionLeft | DirectionRight)) {
            effective_bits = DirectionUp;
        } else if (previous_left_diagonal && next_left_diagonal &&
                   ((previous_bits ^ next_bits) & (DirectionUp | DirectionDown)) == (DirectionUp | DirectionDown)) {
            effective_bits = DirectionLeft;
        } else if (previous_right_diagonal && next_right_diagonal &&
                   ((previous_bits ^ next_bits) & (DirectionUp | DirectionDown)) == (DirectionUp | DirectionDown)) {
            effective_bits = DirectionRight;
        }

        if (effective_bits != next_bits)
            xinput_motion_assist_polls_remaining_ = 1;
    }

    current_xinput_direction_bits_ = effective_bits;
    setXInputDirectionBits(effective_bits);
}

void LibretroCore::finishKeyboardMotionAssist() {
    if (motion_assist_polls_remaining_ <= 0)
        return;

    --motion_assist_polls_remaining_;
    if (motion_assist_polls_remaining_ > 0)
        return;

    current_keyboard_direction_bits_ = pending_keyboard_direction_bits_;
    keyboard_joypad_state_ = raw_keyboard_joypad_state_;
    setKeyboardDirectionBits(pending_keyboard_direction_bits_);
}

void LibretroCore::finishXInputMotionAssist() {
    if (xinput_motion_assist_polls_remaining_ <= 0)
        return;

    --xinput_motion_assist_polls_remaining_;
    if (xinput_motion_assist_polls_remaining_ > 0)
        return;

    current_xinput_direction_bits_ = pending_xinput_direction_bits_;
    xinput_joypad_state_ = raw_xinput_joypad_state_;
    setXInputDirectionBits(pending_xinput_direction_bits_);
}

void LibretroCore::setError(const QString &message) {
    last_error_ = message;
    qWarning().noquote() << message;
}

bool LibretroCore::resolveSymbols() {
    retro_set_environment_ = reinterpret_cast<retro_set_environment_t>(library_.resolve("retro_set_environment"));
    retro_set_video_refresh_ = reinterpret_cast<retro_set_video_refresh_t>(library_.resolve("retro_set_video_refresh"));
    retro_set_audio_sample_ = reinterpret_cast<retro_set_audio_sample_t>(library_.resolve("retro_set_audio_sample"));
    retro_set_audio_sample_batch_ = reinterpret_cast<retro_set_audio_sample_batch_t>(library_.resolve("retro_set_audio_sample_batch"));
    retro_set_input_poll_ = reinterpret_cast<retro_set_input_poll_t>(library_.resolve("retro_set_input_poll"));
    retro_set_input_state_ = reinterpret_cast<retro_set_input_state_t>(library_.resolve("retro_set_input_state"));
    retro_init_ = reinterpret_cast<retro_init_t>(library_.resolve("retro_init"));
    retro_deinit_ = reinterpret_cast<retro_deinit_t>(library_.resolve("retro_deinit"));
    retro_reset_ = reinterpret_cast<retro_reset_t>(library_.resolve("retro_reset"));
    retro_load_game_ = reinterpret_cast<retro_load_game_t>(library_.resolve("retro_load_game"));
    retro_unload_game_ = reinterpret_cast<retro_unload_game_t>(library_.resolve("retro_unload_game"));
    retro_run_ = reinterpret_cast<retro_run_t>(library_.resolve("retro_run"));
    retro_get_system_info_ = reinterpret_cast<retro_get_system_info_t>(library_.resolve("retro_get_system_info"));
    retro_get_system_av_info_ = reinterpret_cast<retro_get_system_av_info_t>(library_.resolve("retro_get_system_av_info"));
    retro_serialize_size_ = reinterpret_cast<retro_serialize_size_t>(library_.resolve("retro_serialize_size"));
    retro_serialize_ = reinterpret_cast<retro_serialize_t>(library_.resolve("retro_serialize"));
    retro_unserialize_ = reinterpret_cast<retro_unserialize_t>(library_.resolve("retro_unserialize"));
    retro_get_memory_data_ = reinterpret_cast<retro_get_memory_data_t>(library_.resolve("retro_get_memory_data"));
    retro_get_memory_size_ = reinterpret_cast<retro_get_memory_size_t>(library_.resolve("retro_get_memory_size"));

    if (!retro_set_environment_ || !retro_set_video_refresh_ || !retro_set_audio_sample_ ||
        !retro_set_audio_sample_batch_ || !retro_set_input_poll_ || !retro_set_input_state_ ||
        !retro_init_ || !retro_deinit_ || !retro_reset_ || !retro_load_game_ || !retro_unload_game_ ||
        !retro_run_ || !retro_get_system_info_ || !retro_get_system_av_info_ ||
        !retro_serialize_size_ || !retro_serialize_ || !retro_unserialize_) {
        setError(QStringLiteral("%1 is missing required libretro exports.").arg(coreFileName()));
        resetSymbols();
        return false;
    }

    return true;
}

void LibretroCore::resetSymbols() {
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

void LibretroCore::installCallbacks() {
    retro_set_environment_(environmentCallback);
    retro_set_video_refresh_(videoCallback);
    retro_set_audio_sample_(audioSampleCallback);
    retro_set_audio_sample_batch_(audioBatchCallback);
    retro_set_input_poll_(inputPollCallback);
    retro_set_input_state_(inputStateCallback);
}

QByteArray LibretroCore::pathToUtf8(const QString &path) const {
    return QDir::toNativeSeparators(QFileInfo(path).absoluteFilePath()).toUtf8();
}

void LibretroCore::pollXInput() {
    raw_xinput_joypad_state_.fill(false);
    xinput_joypad_state_.fill(false);

    if (QApplication::activeModalWidget()) {
        current_xinput_direction_bits_ = 0;
        pending_xinput_direction_bits_ = 0;
        xinput_motion_assist_polls_remaining_ = 0;
        return;
    }

    XINPUT_STATE state {};
    if (::XInputGetState(xinput_user_index_, &state) != ERROR_SUCCESS) {
        current_xinput_direction_bits_ = 0;
        pending_xinput_direction_bits_ = 0;
        xinput_motion_assist_polls_remaining_ = 0;
        return;
    }

    for (size_t id = 0; id < raw_xinput_joypad_state_.size(); ++id)
        raw_xinput_joypad_state_[id] = isXInputControlPressed(xinput_bindings_[id], &state);

    applyXInputInputState();
}

bool LibretroCore::isXInputControlPressed(int control, const void *statePtr) {
    const auto &state = *static_cast<const XINPUT_STATE *>(statePtr);
    const WORD buttons = state.Gamepad.wButtons;

    switch (control) {
    case XInputDpadUp:
        return (buttons & XINPUT_GAMEPAD_DPAD_UP) != 0;
    case XInputDpadDown:
        return (buttons & XINPUT_GAMEPAD_DPAD_DOWN) != 0;
    case XInputDpadLeft:
        return (buttons & XINPUT_GAMEPAD_DPAD_LEFT) != 0;
    case XInputDpadRight:
        return (buttons & XINPUT_GAMEPAD_DPAD_RIGHT) != 0;
    case XInputStart:
        return (buttons & XINPUT_GAMEPAD_START) != 0;
    case XInputBack:
        return (buttons & XINPUT_GAMEPAD_BACK) != 0;
    case XInputA:
        return (buttons & XINPUT_GAMEPAD_A) != 0;
    case XInputB:
        return (buttons & XINPUT_GAMEPAD_B) != 0;
    case XInputX:
        return (buttons & XINPUT_GAMEPAD_X) != 0;
    case XInputY:
        return (buttons & XINPUT_GAMEPAD_Y) != 0;
    case XInputLeftShoulder:
        return (buttons & XINPUT_GAMEPAD_LEFT_SHOULDER) != 0;
    case XInputRightShoulder:
        return (buttons & XINPUT_GAMEPAD_RIGHT_SHOULDER) != 0;
    case XInputLeftTrigger:
        return state.Gamepad.bLeftTrigger > XINPUT_GAMEPAD_TRIGGER_THRESHOLD;
    case XInputRightTrigger:
        return state.Gamepad.bRightTrigger > XINPUT_GAMEPAD_TRIGGER_THRESHOLD;
    case XInputLeftThumbUp:
        return state.Gamepad.sThumbLY > XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE;
    case XInputLeftThumbDown:
        return state.Gamepad.sThumbLY < -XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE;
    case XInputLeftThumbLeft:
        return state.Gamepad.sThumbLX < -XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE;
    case XInputLeftThumbRight:
        return state.Gamepad.sThumbLX > XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE;
    default:
        return false;
    }
}

bool LibretroCore::initializeResampler(int sourceSampleRate) {
    releaseResampler();

    if (sourceSampleRate <= 0) {
        setError(QStringLiteral("Invalid core audio sample rate."));
        return false;
    }

    source_sample_rate_ = sourceSampleRate;

    int error = 0;
    resampler_ = src_new(SRC_SINC_BEST_QUALITY, 2, &error);
    if (!resampler_) {
        setError(QStringLiteral("Could not create libsamplerate state: %1").arg(src_strerror(error)));
        return false;
    }

    return true;
}

void LibretroCore::releaseResampler() {
    if (resampler_) {
        src_delete(resampler_);
        resampler_ = nullptr;
    }

    source_sample_rate_ = 0;
    input_float_buffer_.clear();
    output_float_buffer_.clear();
}

int LibretroCore::buttonForKey(int key) const {
    for (size_t index = 0; index < key_bindings_.size(); ++index) {
        const int binding = key_bindings_[index];
        if (binding != 0 && binding == key)
            return static_cast<int>(index);
    }

    return -1;
}
