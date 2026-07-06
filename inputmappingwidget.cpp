#include "inputmappingwidget.h"

#include "libretrocore.h"

#include <QGridLayout>
#include <QKeyEvent>
#include <QKeySequence>
#include <QLabel>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>

#include "libretro.h"

namespace {
QString keyName(int key) {
    if (key == 0)
        return QStringLiteral("Unbound");

    const QString text = QKeySequence(key).toString(QKeySequence::NativeText);
    return text.isEmpty() ? QString::number(key) : text;
}
}

InputMappingWidget::InputMappingWidget(LibretroCore *core, QWidget *parent)
    : QWidget(parent)
    , core_(core) {
    setFocusPolicy(Qt::StrongFocus);
    buildUi();

    xinput_timer_ = new QTimer(this);
    xinput_timer_->setInterval(10);
	xinput_timer_->setTimerType(Qt::PreciseTimer);
    connect(xinput_timer_, &QTimer::timeout, this, &InputMappingWidget::pollXInputCapture);
    xinput_timer_->start();

    refreshButtons();
}

void InputMappingWidget::keyPressEvent(QKeyEvent *event) {
    if (capture_index_ < 0) {
        QWidget::keyPressEvent(event);
        return;
    }

    if (event->key() == Qt::Key_Escape) {
        capture_index_ = -1;
        capture_type_ = CaptureType::None;
        refreshButtons();
        return;
    }

    if (capture_type_ == CaptureType::XInput) {
        if (event->key() == Qt::Key_Delete || event->key() == Qt::Key_Backspace) {
            if (core_)
                core_->setXInputBinding(bindings_[static_cast<size_t>(capture_index_)].retro_button_id, LibretroCore::XInputNone);
            capture_index_ = -1;
            capture_type_ = CaptureType::None;
            waiting_for_xinput_release_ = false;
            refreshButtons();
        }
        return;
    }

    const int key = (event->key() == Qt::Key_Delete || event->key() == Qt::Key_Backspace) ? 0 : event->key();
    if (core_)
        core_->setKeyBinding(bindings_[static_cast<size_t>(capture_index_)].retro_button_id, key);

    capture_index_ = -1;
    capture_type_ = CaptureType::None;
    refreshButtons();
}

void InputMappingWidget::buildUi() {
    bindings_ = {
        { QStringLiteral("Up"), RETRO_DEVICE_ID_JOYPAD_UP },
        { QStringLiteral("Down"), RETRO_DEVICE_ID_JOYPAD_DOWN },
        { QStringLiteral("Left"), RETRO_DEVICE_ID_JOYPAD_LEFT },
        { QStringLiteral("Right"), RETRO_DEVICE_ID_JOYPAD_RIGHT },
        { QStringLiteral("Start"), RETRO_DEVICE_ID_JOYPAD_START },
        { QStringLiteral("Coin / Select"), RETRO_DEVICE_ID_JOYPAD_SELECT },
        { QStringLiteral("A"), RETRO_DEVICE_ID_JOYPAD_B },
        { QStringLiteral("B"), RETRO_DEVICE_ID_JOYPAD_A },
        { QStringLiteral("C"), RETRO_DEVICE_ID_JOYPAD_Y },
        { QStringLiteral("D"), RETRO_DEVICE_ID_JOYPAD_X },
        { QStringLiteral("B + C"), RETRO_DEVICE_ID_JOYPAD_L },
        { QStringLiteral("A + B"), RETRO_DEVICE_ID_JOYPAD_L2 },
        { QStringLiteral("A + B + C"), RETRO_DEVICE_ID_JOYPAD_R2 },
    };

    auto *root_layout = new QVBoxLayout(this);
    root_layout->setContentsMargins(16, 16, 16, 16);
    root_layout->setSpacing(12);

    auto *hint = new QLabel(QStringLiteral("Click a Keyboard or XInput field, then press the key or controller button. Esc cancels capture. Delete clears a binding. OK saves to config/input.ini."), this);
    hint->setWordWrap(true);
    root_layout->addWidget(hint);

    auto *grid = new QGridLayout;
    grid->setHorizontalSpacing(16);
    grid->setVerticalSpacing(8);
    root_layout->addLayout(grid);

    grid->addWidget(new QLabel(QStringLiteral("Input"), this), 0, 0);
    grid->addWidget(new QLabel(QStringLiteral("Keyboard"), this), 0, 1);
    grid->addWidget(new QLabel(QStringLiteral("XInput"), this), 0, 2);

    for (int i = 0; i < static_cast<int>(bindings_.size()); ++i) {
        auto &binding = bindings_[static_cast<size_t>(i)];
        grid->addWidget(new QLabel(binding.label, this), i + 1, 0);

        auto *keyboard_button = new QPushButton(this);
        keyboard_button->setMinimumWidth(160);
        keyboard_button->setFocusPolicy(Qt::NoFocus);
        connect(keyboard_button, &QPushButton::clicked, this, [this, i] {
            beginCapture(i, CaptureType::Keyboard);
        });
        binding.keyboard_button = keyboard_button;
        grid->addWidget(keyboard_button, i + 1, 1);

        auto *xinput_button = new QPushButton(this);
        xinput_button->setMinimumWidth(170);
        xinput_button->setFocusPolicy(Qt::NoFocus);
        connect(xinput_button, &QPushButton::clicked, this, [this, i] {
            beginCapture(i, CaptureType::XInput);
        });
        binding.xinput_button = xinput_button;
        grid->addWidget(xinput_button, i + 1, 2);
    }

    grid->setColumnStretch(2, 1);
    root_layout->addStretch();
}

void InputMappingWidget::refreshButtons() {
    for (int i = 0; i < static_cast<int>(bindings_.size()); ++i) {
        auto &binding = bindings_[static_cast<size_t>(i)];

        if (binding.keyboard_button) {
            if (capture_index_ == i && capture_type_ == CaptureType::Keyboard) {
                binding.keyboard_button->setText(QStringLiteral("Press key..."));
                binding.keyboard_button->setDefault(true);
            } else {
                binding.keyboard_button->setText(keyboardBindingText(binding.retro_button_id));
                binding.keyboard_button->setDefault(false);
            }
        }

        if (binding.xinput_button) {
            if (capture_index_ == i && capture_type_ == CaptureType::XInput) {
                binding.xinput_button->setText(waiting_for_xinput_release_ ? QStringLiteral("Release pad...") : QStringLiteral("Press pad..."));
                binding.xinput_button->setDefault(true);
            } else {
                binding.xinput_button->setText(xinputBindingText(binding.retro_button_id));
                binding.xinput_button->setDefault(false);
            }
        }
    }
}

QString InputMappingWidget::keyboardBindingText(unsigned retroButtonId) const {
    return core_ ? keyName(core_->keyBinding(retroButtonId)) : QStringLiteral("Unbound");
}

QString InputMappingWidget::xinputBindingText(unsigned retroButtonId) const {
    return core_ ? LibretroCore::xinputControlName(core_->xinputBinding(retroButtonId)) : QStringLiteral("Unbound");
}

void InputMappingWidget::beginCapture(int index, CaptureType type) {
    capture_index_ = index;
    capture_type_ = type;
    waiting_for_xinput_release_ = type == CaptureType::XInput && LibretroCore::firstPressedXInputControl() != LibretroCore::XInputNone;
    setFocus();
    refreshButtons();
}

void InputMappingWidget::pollXInputCapture() {
    if (capture_index_ < 0 || capture_type_ != CaptureType::XInput)
        return;

    const int control = LibretroCore::firstPressedXInputControl();
    if (waiting_for_xinput_release_) {
        if (control == LibretroCore::XInputNone) {
            waiting_for_xinput_release_ = false;
            refreshButtons();
        }
        return;
    }

    if (control == LibretroCore::XInputNone)
        return;

    if (core_)
        core_->setXInputBinding(bindings_[static_cast<size_t>(capture_index_)].retro_button_id, control);

    capture_index_ = -1;
    capture_type_ = CaptureType::None;
    waiting_for_xinput_release_ = false;
    refreshButtons();
}
