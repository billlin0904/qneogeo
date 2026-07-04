#pragma once

#include <QWidget>

#include <vector>

class LibretroCore;
class QPushButton;
class QTimer;

class InputMappingWidget final : public QWidget {
    Q_OBJECT

public:
    explicit InputMappingWidget(LibretroCore *core, QWidget *parent = nullptr);

protected:
    void keyPressEvent(QKeyEvent *event) override;

private:
    enum class CaptureType {
        None,
        Keyboard,
        XInput
    };

    struct Binding {
        QString label;
        unsigned retro_button_id = 0;
        QPushButton *keyboard_button = nullptr;
        QPushButton *xinput_button = nullptr;
    };

    void buildUi();
    void refreshButtons();
    QString keyboardBindingText(unsigned retroButtonId) const;
    QString xinputBindingText(unsigned retroButtonId) const;
    void beginCapture(int index, CaptureType type);
    void pollXInputCapture();

    LibretroCore *core_ = nullptr;
    std::vector<Binding> bindings_;
    int capture_index_ = -1;
    CaptureType capture_type_ = CaptureType::None;
    bool waiting_for_xinput_release_ = false;
    QTimer *xinput_timer_ = nullptr;
};
