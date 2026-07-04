#include "inputmappingwidget.h"
#include "emulatorview.h"
#include "libretrocore.h"
#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QAction>
#include <QActionGroup>
#include <QPushButton>
#include <QCoreApplication>
#include <QDir>
#include <QDialog>
#include <QDialogButtonBox>
#include <QEvent>
#include <QFileInfo>
#include <QDirIterator>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QMenuBar>
#include <QMessageBox>
#include <QRegularExpression>
#include <QSettings>
#include <QSlider>
#include <QVBoxLayout>

#include <functional>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui_(new Ui::MainWindow)
    , emulator_view_(nullptr)
    , core_(nullptr)
    , pause_action_(nullptr)
    , save_state_action_(nullptr)
    , load_state_action_(nullptr)
    , pause_when_inactive_action_(nullptr)
    , show_fps_action_(nullptr)
    , fps_label_(nullptr) {
    ui_->setupUi(this);
    setWindowTitle(QStringLiteral("qneogeo"));

    emulator_view_ = new EmulatorView(this);
    setCentralWidget(emulator_view_);

    fps_label_ = new QLabel(emulator_view_);
    fps_label_->setText(QStringLiteral("FPS: --"));
    fps_label_->setStyleSheet(QStringLiteral(
        "QLabel {"
        "background: rgba(0, 0, 0, 150);"
        "color: white;"
        "border: 1px solid rgba(255, 255, 255, 90);"
        "padding: 3px 7px;"
        "font: 700 12px 'Consolas';"
        "}"
    ));
    fps_label_->setAttribute(Qt::WA_TransparentForMouseEvents);
    fps_label_->adjustSize();
    fps_label_->move(10, 10);
    fps_label_->raise();

    core_ = new LibretroCore(emulator_view_, this);
    core_->loadInputConfiguration(inputConfigPath());

    menuBar()->setNativeMenuBar(false);
    auto *main_menu = menuBar()->addMenu(QStringLiteral("Menu"));
    auto *load_game_action = main_menu->addAction(QStringLiteral("Load Game"));
    auto *configuration_input_action = main_menu->addAction(QStringLiteral("Configuration Input"));

    auto *input_menu = menuBar()->addMenu(QStringLiteral("Input"));
    arcade_socd_clean_action_ = input_menu->addAction(QStringLiteral("Arcade SOCD Clean"));
    arcade_socd_clean_action_->setCheckable(true);
    arcade_socd_clean_action_->setChecked(core_->arcadeSocdClean());
    keyboard_motion_assist_action_ = input_menu->addAction(QStringLiteral("Keyboard Motion Assist"));
    keyboard_motion_assist_action_->setCheckable(true);
    keyboard_motion_assist_action_->setChecked(core_->keyboardMotionAssist());

    auto *state_menu = menuBar()->addMenu(QStringLiteral("State"));
    pause_action_ = state_menu->addAction(QStringLiteral("Pause"));
    pause_action_->setCheckable(true);
    pause_action_->setEnabled(false);
    save_state_action_ = state_menu->addAction(QStringLiteral("Save State"));
    save_state_action_->setEnabled(false);
    load_state_action_ = state_menu->addAction(QStringLiteral("Load State"));
    load_state_action_->setEnabled(false);
    state_menu->addSeparator();
    pause_when_inactive_action_ = state_menu->addAction(QStringLiteral("Pause When Focus Lost"));
    pause_when_inactive_action_->setCheckable(true);
    pause_when_inactive_action_->setChecked(true);

    auto *video_menu = menuBar()->addMenu(QStringLiteral("Video"));
    show_fps_action_ = video_menu->addAction(QStringLiteral("Show FPS"));
    show_fps_action_->setCheckable(true);
    show_fps_action_->setChecked(true);
    video_menu->addSeparator();
    auto *filter_menu = video_menu->addMenu(QStringLiteral("Filter"));
    auto *filter_group = new QActionGroup(this);
    filter_group->setExclusive(true);

    auto *nearest_action = filter_menu->addAction(QStringLiteral("Nearest"));
    nearest_action->setCheckable(true);
    nearest_action->setChecked(true);
    nearest_action->setData(static_cast<int>(EmulatorView::ScalingFilter::Nearest));
    filter_group->addAction(nearest_action);

    auto *linear_action = filter_menu->addAction(QStringLiteral("Linear"));
    linear_action->setCheckable(true);
    linear_action->setData(static_cast<int>(EmulatorView::ScalingFilter::Linear));
    filter_group->addAction(linear_action);

    auto *super2xsai_action = filter_menu->addAction(QStringLiteral("Super2xSaI"));
    super2xsai_action->setCheckable(true);
    super2xsai_action->setData(static_cast<int>(EmulatorView::ScalingFilter::Super2xSai));
    filter_group->addAction(super2xsai_action);

    auto *xbrz_action = filter_menu->addAction(QStringLiteral("xBRZ Freescale"));
    xbrz_action->setCheckable(true);
    xbrz_action->setData(static_cast<int>(EmulatorView::ScalingFilter::XbrzFreescale));
    filter_group->addAction(xbrz_action);

    auto *libretro_xbrz_action = filter_menu->addAction(QStringLiteral("libretro xBRZ Freescale"));
    libretro_xbrz_action->setCheckable(true);
    libretro_xbrz_action->setData(static_cast<int>(EmulatorView::ScalingFilter::LibretroXbrzFreescale));
    filter_group->addAction(libretro_xbrz_action);

    auto *libretro_6xbrz_action = filter_menu->addAction(QStringLiteral("libretro 6xBRZ"));
    libretro_6xbrz_action->setCheckable(true);
    libretro_6xbrz_action->setData(static_cast<int>(EmulatorView::ScalingFilter::Libretro6xbrz));
    filter_group->addAction(libretro_6xbrz_action);

    {
        QSettings settings(inputConfigPath(), QSettings::IniFormat);
        const int saved_filter = settings.value(QStringLiteral("Video/ScalingFilter"),
                                                static_cast<int>(EmulatorView::ScalingFilter::Nearest)).toInt();
        for (QAction *action : filter_group->actions()) {
            if (action->data().toInt() != saved_filter)
                continue;

            action->setChecked(true);
            emulator_view_->setScalingFilter(static_cast<EmulatorView::ScalingFilter>(saved_filter));
            break;
        }
    }

    connect(load_game_action, &QAction::triggered, this, &MainWindow::showLoadGameDialog);

    connect(configuration_input_action, &QAction::triggered, this, &MainWindow::showInputConfiguration);

    connect(arcade_socd_clean_action_, &QAction::toggled, this, [this](bool enabled) {
        core_->setArcadeSocdClean(enabled);
        core_->saveInputConfiguration(inputConfigPath());
    });

    connect(keyboard_motion_assist_action_, &QAction::toggled, this, [this](bool enabled) {
        core_->setKeyboardMotionAssist(enabled);
        core_->saveInputConfiguration(inputConfigPath());
    });

    connect(pause_action_, &QAction::toggled, core_, &LibretroCore::setPaused);
    connect(save_state_action_, &QAction::triggered, this, &MainWindow::saveState);
    connect(load_state_action_, &QAction::triggered, this, &MainWindow::loadState);
    connect(core_, &LibretroCore::pausedChanged, this, [this](bool paused) {
        if (pause_action_->isChecked() != paused)
            pause_action_->setChecked(paused);
    });

    connect(show_fps_action_, &QAction::toggled, fps_label_, &QLabel::setVisible);
    connect(emulator_view_, &EmulatorView::fpsChanged, this, &MainWindow::updateFpsOverlay);

    connect(filter_group, &QActionGroup::triggered, this, [this](QAction *action) {
        const auto filter = static_cast<EmulatorView::ScalingFilter>(action->data().toInt());
        emulator_view_->setScalingFilter(filter);

        QFileInfo file_info(inputConfigPath());
        QDir().mkpath(file_info.absolutePath());
        QSettings settings(inputConfigPath(), QSettings::IniFormat);
        settings.setValue(QStringLiteral("Video/ScalingFilter"), action->data().toInt());
        settings.sync();

        if (filter == EmulatorView::ScalingFilter::Super2xSai)
            showSuper2xSaiSettingsDialog();
    });

    const QStringList game_images = scanGameImages();

    if (game_images.isEmpty()) {
        QMessageBox::warning(this,
                             QStringLiteral("qneogeo"),
                             QStringLiteral("找不到 Neo Geo CD 遊戲。請確認 .cue 或 .chd 已放在 roms/neocd。"));
        return;
    }

    //loadGame(game_images.first());
}

MainWindow::~MainWindow() {
    if (core_) {
        core_->stop();
        delete core_;
        core_ = nullptr;
    }

    delete ui_;
}

bool MainWindow::event(QEvent *event) {
    if (event->type() == QEvent::WindowDeactivate) {
        if (pause_when_inactive_action_ && pause_when_inactive_action_->isChecked() &&
            core_ && core_->isGameLoaded() && !core_->isPaused()) {
            auto_paused_for_focus_loss_ = true;
            core_->setPaused(true);
        }
    } else if (event->type() == QEvent::WindowActivate) {
        if (auto_paused_for_focus_loss_ && core_) {
            auto_paused_for_focus_loss_ = false;
            core_->setPaused(false);
        }
    }

    return QMainWindow::event(event);
}

QStringList MainWindow::scanGameImages() const {
    const QString root = QDir(projectRoot()).absoluteFilePath(QStringLiteral("roms/neocd"));
    QStringList result;

    QDirIterator iterator(root,
                          QStringList { QStringLiteral("*.cue"), QStringLiteral("*.chd") },
                          QDir::Files,
                          QDirIterator::Subdirectories);

    while (iterator.hasNext())
        result.push_back(QFileInfo(iterator.next()).absoluteFilePath());

    result.sort(Qt::CaseInsensitive);
    return result;
}

QString MainWindow::gameDisplayName(const QString &path) const {
    const QFileInfo file_info(path);
    const QDir root_dir(QDir(projectRoot()).absoluteFilePath(QStringLiteral("roms/neocd")));
    const QString relative_path = root_dir.relativeFilePath(file_info.absoluteFilePath());

    if (file_info.dir().absolutePath() != root_dir.absolutePath())
        return file_info.dir().dirName().section(QLatin1Char(','), 0, 0).trimmed();

    return file_info.completeBaseName().section(QLatin1Char(','), 0, 0).trimmed();
}

QString MainWindow::stateFilePath() const {
    QString name = gameDisplayName(current_game_path_);
    if (name.isEmpty())
        name = QFileInfo(current_game_path_).completeBaseName();

    static const QRegularExpression invalid_characters(QStringLiteral(R"([<>:"/\\|?*\x00-\x1f])"));
    name.replace(invalid_characters, QStringLiteral("_"));

    const QString states_dir = QDir(saveDirectory()).absoluteFilePath(QStringLiteral("states"));
    return QDir(states_dir).absoluteFilePath(QStringLiteral("%1.slot1.state").arg(name));
}

QString MainWindow::projectRoot() const {
    const QDir appDir(QCoreApplication::applicationDirPath());
    return QDir(appDir.absoluteFilePath(QStringLiteral("../.."))).absolutePath();
}

QString MainWindow::corePath() const {
    return QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(QStringLiteral("neocd_libretro.dll"));
}

QString MainWindow::systemDirectory() const {
    return QDir(projectRoot()).absoluteFilePath(QStringLiteral("system"));
}

QString MainWindow::saveDirectory() const {
    return QDir(projectRoot()).absoluteFilePath(QStringLiteral("saves"));
}

QString MainWindow::inputConfigPath() const {
    return QDir(projectRoot()).absoluteFilePath(QStringLiteral("config/input.ini"));
}

void MainWindow::loadGame(const QString &path) {
    if (!core_->loadCore(corePath()) || !core_->startGame(path, systemDirectory(), saveDirectory())) {
        QMessageBox::critical(this,
                              QStringLiteral("qneogeo"),
                              QStringLiteral("NeoCD 啟動失敗：\n%1").arg(core_->lastError()));
        current_game_path_.clear();
        pause_action_->setEnabled(false);
        save_state_action_->setEnabled(false);
        load_state_action_->setEnabled(false);
        pause_action_->setChecked(false);
        auto_paused_for_focus_loss_ = false;
        return;
    }

    current_game_path_ = path;
    pause_action_->setEnabled(true);
    save_state_action_->setEnabled(true);
    load_state_action_->setEnabled(true);
    pause_action_->setChecked(false);
    auto_paused_for_focus_loss_ = false;
}

void MainWindow::showLoadGameDialog() {
    const QStringList game_images = scanGameImages();
    if (game_images.isEmpty()) {
        QMessageBox::information(this,
                                 QStringLiteral("Load Game"),
                                 QStringLiteral("roms/neocd 裡找不到 .cue 或 .chd。"));
        return;
    }

    QDialog dialog(this);
    dialog.setWindowTitle(QStringLiteral("Load Game"));
    dialog.resize(720, 420);

    auto *layout = new QVBoxLayout(&dialog);
    layout->setContentsMargins(12, 12, 12, 12);

    auto *label = new QLabel(QStringLiteral("Select a Neo Geo CD image:"), &dialog);
    layout->addWidget(label);

    auto *list = new QListWidget(&dialog);
    for (const QString &path : game_images) {
        auto *item = new QListWidgetItem(gameDisplayName(path), list);
        item->setData(Qt::UserRole, path);
        item->setToolTip(path);
    }
    list->setCurrentRow(0);
    layout->addWidget(list);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    connect(list, &QListWidget::itemDoubleClicked, &dialog, &QDialog::accept);
    layout->addWidget(buttons);

    if (dialog.exec() != QDialog::Accepted || !list->currentItem())
        return;

    loadGame(list->currentItem()->data(Qt::UserRole).toString());
}

void MainWindow::showInputConfiguration() {
    const auto original_key_bindings = core_->keyBindings();
    const auto original_xinput_bindings = core_->xinputBindings();

    QDialog dialog(this);
    dialog.setWindowTitle(QStringLiteral("Configuration Input"));
    dialog.setModal(true);
    dialog.resize(420, 520);

    auto *layout = new QVBoxLayout(&dialog);
    layout->setContentsMargins(12, 12, 12, 12);
    auto *widget = new InputMappingWidget(core_, &dialog);
    layout->addWidget(widget);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(buttons);

    if (dialog.exec() == QDialog::Accepted) {
        core_->saveInputConfiguration(inputConfigPath());
    } else {
        core_->setKeyBindings(original_key_bindings);
        core_->setXInputBindings(original_xinput_bindings);
    }
}

void MainWindow::saveState() {
    if (current_game_path_.isEmpty() || !core_->isGameLoaded()) {
        QMessageBox::information(this, QStringLiteral("Save State"), QStringLiteral("目前沒有載入遊戲。"));
        return;
    }

    const QString path = stateFilePath();
    if (!core_->saveState(path)) {
        QMessageBox::critical(this,
                              QStringLiteral("Save State"),
                              QStringLiteral("儲存狀態失敗：\n%1").arg(core_->lastError()));
        return;
    }

    QMessageBox::information(this,
                             QStringLiteral("Save State"),
                             QStringLiteral("已儲存：\n%1").arg(QDir::toNativeSeparators(path)));
}

void MainWindow::loadState() {
    if (current_game_path_.isEmpty() || !core_->isGameLoaded()) {
        QMessageBox::information(this, QStringLiteral("Load State"), QStringLiteral("目前沒有載入遊戲。"));
        return;
    }

    const QString path = stateFilePath();
    if (!QFileInfo::exists(path)) {
        QMessageBox::information(this,
                                 QStringLiteral("Load State"),
                                 QStringLiteral("找不到狀態檔：\n%1").arg(QDir::toNativeSeparators(path)));
        return;
    }

    if (!core_->loadState(path)) {
        QMessageBox::critical(this,
                              QStringLiteral("Load State"),
                              QStringLiteral("讀取狀態失敗：\n%1").arg(core_->lastError()));
    }
}

void MainWindow::showSuper2xSaiSettingsDialog() {
    if (super2xsai_dialog_) {
        super2xsai_dialog_->show();
        super2xsai_dialog_->raise();
        super2xsai_dialog_->activateWindow();
        return;
    }

    super2xsai_dialog_ = new QDialog(this);
    super2xsai_dialog_->setWindowTitle(QStringLiteral("Super2xSaI Settings"));
    super2xsai_dialog_->setAttribute(Qt::WA_DeleteOnClose);
    super2xsai_dialog_->setModal(false);
    super2xsai_dialog_->resize(360, 170);

    connect(super2xsai_dialog_, &QObject::destroyed, this, [this] {
        super2xsai_dialog_ = nullptr;
    });

    auto *layout = new QVBoxLayout(super2xsai_dialog_);
    layout->setContentsMargins(12, 12, 12, 12);
    layout->setSpacing(10);

    auto updateParameters = [this](float sharpAmount, float edgeBlend, float nearestHold) {
        emulator_view_->setSuper2xSaiParameters(sharpAmount, edgeBlend, nearestHold);
    };

    auto addSlider = [this, layout, updateParameters](const QString &name,
                                                      float initialValue,
                                                      const std::function<void(float)> &setValue) {
        auto *row = new QHBoxLayout;
        auto *name_label = new QLabel(name, super2xsai_dialog_);
        auto *slider = new QSlider(Qt::Horizontal, super2xsai_dialog_);
        auto *value_label = new QLabel(super2xsai_dialog_);

        name_label->setMinimumWidth(90);
        value_label->setMinimumWidth(42);
        value_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

        slider->setRange(0, 100);
        slider->setSingleStep(1);
        slider->setPageStep(5);
        slider->setValue(qRound(initialValue * 100.0f));
        value_label->setText(QString::number(initialValue, 'f', 2));

        connect(slider, &QSlider::valueChanged, this, [value_label, setValue](int value) {
            const float normalized = static_cast<float>(value) / 100.0f;
            value_label->setText(QString::number(normalized, 'f', 2));
            setValue(normalized);
        });

        row->addWidget(name_label);
        row->addWidget(slider, 1);
        row->addWidget(value_label);
        layout->addLayout(row);

        return slider;
    };

    auto *sharp_amount_slider = addSlider(QStringLiteral("Sharp Amount"),
                                          emulator_view_->super2xSaiSharpAmount(),
                                          [this, updateParameters](float value) {
                                              updateParameters(value,
                                                               emulator_view_->super2xSaiEdgeBlend(),
                                                               emulator_view_->super2xSaiNearestHold());
                                          });

    auto *edge_blend_slider = addSlider(QStringLiteral("Edge Blend"),
                                        emulator_view_->super2xSaiEdgeBlend(),
                                        [this, updateParameters](float value) {
                                            updateParameters(emulator_view_->super2xSaiSharpAmount(),
                                                             value,
                                                             emulator_view_->super2xSaiNearestHold());
                                        });

    auto *nearest_hold_slider = addSlider(QStringLiteral("Nearest Hold"),
                                          emulator_view_->super2xSaiNearestHold(),
                                          [this, updateParameters](float value) {
                                              updateParameters(emulator_view_->super2xSaiSharpAmount(),
                                                               emulator_view_->super2xSaiEdgeBlend(),
                                                               value);
                                          });

    auto *button_row = new QHBoxLayout;
    button_row->addStretch(1);
    auto *reset_button = new QPushButton(QStringLiteral("Reset"), super2xsai_dialog_);
    button_row->addWidget(reset_button);
    layout->addLayout(button_row);

    connect(reset_button, &QPushButton::clicked, this, [sharp_amount_slider, edge_blend_slider, nearest_hold_slider] {
        sharp_amount_slider->setValue(25);
        edge_blend_slider->setValue(45);
        nearest_hold_slider->setValue(10);
    });

    super2xsai_dialog_->show();
    super2xsai_dialog_->raise();
    super2xsai_dialog_->activateWindow();
}

void MainWindow::updateFpsOverlay(double fps) {
    if (!fps_label_)
        return;

    if (fps <= 0.0) {
        fps_label_->setText(QStringLiteral("FPS: --"));
    } else {
        fps_label_->setText(QStringLiteral("FPS: %1").arg(fps, 0, 'f', 1));
    }

    fps_label_->adjustSize();
}
