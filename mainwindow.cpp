#include "inputmappingwidget.h"
#include "emulatorview.h"
#include "fbneolibretrocore.h"
#include "kof98hitboxoverlay.h"
#include "libretrocore.h"
#include "mainwindow.h"
#include "memorysearchdialog.h"
#include "neocdlibretrocore.h"
#include "ui_mainwindow.h"

#include <QAction>
#include <QActionGroup>
#include <QApplication>
#include <QCheckBox>
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
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <array>
#include <functional>
#include <utility>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui_(new Ui::MainWindow)
    , emulator_view_(nullptr)
    , core_(nullptr)
    , pause_action_(nullptr)
    , reset_emulation_action_(nullptr)
    , save_state_action_(nullptr)
    , load_state_action_(nullptr)
    , pause_when_inactive_action_(nullptr)
    , show_fps_action_(nullptr)
    , show_hitboxes_action_(nullptr)
    , neocd_core_action_(nullptr)
    , fbneo_core_action_(nullptr)
    , region_group_(nullptr)
    , mode_group_(nullptr)
    , cpu_clock_group_(nullptr)
    , fps_label_(nullptr)
    , health_label_(nullptr) {
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

    health_label_ = new QLabel(emulator_view_);
    health_label_->setText(QStringLiteral("P1 HP: --  P2 HP: --"));
    health_label_->setStyleSheet(QStringLiteral(
        "QLabel {"
        "background: rgba(0, 0, 0, 150);"
        "color: white;"
        "border: 1px solid rgba(255, 255, 255, 90);"
        "padding: 3px 7px;"
        "font: 700 12px 'Consolas';"
        "}"
    ));
    health_label_->setAttribute(Qt::WA_TransparentForMouseEvents);
    health_label_->adjustSize();
    health_label_->move(fps_label_->x() + fps_label_->width() + 8, fps_label_->y());
    health_label_->raise();

    core_kind_ = savedCoreKind();
    core_ = createCore(core_kind_);
    core_->loadInputConfiguration(inputConfigPath());
    loadSystemOptionsIntoCore();
    saveSystemOptions();

    menuBar()->setNativeMenuBar(false);
    auto *file_menu = menuBar()->addMenu(QStringLiteral("File"));
    auto *load_game_action = file_menu->addAction(QStringLiteral("Load Game"));
    auto *configuration_input_action = file_menu->addAction(QStringLiteral("Configuration Input"));
    auto *core_menu = file_menu->addMenu(QStringLiteral("Core"));
    auto *core_group = new QActionGroup(this);
    core_group->setExclusive(true);

    neocd_core_action_ = core_menu->addAction(QStringLiteral("Neo Geo CD"));
    neocd_core_action_->setCheckable(true);
    neocd_core_action_->setData(static_cast<int>(CoreKind::NeoCd));
    core_group->addAction(neocd_core_action_);

    fbneo_core_action_ = core_menu->addAction(QStringLiteral("FBNeo Neo Geo Arcade"));
    fbneo_core_action_->setCheckable(true);
    fbneo_core_action_->setData(static_cast<int>(CoreKind::Fbneo));
    core_group->addAction(fbneo_core_action_);
    updateCoreActions();

    auto *region_menu = file_menu->addMenu(QStringLiteral("Region"));
    region_group_ = new QActionGroup(this);
    region_group_->setExclusive(true);
    const std::array<std::pair<const char *, const char *>, 3> region_options {{
        { "Japan", "Japan" },
        { "USA", "USA" },
        { "Europe", "Europe/Asia" },
    }};
    for (const auto &[value, label] : region_options) {
        auto *action = region_menu->addAction(QString::fromLatin1(label));
        action->setCheckable(true);
        action->setData(QString::fromLatin1(value));
        region_group_->addAction(action);
    }

    auto *mode_menu = file_menu->addMenu(QStringLiteral("Mode"));
    mode_group_ = new QActionGroup(this);
    mode_group_->setExclusive(true);
    const std::array<std::pair<const char *, const char *>, 2> mode_options {{
        { "MVS", "MVS (Arcade)" },
        { "AES", "AES (Console)" },
    }};
    for (const auto &[value, label] : mode_options) {
        auto *action = mode_menu->addAction(QString::fromLatin1(label));
        action->setCheckable(true);
        action->setData(QString::fromLatin1(value));
        mode_group_->addAction(action);
    }
    updateSystemOptionActions();

    auto *cpu_clock_menu = file_menu->addMenu(QStringLiteral("CPU Clock"));
    cpu_clock_group_ = new QActionGroup(this);
    cpu_clock_group_->setExclusive(true);
    for (int percent : {50, 100, 150, 200}) {
        const QString value = QStringLiteral("%1%").arg(percent);
        auto *action = cpu_clock_menu->addAction(value);
        action->setCheckable(true);
        action->setData(value);
        cpu_clock_group_->addAction(action);
    }
    updateSystemOptionActions();

    file_menu->addSeparator();
    pause_action_ = file_menu->addAction(QStringLiteral("Pause"));
    pause_action_->setCheckable(true);
    pause_action_->setEnabled(false);
    reset_emulation_action_ = file_menu->addAction(QStringLiteral("Reset Emulation"));
    reset_emulation_action_->setEnabled(false);
    save_state_action_ = file_menu->addAction(QStringLiteral("Save State"));
    save_state_action_->setEnabled(false);
    load_state_action_ = file_menu->addAction(QStringLiteral("Load State"));
    load_state_action_->setEnabled(false);
    file_menu->addSeparator();
    pause_when_inactive_action_ = file_menu->addAction(QStringLiteral("Pause When Focus Lost"));
    pause_when_inactive_action_->setCheckable(true);
    pause_when_inactive_action_->setChecked(true);

    auto *tools_menu = menuBar()->addMenu(QStringLiteral("Tools"));
    auto *memory_search_action = tools_menu->addAction(QStringLiteral("Memory View"));

    auto *video_menu = menuBar()->addMenu(QStringLiteral("Video"));
    show_fps_action_ = video_menu->addAction(QStringLiteral("Show FPS"));
    show_fps_action_->setCheckable(true);
    show_fps_action_->setChecked(true);
    show_hitboxes_action_ = video_menu->addAction(QStringLiteral("Show Hitboxes"));
    show_hitboxes_action_->setCheckable(true);
    show_hitboxes_action_->setChecked(true);
    emulator_view_->setHitboxOverlayEnabled(true);
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

    auto *zfast_crt_action = filter_menu->addAction(QStringLiteral("zfast-CRT"));
    zfast_crt_action->setCheckable(true);
    zfast_crt_action->setData(static_cast<int>(EmulatorView::ScalingFilter::ZfastCrt));
    filter_group->addAction(zfast_crt_action);

    auto *zfast_lcd_action = filter_menu->addAction(QStringLiteral("zfast-LCD"));
    zfast_lcd_action->setCheckable(true);
    zfast_lcd_action->setData(static_cast<int>(EmulatorView::ScalingFilter::ZfastLcd));
    filter_group->addAction(zfast_lcd_action);

    auto *scanline_fract_action = filter_menu->addAction(QStringLiteral("Scanline-fract"));
    scanline_fract_action->setCheckable(true);
    scanline_fract_action->setData(static_cast<int>(EmulatorView::ScalingFilter::ScanlineFract));
    filter_group->addAction(scanline_fract_action);

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

    connect(core_group, &QActionGroup::triggered, this, [this](QAction *action) {
        setCoreKind(static_cast<CoreKind>(action->data().toInt()));
    });
    connect(region_group_, &QActionGroup::triggered, this, [this](QAction *action) {
        setSystemRegion(action->data().toString());
    });
    connect(mode_group_, &QActionGroup::triggered, this, [this](QAction *action) {
        setSystemMode(action->data().toString());
    });
    connect(cpu_clock_group_, &QActionGroup::triggered, this, [this](QAction *action) {
        setFbneoCpuClock(action->data().toString());
    });

    connectCoreSignals();

    connect(reset_emulation_action_, &QAction::triggered, this, [this] {
        if (!core_->reset()) {
            QMessageBox::critical(this,
                                  QStringLiteral("Reset Emulation"),
                                  QStringLiteral("重新模擬失敗：\n%1").arg(core_->lastError()));
        }
    });
    connect(save_state_action_, &QAction::triggered, this, &MainWindow::saveState);
    connect(load_state_action_, &QAction::triggered, this, &MainWindow::loadState);
    connect(memory_search_action, &QAction::triggered, this, &MainWindow::showMemorySearchDialog);

    connect(show_fps_action_, &QAction::toggled, fps_label_, &QLabel::setVisible);
    connect(show_fps_action_, &QAction::toggled, health_label_, &QLabel::setVisible);
    connect(show_hitboxes_action_, &QAction::toggled, emulator_view_, &EmulatorView::setHitboxOverlayEnabled);
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

    QTimer::singleShot(0, this, &MainWindow::autoLoadStartupState);
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
        QTimer::singleShot(0, this, [this] {
            if (!pause_when_inactive_action_ || !pause_when_inactive_action_->isChecked() ||
                !core_ || !core_->isGameLoaded() || core_->isPaused() || auto_paused_for_focus_loss_) {
                return;
            }

            if (QApplication::activePopupWidget() || QApplication::activeModalWidget() || isActiveWindow())
                return;

            auto_paused_for_focus_loss_ = true;
            core_->setPaused(true);
        });
    } else if (event->type() == QEvent::WindowActivate) {
        if (auto_paused_for_focus_loss_ && core_) {
            auto_paused_for_focus_loss_ = false;
            core_->setPaused(false);
        }
    }

    return QMainWindow::event(event);
}

QStringList MainWindow::scanGameImages() const {
    if (!core_)
        return {};

    const QString root = gameRootDirectory();
    QStringList name_filters;
    for (const QString &extension : core_->supportedExtensions())
        name_filters.push_back(QStringLiteral("*.%1").arg(extension));

    QStringList result;

    QDirIterator iterator(root,
                          name_filters,
                          QDir::Files,
                          QDirIterator::Subdirectories);

    while (iterator.hasNext())
        result.push_back(QFileInfo(iterator.next()).absoluteFilePath());

    result.sort(Qt::CaseInsensitive);
    return result;
}

QString MainWindow::gameDisplayName(const QString &path) const {
    const QFileInfo file_info(path);
    const QDir root_dir(gameRootDirectory());
    const QString relative_path = root_dir.relativeFilePath(file_info.absoluteFilePath());

    if (file_info.dir().absolutePath() != root_dir.absolutePath())
        return file_info.dir().dirName().section(QLatin1Char(','), 0, 0).trimmed();

    return file_info.completeBaseName().section(QLatin1Char(','), 0, 0).trimmed();
}

QString MainWindow::gameRootDirectory() const {
    if (!core_)
        return QDir(projectRoot()).absoluteFilePath(QStringLiteral("roms"));

    return QDir(projectRoot()).absoluteFilePath(QStringLiteral("roms/%1").arg(core_->romDirectoryName()));
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
    return QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(core_->coreFileName());
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

LibretroCore *MainWindow::createCore(CoreKind kind) {
    switch (kind) {
    case CoreKind::Fbneo:
        return new FbneoLibretroCore(emulator_view_, this);
    case CoreKind::NeoCd:
    default:
        return new NeoCdLibretroCore(emulator_view_, this);
    }
}

void MainWindow::setCoreKind(CoreKind kind) {
    if (core_ && core_kind_ == kind) {
        updateCoreActions();
        return;
    }

    if (core_) {
        core_->stop();
        delete core_;
        core_ = nullptr;
    }

    core_kind_ = kind;
    core_ = createCore(core_kind_);
    core_->loadInputConfiguration(inputConfigPath());
    loadSystemOptionsIntoCore();
    saveSystemOptions();
    connectCoreSignals();
    saveCoreKind(core_kind_);

    current_game_path_.clear();
    auto_paused_for_focus_loss_ = false;
    pause_action_->setEnabled(false);
    reset_emulation_action_->setEnabled(false);
    save_state_action_->setEnabled(false);
    load_state_action_->setEnabled(false);
    pause_action_->setChecked(false);
    updateCoreActions();
    updateSystemOptionActions();
    if (memory_search_dialog_)
        memory_search_dialog_->setCore(core_);
}

void MainWindow::connectCoreSignals() {
    if (!core_ || !pause_action_)
        return;

    connect(pause_action_, &QAction::toggled, core_, &LibretroCore::setPaused);
    connect(core_, &LibretroCore::pausedChanged, this, [this](bool paused) {
        if (pause_action_->isChecked() != paused)
            pause_action_->setChecked(paused);
    });
    connect(core_, &LibretroCore::frameAdvanced, this, &MainWindow::updateKof98Overlay);
}

void MainWindow::updateCoreActions() {
    if (neocd_core_action_)
        neocd_core_action_->setChecked(core_kind_ == CoreKind::NeoCd);
    if (fbneo_core_action_)
        fbneo_core_action_->setChecked(core_kind_ == CoreKind::Fbneo);
    if (core_)
        setWindowTitle(QStringLiteral("qneogeo - %1").arg(core_->displayName()));
}

void MainWindow::loadSystemOptionsIntoCore() const {
    if (!core_)
        return;

    QSettings settings(inputConfigPath(), QSettings::IniFormat);
    core_->setSystemRegionOption(settings.value(QStringLiteral("System/Region"),
                                                QStringLiteral("Japan")).toString());
    QString mode = settings.value(QStringLiteral("System/Mode"),
                                  QStringLiteral("MVS")).toString();
    if (mode != QStringLiteral("MVS") && mode != QStringLiteral("AES"))
        mode = QStringLiteral("MVS");
    core_->setSystemModeOption(mode);

    QString cpu_clock = settings.value(QStringLiteral("FBNeo/CpuClock"),
                                       QStringLiteral("100%")).toString();
    static const QRegularExpression valid_cpu_clock(QStringLiteral(R"(^(50|100|150|200)%$)"));
    if (!valid_cpu_clock.match(cpu_clock).hasMatch())
        cpu_clock = QStringLiteral("100%");
    core_->setFbneoCpuClockOption(cpu_clock);
}

void MainWindow::setSystemRegion(const QString &region) {
    if (!core_ || region.isEmpty())
        return;

    if (core_->systemRegionOption() == region) {
        updateSystemOptionActions();
        return;
    }

    core_->setSystemRegionOption(region);
    saveSystemOptions();
    updateSystemOptionActions();
    reloadCurrentGame();
}

void MainWindow::setSystemMode(const QString &mode) {
    if (!core_ || mode.isEmpty())
        return;

    if (core_->systemModeOption() == mode) {
        updateSystemOptionActions();
        return;
    }

    core_->setSystemModeOption(mode);
    saveSystemOptions();
    updateSystemOptionActions();
    reloadCurrentGame();
}

void MainWindow::setFbneoCpuClock(const QString &cpuClock) {
    if (!core_ || cpuClock.isEmpty())
        return;

    if (core_->fbneoCpuClockOption() == cpuClock) {
        updateSystemOptionActions();
        return;
    }

    core_->setFbneoCpuClockOption(cpuClock);
    saveSystemOptions();
    updateSystemOptionActions();
    reloadCurrentGame();
}

void MainWindow::saveSystemOptions() const {
    if (!core_)
        return;

    QFileInfo file_info(inputConfigPath());
    QDir().mkpath(file_info.absolutePath());

    QSettings settings(inputConfigPath(), QSettings::IniFormat);
    settings.setValue(QStringLiteral("System/Region"), core_->systemRegionOption());
    settings.setValue(QStringLiteral("System/Mode"), core_->systemModeOption());
    settings.setValue(QStringLiteral("FBNeo/CpuClock"), core_->fbneoCpuClockOption());
    settings.sync();
}

void MainWindow::updateSystemOptionActions() {
    if (!core_)
        return;

    if (region_group_) {
        for (QAction *action : region_group_->actions())
            action->setChecked(action->data().toString() == core_->systemRegionOption());
    }

    if (mode_group_) {
        for (QAction *action : mode_group_->actions()) {
            action->setChecked(action->data().toString() == core_->systemModeOption());
            action->setEnabled(core_kind_ == CoreKind::Fbneo);
        }
    }

    if (cpu_clock_group_) {
        for (QAction *action : cpu_clock_group_->actions()) {
            action->setChecked(action->data().toString() == core_->fbneoCpuClockOption());
            action->setEnabled(core_kind_ == CoreKind::Fbneo);
        }
    }
}

void MainWindow::reloadCurrentGame() {
    if (!core_ || current_game_path_.isEmpty() || !core_->isGameLoaded())
        return;

    const QString game_path = current_game_path_;
    loadGame(game_path);
}

MainWindow::CoreKind MainWindow::savedCoreKind() const {
    QSettings settings(inputConfigPath(), QSettings::IniFormat);
    const QString saved_core = settings.value(QStringLiteral("Core/Selected"),
                                              QStringLiteral("neocd")).toString();
    if (saved_core == QStringLiteral("fbneo"))
        return CoreKind::Fbneo;

    return CoreKind::NeoCd;
}

void MainWindow::saveCoreKind(CoreKind kind) const {
    QFileInfo file_info(inputConfigPath());
    QDir().mkpath(file_info.absolutePath());

    QSettings settings(inputConfigPath(), QSettings::IniFormat);
    settings.setValue(QStringLiteral("Core/Selected"),
                      kind == CoreKind::Fbneo ? QStringLiteral("fbneo") : QStringLiteral("neocd"));
    settings.sync();
}

void MainWindow::autoLoadStartupState() {
    if (core_kind_ != CoreKind::Fbneo)
        setCoreKind(CoreKind::Fbneo);

    QString game_path = QDir(gameRootDirectory()).absoluteFilePath(QStringLiteral("kof98.zip"));
    if (!QFileInfo::exists(game_path)) {
        const QStringList game_images = scanGameImages();
        if (game_images.isEmpty()) {
            qWarning().noquote() << "Startup auto-load skipped: no FBNeo game found in"
                                 << QDir::toNativeSeparators(gameRootDirectory());
            return;
        }

        game_path = game_images.first();
    }

    loadGame(game_path);
    if (!core_ || !core_->isGameLoaded())
        return;

    const QString state_path = stateFilePath();
    if (!QFileInfo::exists(state_path)) {
        qWarning().noquote() << "Startup auto-load state skipped: state file not found"
                             << QDir::toNativeSeparators(state_path);
        return;
    }

    if (!core_->loadState(state_path)) {
        qWarning().noquote() << "Startup auto-load state failed:" << core_->lastError();
        return;
    }

    qInfo().noquote() << "Startup auto-loaded"
                      << QDir::toNativeSeparators(game_path)
                      << "and state"
                      << QDir::toNativeSeparators(state_path);
}

void MainWindow::loadGame(const QString &path) {
    if (!core_->loadCore(corePath()) || !core_->startGame(path, systemDirectory(), saveDirectory())) {
        QMessageBox::critical(this,
                              QStringLiteral("qneogeo"),
                              QStringLiteral("%1 啟動失敗：\n%2").arg(core_->displayName(), core_->lastError()));
        current_game_path_.clear();
        pause_action_->setEnabled(false);
        reset_emulation_action_->setEnabled(false);
        save_state_action_->setEnabled(false);
        load_state_action_->setEnabled(false);
        pause_action_->setChecked(false);
        auto_paused_for_focus_loss_ = false;
        return;
    }

    current_game_path_ = path;
    pause_action_->setEnabled(true);
    reset_emulation_action_->setEnabled(true);
    save_state_action_->setEnabled(true);
    load_state_action_->setEnabled(true);
    pause_action_->setChecked(false);
    auto_paused_for_focus_loss_ = false;
}

void MainWindow::showLoadGameDialog() {
    const QStringList game_images = scanGameImages();
    if (game_images.isEmpty()) {
        QStringList patterns;
        for (const QString &extension : core_->supportedExtensions())
            patterns.push_back(QStringLiteral("*.%1").arg(extension));

        QMessageBox::information(this,
                                 QStringLiteral("Load Game"),
                                 QStringLiteral("%1 裡找不到 %2。")
                                     .arg(QDir::toNativeSeparators(gameRootDirectory()), patterns.join(QStringLiteral(" 或 "))));
        return;
    }

    QDialog dialog(this);
    dialog.setWindowTitle(QStringLiteral("Load Game"));
    dialog.resize(720, 420);

    auto *layout = new QVBoxLayout(&dialog);
    layout->setContentsMargins(12, 12, 12, 12);

    auto *label = new QLabel(QStringLiteral("Select a %1 game:").arg(core_->displayName()), &dialog);
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
    const bool original_arcade_socd_clean = core_->arcadeSocdClean();
    const bool original_keyboard_motion_assist = core_->keyboardMotionAssist();

    QDialog dialog(this);
    dialog.setWindowTitle(QStringLiteral("Configuration Input"));
    dialog.setModal(true);
    dialog.resize(420, 580);

    auto *layout = new QVBoxLayout(&dialog);
    layout->setContentsMargins(12, 12, 12, 12);
    auto *widget = new InputMappingWidget(core_, &dialog);
    layout->addWidget(widget);

    auto *arcade_socd_clean_checkbox = new QCheckBox(QStringLiteral("Arcade SOCD Clean"), &dialog);
    arcade_socd_clean_checkbox->setChecked(core_->arcadeSocdClean());
    layout->addWidget(arcade_socd_clean_checkbox);

    auto *keyboard_motion_assist_checkbox = new QCheckBox(QStringLiteral("Motion Assist"), &dialog);
    keyboard_motion_assist_checkbox->setChecked(core_->keyboardMotionAssist());
    layout->addWidget(keyboard_motion_assist_checkbox);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(buttons);

    if (dialog.exec() == QDialog::Accepted) {
        core_->setArcadeSocdClean(arcade_socd_clean_checkbox->isChecked());
        core_->setKeyboardMotionAssist(keyboard_motion_assist_checkbox->isChecked());
        core_->saveInputConfiguration(inputConfigPath());
    } else {
        core_->setKeyBindings(original_key_bindings);
        core_->setXInputBindings(original_xinput_bindings);
        core_->setArcadeSocdClean(original_arcade_socd_clean);
        core_->setKeyboardMotionAssist(original_keyboard_motion_assist);
    }
}

void MainWindow::showMemorySearchDialog() {
    if (memory_search_dialog_) {
        memory_search_dialog_->setCore(core_);
        memory_search_dialog_->show();
        memory_search_dialog_->raise();
        memory_search_dialog_->activateWindow();
        return;
    }

    memory_search_dialog_ = new MemorySearchDialog(core_, this);
    connect(memory_search_dialog_, &QObject::destroyed, this, [this] {
        memory_search_dialog_ = nullptr;
    });
    memory_search_dialog_->show();
    memory_search_dialog_->raise();
    memory_search_dialog_->activateWindow();
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
    if (health_label_)
        health_label_->move(fps_label_->x() + fps_label_->width() + 8, fps_label_->y());
}

void MainWindow::updateKof98Overlay() {
    if (!emulator_view_ || !core_ || !core_->isGameLoaded()) {
        if (emulator_view_)
            emulator_view_->setHitboxOverlay({}, {});
        return;
    }

    QByteArray ram;
    if (!core_->readSystemRam(ram)) {
        emulator_view_->setHitboxOverlay({}, {});
        return;
    }

    KofHitboxOverlayBuilder builder(std::move(ram), emulator_view_->sourceSize());
    if (emulator_view_->hitboxOverlayEnabled()) {
        KofHitboxOverlayBuilder::Result overlay = builder.build();
        emulator_view_->setHitboxOverlay(std::move(overlay.boxes), std::move(overlay.axes));
    } else {
        emulator_view_->setHitboxOverlay({}, {});
    }

    if (!health_label_)
        return;

    auto healthText = [](int health) {
        if (health < 0)
            return QStringLiteral("--");

        const int clamped = qBound(0, health, KofHitboxOverlayBuilder::MaxHealth);
        return QStringLiteral("%1/%2").arg(clamped).arg(KofHitboxOverlayBuilder::MaxHealth);
    };

    health_label_->setText(QStringLiteral("P1 HP: %1  P2 HP: %2")
                               .arg(healthText(builder.readP1Health()),
                                    healthText(builder.readP2Health())));
    health_label_->adjustSize();
    if (fps_label_)
        health_label_->move(fps_label_->x() + fps_label_->width() + 8, fps_label_->y());
}
