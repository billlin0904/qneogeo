#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QVector>

#include <cstdint>

class QAction;
class QActionGroup;
class EmulatorView;
class QEvent;
class QLabel;
class LibretroCore;
class MemorySearchDialog;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    enum class CoreKind {
        NeoCd,
        Fbneo
    };

    bool event(QEvent *event) override;

    QStringList scanGameImages() const;
    QString gameDisplayName(const QString &path) const;
    QString gameRootDirectory() const;
    QString stateFilePath(int32_t slot = 1) const;
    QString projectRoot() const;
    QString corePath() const;
    QString systemDirectory() const;
    QString saveDirectory() const;
    QString inputConfigPath() const;
    LibretroCore *createCore(CoreKind kind);
    void setCoreKind(CoreKind kind);
    void connectCoreSignals();
    void updateCoreActions();
    void loadSystemOptionsIntoCore() const;
    void setSystemRegion(const QString &region);
    void setSystemMode(const QString &mode);
    void setFbneoCpuClock(const QString &cpuClock);
    void saveSystemOptions() const;
    void updateSystemOptionActions();
    void reloadCurrentGame();
    CoreKind savedCoreKind() const;
    void saveCoreKind(CoreKind kind) const;
    void autoLoadStartupState();
    void loadGame(const QString &path);
    void showLoadGameDialog();
    void showInputConfiguration();
    void showMemorySearchDialog();
    void saveState(int32_t slot = 1);
    void loadState(int32_t slot = 1);
    void updateFpsOverlay(double fps);
    void updateKof98Overlay();

    Ui::MainWindow *ui_;
    EmulatorView *emulator_view_;
    LibretroCore *core_;
    QAction *pause_action_;
    QAction *reset_emulation_action_;
    QVector<QAction *> save_state_actions_;
    QVector<QAction *> load_state_actions_;
    QAction *pause_when_inactive_action_;
    QAction *show_fps_action_;
    QAction *show_hitboxes_action_;
    QAction *neocd_core_action_;
    QAction *fbneo_core_action_;
    QActionGroup *region_group_;
    QActionGroup *mode_group_;
    QActionGroup *cpu_clock_group_;
    QLabel *fps_label_;
    QLabel *health_label_;
    MemorySearchDialog *memory_search_dialog_ = nullptr;
    QString current_game_path_;
    CoreKind core_kind_ = CoreKind::NeoCd;
    bool auto_paused_for_focus_loss_ = false;
};
#endif // MAINWINDOW_H
