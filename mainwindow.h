#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

class QAction;
class QDialog;
class EmulatorView;
class QEvent;
class QLabel;
class LibretroCore;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    bool event(QEvent *event) override;

    QStringList scanGameImages() const;
    QString gameDisplayName(const QString &path) const;
    QString stateFilePath() const;
    QString projectRoot() const;
    QString corePath() const;
    QString systemDirectory() const;
    QString saveDirectory() const;
    QString inputConfigPath() const;
    void loadGame(const QString &path);
    void showLoadGameDialog();
    void showInputConfiguration();
    void saveState();
    void loadState();
    void showSuper2xSaiSettingsDialog();
    void updateFpsOverlay(double fps);

    Ui::MainWindow *ui_;
    EmulatorView *emulator_view_;
    LibretroCore *core_;
    QAction *pause_action_;
    QAction *save_state_action_;
    QAction *load_state_action_;
    QAction *pause_when_inactive_action_;
    QAction *show_fps_action_;
    QLabel *fps_label_;
    QDialog *super2xsai_dialog_ = nullptr;
    QString current_game_path_;
    bool auto_paused_for_focus_loss_ = false;
};
#endif // MAINWINDOW_H
