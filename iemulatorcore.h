#pragma once

#include <QObject>
#include <QString>
#include <QStringList>

class IEmulatorCore : public QObject {
    Q_OBJECT

public:
    explicit IEmulatorCore(QObject *parent = nullptr);
    ~IEmulatorCore() override;

    virtual QString displayName() const = 0;
    virtual QString coreFileName() const = 0;
    virtual QString romDirectoryName() const = 0;
    virtual QStringList supportedExtensions() const = 0;

    virtual bool loadCore(const QString &corePath) = 0;
    virtual bool startGame(const QString &contentPath, const QString &systemDirectory, const QString &saveDirectory) = 0;
    virtual void stop() = 0;
    virtual bool reset() = 0;
    virtual void setPaused(bool paused) = 0;
    virtual bool isPaused() const = 0;
    virtual bool isGameLoaded() const = 0;
    virtual bool saveState(const QString &statePath) = 0;
    virtual bool loadState(const QString &statePath) = 0;
    virtual QString lastError() const = 0;

signals:
    void pausedChanged(bool paused);
};
