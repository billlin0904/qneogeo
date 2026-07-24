#pragma once

#include <QObject>
#include <QProcess>

class PpoAgentBridge final : public QObject {
    Q_OBJECT

public:
    explicit PpoAgentBridge(QObject *parent = nullptr);

    bool start(const QString &pythonPath,
               const QString &scriptPath,
               const QString &modelPath,
               const QString &comboModelPath,
               bool purePolicy);
    void stop();
    bool isReady() const;
    int observationSize() const;
    QString observationSchemaId() const;
    bool hasPendingRequest() const;
    bool requestAction(const QVector<float> &observation,
                       const QVector<bool> &mask);
    QString lastError() const;

signals:
    void ready();
    void actionReady(int32_t actionId);
    void failed(const QString &message);

private:
    void readStandardOutput();
    void readStandardError();
    void processMessage(const QByteArray &line);
    void setFailure(const QString &message);

    QProcess process_;
    QByteArray stdout_buffer_;
    QByteArray stderr_buffer_;
    QString last_error_;
    qint64 next_request_id_ = 1;
    qint64 pending_request_id_ = 0;
    int observation_size_ = 0;
    QString observation_schema_id_;
    bool ready_ = false;
};
