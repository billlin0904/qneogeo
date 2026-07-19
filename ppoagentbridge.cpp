#include "ppoagentbridge.h"

#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

PpoAgentBridge::PpoAgentBridge(QObject *parent)
    : QObject(parent) {
    process_.setProcessChannelMode(QProcess::SeparateChannels);
    connect(&process_, &QProcess::readyReadStandardOutput,
            this, &PpoAgentBridge::readStandardOutput);
    connect(&process_, &QProcess::readyReadStandardError,
            this, &PpoAgentBridge::readStandardError);
    connect(&process_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
        setFailure(process_.errorString());
    });
    connect(&process_,
            qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
            this,
            [this](int exitCode, QProcess::ExitStatus) {
                if (ready_ || exitCode != 0) {
                    QString detail = QString::fromUtf8(stderr_buffer_).trimmed();
                    if (detail.isEmpty())
                        detail = QStringLiteral("PPO bridge exited with code %1.").arg(exitCode);
                    setFailure(detail);
                }
            });
}

bool PpoAgentBridge::start(const QString &pythonPath,
                           const QString &scriptPath,
                           const QString &modelPath,
                           const QString &comboModelPath) {
    stop();
    if (!QFileInfo::exists(pythonPath)) {
        setFailure(QStringLiteral("Python executable not found: %1").arg(pythonPath));
        return false;
    }
    if (!QFileInfo::exists(scriptPath)) {
        setFailure(QStringLiteral("PPO bridge script not found: %1").arg(scriptPath));
        return false;
    }
    if (!QFileInfo::exists(modelPath)) {
        setFailure(QStringLiteral("PPO model not found: %1").arg(modelPath));
        return false;
    }
    if (!QFileInfo::exists(comboModelPath)) {
        setFailure(
            QStringLiteral("PPO combo model not found: %1").arg(comboModelPath));
        return false;
    }

    last_error_.clear();
    stderr_buffer_.clear();
    stdout_buffer_.clear();
    process_.setProgram(pythonPath);
    process_.setArguments({
        QStringLiteral("-u"),
        scriptPath,
        QStringLiteral("--model"),
        modelPath,
        QStringLiteral("--combo-model"),
        comboModelPath,
        QStringLiteral("--device"),
        QStringLiteral("cpu"),
    });
    process_.start();
    if (!process_.waitForStarted(3000)) {
        setFailure(process_.errorString());
        return false;
    }
    return true;
}

void PpoAgentBridge::stop() {
    ready_ = false;
    pending_request_id_ = 0;
    stdout_buffer_.clear();
    if (process_.state() == QProcess::NotRunning)
        return;

    process_.closeWriteChannel();
    if (!process_.waitForFinished(1000)) {
        process_.kill();
        process_.waitForFinished(1000);
    }
}

bool PpoAgentBridge::isReady() const {
    return ready_ && process_.state() == QProcess::Running;
}

bool PpoAgentBridge::hasPendingRequest() const {
    return pending_request_id_ != 0;
}

bool PpoAgentBridge::requestAction(const QVector<float> &observation,
                                   const QVector<bool> &mask) {
    if (!isReady() || hasPendingRequest())
        return false;

    QJsonArray observation_json;
    for (float value : observation)
        observation_json.append(value);

    QJsonArray mask_json;
    for (bool enabled : mask)
        mask_json.append(enabled);

    const qint64 request_id = next_request_id_++;
    QJsonObject request {
        { QStringLiteral("id"), request_id },
        { QStringLiteral("observation"), observation_json },
        { QStringLiteral("mask"), mask_json },
    };
    QByteArray data = QJsonDocument(request).toJson(QJsonDocument::Compact);
    data.append('\n');
    if (process_.write(data) != data.size())
        return false;

    pending_request_id_ = request_id;
    return true;
}

QString PpoAgentBridge::lastError() const {
    return last_error_;
}

void PpoAgentBridge::readStandardOutput() {
    stdout_buffer_.append(process_.readAllStandardOutput());
    while (true) {
        const qsizetype newline = stdout_buffer_.indexOf('\n');
        if (newline < 0)
            break;
        const QByteArray line = stdout_buffer_.left(newline).trimmed();
        stdout_buffer_.remove(0, newline + 1);
        if (!line.isEmpty())
            processMessage(line);
    }
}

void PpoAgentBridge::readStandardError() {
    stderr_buffer_.append(process_.readAllStandardError());
    constexpr qsizetype MAX_STDERR_SIZE = 16 * 1024;
    if (stderr_buffer_.size() > MAX_STDERR_SIZE)
        stderr_buffer_ = stderr_buffer_.right(MAX_STDERR_SIZE);
}

void PpoAgentBridge::processMessage(const QByteArray &line) {
    QJsonParseError parse_error;
    const QJsonDocument document = QJsonDocument::fromJson(line, &parse_error);
    if (parse_error.error != QJsonParseError::NoError || !document.isObject()) {
        setFailure(QStringLiteral("Invalid PPO bridge response: %1")
                       .arg(QString::fromUtf8(line)));
        return;
    }

    const QJsonObject object = document.object();
    const QString type = object.value(QStringLiteral("type")).toString();
    if (type == QStringLiteral("ready")) {
        const int observation_size =
            object.value(QStringLiteral("observation_size")).toInt();
        const int action_count = object.value(QStringLiteral("action_count")).toInt();
        if (observation_size != 26 || action_count != 29) {
            setFailure(
                QStringLiteral("PPO model ABI mismatch: expected 26 observations "
                               "and 29 actions, got %1 and %2.")
                    .arg(observation_size)
                    .arg(action_count));
            return;
        }
        ready_ = true;
        emit ready();
        return;
    }

    if (type == QStringLiteral("action")) {
        const qint64 request_id =
            object.value(QStringLiteral("id")).toInteger();
        if (request_id != pending_request_id_)
            return;
        pending_request_id_ = 0;
        emit actionReady(object.value(QStringLiteral("action")).toInt());
        return;
    }

    if (type == QStringLiteral("error")) {
        pending_request_id_ = 0;
        setFailure(object.value(QStringLiteral("message")).toString());
    }
}

void PpoAgentBridge::setFailure(const QString &message) {
    if (message.isEmpty())
        return;
    const bool changed = last_error_ != message;
    last_error_ = message;
    ready_ = false;
    pending_request_id_ = 0;
    if (changed)
        emit failed(message);
}
