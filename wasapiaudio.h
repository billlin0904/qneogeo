#pragma once

#include <QObject>
#include <QString>

#include <memory>
#include <mutex>
#include <vector>
#include <cstdint>

class RtAudio;

class WasapiAudio final : public QObject {
    Q_OBJECT

public:
    explicit WasapiAudio(QObject *parent = nullptr);
    ~WasapiAudio() override;

    bool start(int sampleRate);
    void stop();
    size_t writeSamples(const float *samples, size_t frames);

    QString lastError() const;

private:
    static int audioCallback(void *outputBuffer,
                             void *inputBuffer,
                             unsigned int frameCount,
                             double streamTime,
                             unsigned int status,
                             void *userData);

    int render(float *outputBuffer, unsigned int frameCount);
    void setError(const QString &message);
    void resetBuffer(size_t frameCapacity);
    size_t freeFrames() const;

    QString last_error_;
    std::unique_ptr<RtAudio> audio_;
    std::mutex buffer_mutex_;
    std::vector<float> ring_buffer_;
    size_t read_frame_ = 0;
    size_t write_frame_ = 0;
    size_t queued_frames_ = 0;
    size_t max_queued_frames_ = 0;
    bool running_ = false;
};
