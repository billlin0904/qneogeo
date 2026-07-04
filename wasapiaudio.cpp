#include "wasapiaudio.h"

#include <QDebug>

#include <rtaudio/RtAudio.h>

#include <algorithm>
#include <cstring>

namespace {
constexpr unsigned int ChannelCount = 2;
constexpr unsigned int PreferredBufferFrames = 64;
constexpr size_t RingBufferSeconds = 1;
constexpr size_t MaxLatencyMilliseconds = 30;

QString rtAudioErrorMessage(RtAudioErrorType error) {
    switch (error) {
    case RTAUDIO_NO_ERROR:
        return QString();
    case RTAUDIO_WARNING:
        return QStringLiteral("RtAudio warning.");
    case RTAUDIO_NO_DEVICES_FOUND:
        return QStringLiteral("RtAudio found no audio devices.");
    case RTAUDIO_INVALID_DEVICE:
        return QStringLiteral("RtAudio invalid audio device.");
    case RTAUDIO_DEVICE_DISCONNECT:
        return QStringLiteral("RtAudio device disconnected.");
    case RTAUDIO_MEMORY_ERROR:
        return QStringLiteral("RtAudio memory error.");
    case RTAUDIO_INVALID_PARAMETER:
        return QStringLiteral("RtAudio invalid parameter.");
    case RTAUDIO_INVALID_USE:
        return QStringLiteral("RtAudio invalid use.");
    case RTAUDIO_DRIVER_ERROR:
        return QStringLiteral("RtAudio driver error.");
    case RTAUDIO_SYSTEM_ERROR:
        return QStringLiteral("RtAudio system error.");
    case RTAUDIO_THREAD_ERROR:
        return QStringLiteral("RtAudio thread error.");
    case RTAUDIO_UNKNOWN_ERROR:
    default:
        return QStringLiteral("RtAudio unknown error.");
    }
}
}

WasapiAudio::WasapiAudio(QObject *parent)
    : QObject(parent) {
}

WasapiAudio::~WasapiAudio() {
    stop();
}

bool WasapiAudio::start(int sampleRate) {
    stop();

    if (sampleRate <= 0) {
        setError(QStringLiteral("Invalid audio sample rate."));
        return false;
    }

    audio_ = std::make_unique<RtAudio>(RtAudio::WINDOWS_WASAPI, [this](RtAudioErrorType, const std::string &message) {
        last_error_ = QString::fromStdString(message);
        qWarning().noquote() << last_error_;
    });

    if (audio_->getDeviceCount() == 0) {
        setError(QStringLiteral("RtAudio could not find an output device."));
        audio_.reset();
        return false;
    }

    RtAudio::StreamParameters output_parameters;
    output_parameters.deviceId = audio_->getDefaultOutputDevice();
    output_parameters.nChannels = ChannelCount;
    output_parameters.firstChannel = 0;

    RtAudio::StreamOptions options;
    options.flags = RTAUDIO_MINIMIZE_LATENCY;
    options.streamName = "qneogeo";

    unsigned int buffer_frames = PreferredBufferFrames;
    RtAudioErrorType result = audio_->openStream(&output_parameters,
                                                 nullptr,
                                                 RTAUDIO_FLOAT32,
                                                 static_cast<unsigned int>(sampleRate),
                                                 &buffer_frames,
                                                 &WasapiAudio::audioCallback,
                                                 this,
                                                 &options);
    if (result != RTAUDIO_NO_ERROR) {
        setError(last_error_.isEmpty() ? rtAudioErrorMessage(result) : last_error_);
        audio_.reset();
        return false;
    }

    resetBuffer(static_cast<size_t>(sampleRate) * RingBufferSeconds);

    result = audio_->startStream();
    if (result != RTAUDIO_NO_ERROR && result != RTAUDIO_WARNING) {
        setError(last_error_.isEmpty() ? rtAudioErrorMessage(result) : last_error_);
        audio_->closeStream();
        audio_.reset();
        return false;
    }

    running_ = true;
    return true;
}

void WasapiAudio::stop() {
    if (audio_) {
        if (audio_->isStreamRunning())
            audio_->stopStream();
        if (audio_->isStreamOpen())
            audio_->closeStream();
    }

    running_ = false;
    audio_.reset();

    {
        std::lock_guard lock(buffer_mutex_);
        ring_buffer_.clear();
        read_frame_ = 0;
        write_frame_ = 0;
        queued_frames_ = 0;
        max_queued_frames_ = 0;
    }
}

size_t WasapiAudio::writeSamples(const float *samples, size_t frames) {
    if (!running_ || !samples || frames == 0)
        return 0;

    std::lock_guard lock(buffer_mutex_);
    if (ring_buffer_.empty())
        return 0;

    const size_t capacity = ring_buffer_.size() / ChannelCount;
    const size_t frames_to_write = std::min(frames, capacity);

    const size_t free_frames = freeFrames();
    const size_t queued_budget = max_queued_frames_ > queued_frames_ ? max_queued_frames_ - queued_frames_ : 0;
    const size_t accepted_without_latency_growth = std::min(free_frames, queued_budget);
    if (frames_to_write > accepted_without_latency_growth) {
        const size_t frames_to_drop = frames_to_write - accepted_without_latency_growth;
        read_frame_ = (read_frame_ + frames_to_drop) % capacity;
        queued_frames_ -= std::min(frames_to_drop, queued_frames_);
    }

    size_t source_frame = frames - frames_to_write;
    size_t written = 0;

    while (written < frames_to_write) {
        const size_t chunk = std::min(frames_to_write - written, capacity - write_frame_);
        std::memcpy(&ring_buffer_[write_frame_ * ChannelCount],
                    &samples[source_frame * ChannelCount],
                    chunk * ChannelCount * sizeof(float));

        write_frame_ = (write_frame_ + chunk) % capacity;
        source_frame += chunk;
        written += chunk;
    }

    queued_frames_ += written;
    return written;
}

QString WasapiAudio::lastError() const {
    return last_error_;
}

int WasapiAudio::audioCallback(void *outputBuffer,
                               void *,
                               unsigned int frameCount,
                               double,
                               unsigned int status,
                               void *userData) {
    auto *self = static_cast<WasapiAudio *>(userData);
    if (status != 0)
        qWarning().noquote() << "RtAudio stream status:" << status;

    return self ? self->render(static_cast<float *>(outputBuffer), frameCount) : 0;
}

int WasapiAudio::render(float *outputBuffer, unsigned int frameCount) {
    if (!outputBuffer || frameCount == 0)
        return 0;

    std::fill(outputBuffer, outputBuffer + (frameCount * ChannelCount), 0.0f);

    std::lock_guard lock(buffer_mutex_);
    if (ring_buffer_.empty())
        return 0;

    const size_t capacity = ring_buffer_.size() / ChannelCount;
    const size_t frames_to_read = std::min<size_t>(frameCount, queued_frames_);
    size_t read = 0;

    while (read < frames_to_read) {
        const size_t chunk = std::min(frames_to_read - read, capacity - read_frame_);
        std::memcpy(outputBuffer + (read * ChannelCount),
                    &ring_buffer_[read_frame_ * ChannelCount],
                    chunk * ChannelCount * sizeof(float));

        read_frame_ = (read_frame_ + chunk) % capacity;
        read += chunk;
    }

    queued_frames_ -= read;
    return 0;
}

void WasapiAudio::setError(const QString &message) {
    last_error_ = message;
    qWarning().noquote() << message;
}

void WasapiAudio::resetBuffer(size_t frameCapacity) {
    std::lock_guard lock(buffer_mutex_);
    ring_buffer_.assign(frameCapacity * ChannelCount, 0.0f);
    read_frame_ = 0;
    write_frame_ = 0;
    queued_frames_ = 0;
    max_queued_frames_ = std::max<size_t>(1, frameCapacity * MaxLatencyMilliseconds / (RingBufferSeconds * 1000));
}

size_t WasapiAudio::freeFrames() const {
    const size_t capacity = ring_buffer_.size() / ChannelCount;
    return capacity - queued_frames_;
}
