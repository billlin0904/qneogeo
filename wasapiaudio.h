#pragma once

#include <QObject>
#include <QString>

#include <audioclient.h>
#include <mmdeviceapi.h>

#include <cstdint>

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
    void setError(const QString &message);
    void releaseInterfaces();
    bool initializeCom();
    bool initializeDevice(int sampleRate);

    QString last_error_;
    bool com_initialized_ = false;
    bool running_ = false;
    UINT32 buffer_frame_count_ = 0;
    WAVEFORMATEX format_ {};

    IMMDeviceEnumerator *device_enumerator_ = nullptr;
    IMMDevice *device_ = nullptr;
    IAudioClient *audio_client_ = nullptr;
    IAudioRenderClient *render_client_ = nullptr;
};
