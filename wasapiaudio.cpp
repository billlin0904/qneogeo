#include "wasapiaudio.h"

#include <QDebug>

#include <algorithm>
#include <cstring>

namespace {
constexpr REFERENCE_TIME BufferDuration = 1000000; // 100 ms in 100 ns units.

QString hresultString(const QString &context, HRESULT result) {
    return QStringLiteral("%1 failed: 0x%2")
        .arg(context)
        .arg(static_cast<qulonglong>(static_cast<unsigned long>(result)), 8, 16, QLatin1Char('0'));
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

    if (!initializeCom())
        return false;

    if (!initializeDevice(sampleRate)) {
        releaseInterfaces();
        return false;
    }

    const HRESULT start_result = audio_client_->Start();
    if (FAILED(start_result)) {
        setError(hresultString(QStringLiteral("IAudioClient::Start"), start_result));
        releaseInterfaces();
        return false;
    }

    running_ = true;
    return true;
}

void WasapiAudio::stop() {
    if (audio_client_ && running_)
        audio_client_->Stop();

    running_ = false;
    releaseInterfaces();

    if (com_initialized_) {
        CoUninitialize();
        com_initialized_ = false;
    }
}

size_t WasapiAudio::writeSamples(const float *samples, size_t frames) {
    if (!running_ || !render_client_ || !audio_client_ || !samples || frames == 0)
        return 0;

    UINT32 padding = 0;
    HRESULT result = audio_client_->GetCurrentPadding(&padding);
    if (FAILED(result))
        return 0;

    if (padding >= buffer_frame_count_)
        return 0;

    const UINT32 available = buffer_frame_count_ - padding;
    const UINT32 frames_to_write = static_cast<UINT32>(std::min<size_t>(frames, available));
    if (frames_to_write == 0)
        return 0;

    BYTE *buffer = nullptr;
    result = render_client_->GetBuffer(frames_to_write, &buffer);
    if (FAILED(result))
        return 0;

    std::memcpy(buffer, samples, frames_to_write * format_.nBlockAlign);

    result = render_client_->ReleaseBuffer(frames_to_write, 0);
    if (FAILED(result))
        return 0;

    return frames_to_write;
}

QString WasapiAudio::lastError() const {
    return last_error_;
}

void WasapiAudio::setError(const QString &message) {
    last_error_ = message;
    qWarning().noquote() << message;
}

void WasapiAudio::releaseInterfaces() {
    if (render_client_) {
        render_client_->Release();
        render_client_ = nullptr;
    }

    if (audio_client_) {
        audio_client_->Release();
        audio_client_ = nullptr;
    }

    if (device_) {
        device_->Release();
        device_ = nullptr;
    }

    if (device_enumerator_) {
        device_enumerator_->Release();
        device_enumerator_ = nullptr;
    }

    buffer_frame_count_ = 0;
}

bool WasapiAudio::initializeCom() {
    const HRESULT result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (SUCCEEDED(result)) {
        com_initialized_ = true;
        return true;
    }

    if (result == RPC_E_CHANGED_MODE)
        return true;

    setError(hresultString(QStringLiteral("CoInitializeEx"), result));
    return false;
}

bool WasapiAudio::initializeDevice(int sampleRate) {
    HRESULT result = CoCreateInstance(__uuidof(MMDeviceEnumerator),
                                      nullptr,
                                      CLSCTX_ALL,
                                      __uuidof(IMMDeviceEnumerator),
                                      reinterpret_cast<void **>(&device_enumerator_));
    if (FAILED(result)) {
        setError(hresultString(QStringLiteral("CoCreateInstance(MMDeviceEnumerator)"), result));
        return false;
    }

    result = device_enumerator_->GetDefaultAudioEndpoint(eRender, eConsole, &device_);
    if (FAILED(result)) {
        setError(hresultString(QStringLiteral("GetDefaultAudioEndpoint"), result));
        return false;
    }

    result = device_->Activate(__uuidof(IAudioClient),
                               CLSCTX_ALL,
                               nullptr,
                               reinterpret_cast<void **>(&audio_client_));
    if (FAILED(result)) {
        setError(hresultString(QStringLiteral("IMMDevice::Activate(IAudioClient)"), result));
        return false;
    }

    format_.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
    format_.nChannels = 2;
    format_.nSamplesPerSec = static_cast<DWORD>(sampleRate);
    format_.wBitsPerSample = 32;
    format_.nBlockAlign = static_cast<WORD>((format_.nChannels * format_.wBitsPerSample) / 8);
    format_.nAvgBytesPerSec = format_.nSamplesPerSec * format_.nBlockAlign;
    format_.cbSize = 0;

    WAVEFORMATEX *closest_match = nullptr;
    result = audio_client_->IsFormatSupported(AUDCLNT_SHAREMODE_SHARED, &format_, &closest_match);
    if (closest_match)
        CoTaskMemFree(closest_match);

    if (result != S_OK) {
        setError(QStringLiteral("WASAPI shared mode does not support 48000 Hz stereo float on the default device."));
        return false;
    }

    result = audio_client_->Initialize(AUDCLNT_SHAREMODE_SHARED,
                                       0,
                                       BufferDuration,
                                       0,
                                       &format_,
                                       nullptr);
    if (FAILED(result)) {
        setError(hresultString(QStringLiteral("IAudioClient::Initialize"), result));
        return false;
    }

    result = audio_client_->GetBufferSize(&buffer_frame_count_);
    if (FAILED(result)) {
        setError(hresultString(QStringLiteral("IAudioClient::GetBufferSize"), result));
        return false;
    }

    result = audio_client_->GetService(__uuidof(IAudioRenderClient),
                                       reinterpret_cast<void **>(&render_client_));
    if (FAILED(result)) {
        setError(hresultString(QStringLiteral("IAudioClient::GetService(IAudioRenderClient)"), result));
        return false;
    }

    return true;
}
