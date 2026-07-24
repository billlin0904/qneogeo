#pragma once

#include "emulatorview.h"

#include <QFile>
#include <QSize>
#include <QString>

#include <cstdint>

class KofRamTraceLogger final {
public:
    KofRamTraceLogger() = default;
    ~KofRamTraceLogger();

    bool start(const QString &file_path, QString &error);
    void stop();
    bool isActive() const;
    QString filePath() const;

    bool writeFrame(uint64_t frame_number,
                    const QByteArray &ram,
                    QSize source_size,
                    const EmulatorView::JoypadInput &p1_input,
                    const EmulatorView::JoypadInput &p2_input,
                    QString &error);

private:
    struct PreviousPlayerState {
        int32_t health = -1;
        int32_t guard_crush = -1;
        int32_t reaction_d2 = -1;
        int32_t reaction_e3 = -1;
    };

    QFile file_;
    PreviousPlayerState previous_p1_;
    PreviousPlayerState previous_p2_;
    bool has_previous_frame_ = false;
    uint64_t rows_since_flush_ = 0;
};
