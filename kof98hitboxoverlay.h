#pragma once

#include "emulatorview.h"

#include <QByteArray>
#include <QPoint>
#include <QSize>
#include <QVector>

class KofHitboxOverlayBuilder final {
public:
    static constexpr int MaxHealth = 103;
    static constexpr int MaxPower = 128;
    static constexpr int MaxTime = 99;

    struct Result {
        QVector<EmulatorView::HitboxRect> boxes;
        QVector<EmulatorView::HitboxAxis> axes;
    };

    explicit KofHitboxOverlayBuilder(QByteArray ram, QSize sourceSize);

    Result build() const;
    int readRoundTime() const;
    int readP1Health() const;
    int readP2Health() const;
    int readP1Power() const;
    int readP2Power() const;
    int readP1PowerState() const;
    int readP2PowerState() const;
    int readP1Stun() const;
    int readP2Stun() const;
    bool readP1Position(QPoint &position) const;
    bool readP2Position(QPoint &position) const;

    static Result build(QByteArray ram, QSize sourceSize);

private:
    QByteArray ram_;
    QSize source_size_;
};
