#pragma once

#include "emulatorview.h"
#include "gamememreadercore.h"

#include <QByteArray>
#include <QPoint>
#include <QSize>
#include <QVector>

#include <cstdint>

struct HitboxOverlay {
    QVector<EmulatorView::HitboxRect> boxes;
    QVector<EmulatorView::HitboxAxis> axes;
};

class GameMemReader final {
public:
    static constexpr int32_t MaxHealth = 103;
    static constexpr int32_t MaxPower = 128;
    static constexpr int32_t MaxTime = 99;

    explicit GameMemReader(QByteArray ram, QSize sourceSize);

    HitboxOverlay getHitboxOverlay() const;
    int32_t readRoundTime() const;
    int32_t readP1Health() const;
    int32_t readP2Health() const;
    int32_t readP1Power() const;
    int32_t readP2Power() const;
    int32_t readP1PowerState() const;
    int32_t readP2PowerState() const;
    int32_t readP1AdvancedPowerValue() const;
    int32_t readP1AdvancedPowerStocks() const;
    int32_t readP2AdvancedPowerValue() const;
    int32_t readP2AdvancedPowerStocks() const;
    int32_t readP1Stun() const;
    int32_t readP2Stun() const;
    int32_t readP1ComboCount() const;
    int32_t readP2ComboCount() const;
    int32_t readP1HitGuardStopRaw() const;
    int32_t readP2HitGuardStopRaw() const;
    int32_t readP1RecoveryControlRaw() const;
    int32_t readP2RecoveryControlRaw() const;
    int32_t readP1BlockStateRaw() const;
    int32_t readP2BlockStateRaw() const;
    int32_t readP1ReactionD2Raw() const;
    int32_t readP2ReactionD2Raw() const;
    game_memory::PlayerReactionDebugState readP1ReactionDebugState() const;
    game_memory::PlayerReactionDebugState readP2ReactionDebugState() const;
    bool readP1Position(QPoint &position) const;
    bool readP2Position(QPoint &position) const;

private:
    QByteArray ram_;
    QSize source_size_;
};
