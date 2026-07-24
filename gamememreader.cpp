#include "gamememreader.h"

#include "gamememreadercore.h"

#include <QColor>
#include <QPointF>
#include <QRectF>

#include <utility>

namespace {

game_memory::GameMemReaderCore makeCore(const QByteArray &ram, QSize source_size) {
    return game_memory::GameMemReaderCore(
        reinterpret_cast<const uint8_t *>(ram.constData()),
        static_cast<size_t>(ram.size()),
        source_size.width());
}

struct HitboxPalette {
    QColor fill;
    QColor outline;
};

HitboxPalette paletteForHitboxType(int32_t type) {
    switch (type) {
    case game_memory::HitboxAttack:
        return { QColor(255, 0, 0, 64), QColor(255, 0, 0, 255) };
    case game_memory::HitboxProjectileAttack:
        return { QColor(255, 128, 0, 64), QColor(255, 128, 0, 255) };
    case game_memory::HitboxVulnerability:
        return { QColor(0, 64, 255, 48), QColor(0, 64, 255, 255) };
    case game_memory::HitboxProjectileVulnerability:
        return { QColor(0, 220, 255, 48), QColor(0, 220, 255, 255) };
    case game_memory::HitboxPush:
        return { QColor(0, 255, 0, 40), QColor(0, 255, 0, 255) };
    case game_memory::HitboxGuard:
        return { QColor(255, 255, 0, 48), QColor(255, 255, 0, 255) };
    case game_memory::HitboxThrow:
    case game_memory::HitboxAxisThrow:
        return { QColor(255, 0, 255, 56), QColor(255, 0, 255, 255) };
    case game_memory::HitboxThrowable:
        return { QColor(255, 255, 255, 40), QColor(255, 255, 255, 255) };
    case game_memory::HitboxUndefined:
    default:
        return { QColor(160, 160, 160, 40), QColor(160, 160, 160, 255) };
    }
}

} // namespace

GameMemReader::GameMemReader(QByteArray ram, QSize sourceSize)
    : ram_(std::move(ram))
    , source_size_(sourceSize) {
}

HitboxOverlay GameMemReader::getHitboxOverlay() const {
    HitboxOverlay result;
    const game_memory::HitboxOverlay core_overlay = makeCore(ram_, source_size_).getHitboxOverlay();

    result.boxes.reserve(static_cast<qsizetype>(core_overlay.boxes.size()));
    for (const game_memory::HitboxRect &box : core_overlay.boxes) {
        const HitboxPalette palette = paletteForHitboxType(box.type);
        result.boxes.push_back({
            QRectF(box.left, box.top, box.width, box.height),
            palette.fill,
            palette.outline,
        });
    }

    result.axes.reserve(static_cast<qsizetype>(core_overlay.axes.size()));
    for (const game_memory::HitboxAxis &axis : core_overlay.axes)
        result.axes.push_back({ QPointF(axis.x, axis.y), QColor(255, 255, 255, 255) });

    return result;
}

int32_t GameMemReader::readRoundTime() const {
    return makeCore(ram_, source_size_).readRoundTime();
}

int32_t GameMemReader::readP1Health() const {
    return makeCore(ram_, source_size_).readP1Health();
}

int32_t GameMemReader::readP2Health() const {
    return makeCore(ram_, source_size_).readP2Health();
}

int32_t GameMemReader::readP1Power() const {
    return makeCore(ram_, source_size_).readP1Power();
}

int32_t GameMemReader::readP2Power() const {
    return makeCore(ram_, source_size_).readP2Power();
}

int32_t GameMemReader::readP1PowerState() const {
    return makeCore(ram_, source_size_).readP1PowerState();
}

int32_t GameMemReader::readP2PowerState() const {
    return makeCore(ram_, source_size_).readP2PowerState();
}

int32_t GameMemReader::readP1AdvancedPowerValue() const {
    return makeCore(ram_, source_size_).readP1AdvancedPowerValue();
}

int32_t GameMemReader::readP1AdvancedPowerStocks() const {
    return makeCore(ram_, source_size_).readP1AdvancedPowerStocks();
}

int32_t GameMemReader::readP2AdvancedPowerValue() const {
    return makeCore(ram_, source_size_).readP2AdvancedPowerValue();
}

int32_t GameMemReader::readP2AdvancedPowerStocks() const {
    return makeCore(ram_, source_size_).readP2AdvancedPowerStocks();
}

int32_t GameMemReader::readP1Stun() const {
    return makeCore(ram_, source_size_).readP1Stun();
}

int32_t GameMemReader::readP2Stun() const {
    return makeCore(ram_, source_size_).readP2Stun();
}

int32_t GameMemReader::readP1ComboCount() const {
    return makeCore(ram_, source_size_).readP1ComboCount();
}

int32_t GameMemReader::readP2ComboCount() const {
    return makeCore(ram_, source_size_).readP2ComboCount();
}

int32_t GameMemReader::readP1HitGuardStopRaw() const {
    return makeCore(ram_, source_size_).readP1HitGuardStopRaw();
}

int32_t GameMemReader::readP2HitGuardStopRaw() const {
    return makeCore(ram_, source_size_).readP2HitGuardStopRaw();
}

int32_t GameMemReader::readP1RecoveryControlRaw() const {
    return makeCore(ram_, source_size_).readP1RecoveryControlRaw();
}

int32_t GameMemReader::readP2RecoveryControlRaw() const {
    return makeCore(ram_, source_size_).readP2RecoveryControlRaw();
}

int32_t GameMemReader::readP1BlockStateRaw() const {
    return makeCore(ram_, source_size_).readP1BlockStateRaw();
}

int32_t GameMemReader::readP2BlockStateRaw() const {
    return makeCore(ram_, source_size_).readP2BlockStateRaw();
}

int32_t GameMemReader::readP1ReactionD2Raw() const {
    return makeCore(ram_, source_size_).readP1ReactionD2Raw();
}

int32_t GameMemReader::readP2ReactionD2Raw() const {
    return makeCore(ram_, source_size_).readP2ReactionD2Raw();
}

game_memory::PlayerReactionDebugState GameMemReader::readP1ReactionDebugState() const {
    return makeCore(ram_, source_size_).readP1ReactionDebugState();
}

game_memory::PlayerReactionDebugState GameMemReader::readP2ReactionDebugState() const {
    return makeCore(ram_, source_size_).readP2ReactionDebugState();
}

bool GameMemReader::readP1Position(QPoint &position) const {
    game_memory::Point core_position;
    if (!makeCore(ram_, source_size_).readP1Position(core_position))
        return false;

    position = QPoint(core_position.x, core_position.y);
    return true;
}

bool GameMemReader::readP2Position(QPoint &position) const {
    game_memory::Point core_position;
    if (!makeCore(ram_, source_size_).readP2Position(core_position))
        return false;

    position = QPoint(core_position.x, core_position.y);
    return true;
}
