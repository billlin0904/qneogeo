#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace game_memory {

enum HitboxType : int32_t {
    HitboxUndefined = 0,
    HitboxAttack = 1,
    HitboxVulnerability = 2,
    HitboxProjectileVulnerability = 3,
    HitboxProjectileAttack = 4,
    HitboxPush = 5,
    HitboxGuard = 6,
    HitboxThrow = 7,
    HitboxAxisThrow = 8,
    HitboxThrowable = 9,
};

struct Point {
    int32_t x = 0;
    int32_t y = 0;
};

struct HitboxRect {
    int32_t type = HitboxUndefined;
    int32_t owner = 0;
    int32_t left = 0;
    int32_t top = 0;
    int32_t width = 0;
    int32_t height = 0;
};

struct HitboxAxis {
    int32_t x = 0;
    int32_t y = 0;
};

struct HitboxOverlay {
    std::vector<HitboxRect> boxes;
    std::vector<HitboxAxis> axes;
};

class GameMemReaderCore final {
public:
    static constexpr int32_t MaxHealth = 103;
    static constexpr int32_t MaxPower = 128;
    static constexpr int32_t MaxTime = 99;

    GameMemReaderCore(const uint8_t *ram, size_t ram_size, int32_t source_width = 320);

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
    bool readP1Position(Point &position) const;
    bool readP2Position(Point &position) const;
    bool readP1FacingLeft(bool &facing_left) const;
    bool p1ReadyForAction() const;
    bool p2ReadyForAction() const;

private:
    const uint8_t *ram_ = nullptr;
    size_t ram_size_ = 0;
    int32_t source_width_ = 320;
};

} // namespace game_memory
