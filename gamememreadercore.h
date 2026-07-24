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

// 尚未分類的角色反應資料，只供逐幀人工驗證。欄位名稱保留 RAM offset，
// 避免在證據不足時誤稱為 blockstun、hitstun 或 actionable state。
struct PlayerReactionDebugState {
    int32_t hit_guard_stop = -1;
    // 68000 邏輯位址 player+0xD2 與 player+0xD3 的原始 byte。
    int32_t reaction_d2 = -1;
    int32_t reaction_d3 = -1;
    // D2 是高位、D3 是低位。實測已確認這個 big-endian signed 16-bit
    // 值在反應硬直期間逐幀倒數，閒置值 FF:FF 解讀為 -1。
    int32_t reaction_d2d3_unsigned = -1;
    int32_t reaction_d2d3_signed = -1;
    int32_t reaction_e0 = -1;
    int32_t reaction_e1 = -1;
    int32_t reaction_e2 = -1;
    int32_t reaction_e3 = -1;
    int32_t d4_high = -1;
    int32_t d5_low = -1;
    int32_t d4_signed = -1;
    int32_t recovery_control_e7 = -1;
    int32_t guard_crush = -1;
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
    int32_t readP1HitboxActiveMask() const;
    int32_t readP2HitboxActiveMask() const;
    int32_t readP1GuardCrushValue() const;
    int32_t readP2GuardCrushValue() const;
    // player + 0x125 的原始 Hit/Guard Stop 值。逐幀 trace 顯示它在中立、
    // hitstop 與 blockstun 期間皆可維持固定值，因此不是剩餘幀數倒數。
    int32_t readP1HitGuardStopRaw() const;
    int32_t readP2HitGuardStopRaw() const;
    // player + 0xE7 的原始值。Cheat 會寫入 0x81 解除硬直，但欄位語意
    // 尚未確認，因此目前只提供唯讀觀察，不把它視為 blockstun timer。
    int32_t readP1RecoveryControlRaw() const;
    int32_t readP2RecoveryControlRaw() const;
    // player + 0xE3 的原始 bitfield。Fightcade extension 將 0x20/0xA0
    // 當成 guard-reversal 觸發點，MAME-RR 則用 bit 5 排除可投框；它不等於
    // 已確認的 blockstun timer，因此目前不回傳 blockstun bool。
    int32_t readP1BlockStateRaw() const;
    int32_t readP2BlockStateRaw() const;
    // player + 0xD2 的原始高位 byte。單看 D2 只會看到 FF→00→FF；
    // 完整倒數必須連同 player+0xD3 以 big-endian signed 16-bit 解讀。
    int32_t readP1ReactionD2Raw() const;
    int32_t readP2ReactionD2Raw() const;
    PlayerReactionDebugState readP1ReactionDebugState() const;
    PlayerReactionDebugState readP2ReactionDebugState() const;
    // 相容舊呼叫端：這兩個方法回傳的是 hitbox active mask，並不是
    // blockstun、animation 或角色狀態 enum。
    int32_t readP1Status() const;
    int32_t readP2Status() const;
    bool readP1Position(Point &position) const;
    bool readP2Position(Point &position) const;
    bool readP1FacingLeft(bool &facing_left) const;
    bool readP2FacingLeft(bool &facing_left) const;
    bool p1ReadyForAction() const;
    bool p2ReadyForAction() const;

private:
    const uint8_t *ram_ = nullptr;
    size_t ram_size_ = 0;
    int32_t source_width_ = 320;
};

} // namespace game_memory
