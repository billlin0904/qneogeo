#include "kof98movedata.h"

#include <array>

namespace kof98 {
namespace {

constexpr int32_t LIGHT_GROUND_BLOCKSTUN = 9;
constexpr int32_t HEAVY_GROUND_BLOCKSTUN = 17;
constexpr int32_t BLOW_AWAY_GROUND_BLOCKSTUN = 21;
constexpr int32_t AIR_BLOCKSTUN = 10;

constexpr MoveData move(int32_t action_id,
                        MoveVariant variant,
                        MoveClass move_class,
                        int32_t startup,
                        int32_t active,
                        int32_t recovery,
                        int32_t reach_front,
                        int32_t reach_back,
                        int32_t movement_forward,
                        int32_t attack_y_min,
                        int32_t attack_y_max,
                        int32_t anti_ground_small_jump_y,
                        int32_t anti_ground_normal_jump_y,
                        uint32_t flags = MoveFlagNone) {
    const int32_t ground_blockstun = move_class == MoveClass::None
        ? UnknownMoveValue
        : (move_class == MoveClass::Light
               ? LIGHT_GROUND_BLOCKSTUN
               : (move_class == MoveClass::BlowAway
                      ? BLOW_AWAY_GROUND_BLOCKSTUN
                      : HEAVY_GROUND_BLOCKSTUN));
    return {
        action_id,
        variant,
        move_class,
        startup,
        active,
        recovery,
        reach_front,
        reach_back,
        movement_forward,
        attack_y_min,
        attack_y_max,
        anti_ground_small_jump_y,
        anti_ground_normal_jump_y,
        ground_blockstun,
        AIR_BLOCKSTUN,
        flags,
        MoveDataSource::PublishedTable,
    };
}

// 由 all-data1.xls 的「京」工作表正規化而來。多段技若無法從表格唯一拆出
// 每段時序，先保留 -1，等 runtime profiler 逐幀驗證後再補，避免猜測資料。
constexpr std::array<MoveData, 20> KYO_PUBLISHED_MOVES {{
    move(6, MoveVariant::Near, MoveClass::Light, 3, 4, 5, 36, 11, 0, 68, 91, -1, -1),
    move(6, MoveVariant::Far, MoveClass::Light, 3, 4, 5, 48, 7, 0, 76, 91, -1, -1),
    move(7, MoveVariant::Near, MoveClass::Light, 5, 4, 8, 64, 0, 0, 20, 43, -1, -1),
    move(7, MoveVariant::Far, MoveClass::Light, 7, 3, 13, 80, 0, 0, 56, 79, -1, -1),
    move(8, MoveVariant::Near, MoveClass::Heavy, 2, -1, 9, 44, 5, 0, 48, 115, -1, -1, MoveFlagMultiHit),
    move(8, MoveVariant::Far, MoveClass::Heavy, 12, 2, 17, 64, 17, 0, 68, 91, -1, -1),
    move(9, MoveVariant::Near, MoveClass::Heavy, 4, 3, 22, 60, 5, 0, 52, 83, -1, -1),
    move(9, MoveVariant::Far, MoveClass::Heavy, 14, 5, 12, 60, 0, 12, 52, 75, -1, -1),
    move(10, MoveVariant::Default, MoveClass::Light, 3, 4, 5, 44, 0, 0, 44, 59, -1, -1),
    move(11, MoveVariant::Default, MoveClass::Light, 3, 4, 5, 52, 0, 0, 0, 15, -1, -1),
    move(12, MoveVariant::Default, MoveClass::Heavy, 4, -1, 20, 32, 7, 0, 56, 135, -1, -1, MoveFlagMultiHit),
    move(13, MoveVariant::Default, MoveClass::Heavy, 11, 3, 19, 68, 0, 0, 20, 43, -1, -1),
    move(14, MoveVariant::Default, MoveClass::Special, 11, 8, 20, 80, 0, 24, 44, 91, -1, -1,
         MoveFlagGuardPoint | MoveFlagProjectile),
    move(17, MoveVariant::Default, MoveClass::Special, 16, 6, 18, 68, 0, -1, -1, -1, 69, -1,
         MoveFlagMultiHit | MoveFlagAirborne),
    move(20, MoveVariant::Default, MoveClass::Heavy, 6, 4, -1, 36, 19, 0, 12, 35, 51, 46,
         MoveFlagAirborne),
    move(21, MoveVariant::Default, MoveClass::Heavy, 9, 5, -1, 60, 11, 0, 20, 43, 59, 54,
         MoveFlagAirborne),
    move(22, MoveVariant::Default, MoveClass::Heavy, 11, -1, 18, 84, 0, 0, 44, 112, -1, -1,
         MoveFlagMultiHit),
    move(23, MoveVariant::Default, MoveClass::Special, 17, 6, 17, 80, 0, -1, 40, 95, -1, -1,
         MoveFlagGuardPoint | MoveFlagProjectile),
    move(19, MoveVariant::Default, MoveClass::Super, -1, -1, 45, 88, 0, -10, -1, -1, -1, -1,
         MoveFlagMultiHit | MoveFlagProjectile),
    move(5, MoveVariant::Default, MoveClass::None, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1,
         MoveFlagAirborne),
}};

MoveClass fallbackMoveClass(int32_t action_id) {
    switch (action_id) {
    case 6:
    case 7:
    case 10:
    case 11:
        return MoveClass::Light;
    case 8:
    case 9:
    case 12:
    case 13:
    case 20:
    case 21:
    case 22:
        return MoveClass::Heavy;
    case 14:
    case 15:
    case 16:
    case 17:
    case 23:
    case 24:
    case 25:
    case 26:
    case 27:
    case 28:
        return MoveClass::Special;
    case 18:
    case 19:
        return MoveClass::Super;
    default:
        return MoveClass::None;
    }
}

} // namespace

const MoveData *findKyoMoveData(int32_t action_id, MoveVariant variant) {
    const MoveData *fallback = nullptr;
    for (const MoveData &data : KYO_PUBLISHED_MOVES) {
        if (data.action_id != action_id)
            continue;
        if (data.variant == variant)
            return &data;
        if (!fallback || data.variant == MoveVariant::Default)
            fallback = &data;
    }
    return fallback;
}

int32_t expectedBlockstunFrames(int32_t action_id, bool defender_airborne) {
    if (defender_airborne)
        return fallbackMoveClass(action_id) == MoveClass::None
            ? UnknownMoveValue
            : AIR_BLOCKSTUN;

    const MoveClass move_class = fallbackMoveClass(action_id);
    switch (move_class) {
    case MoveClass::Light:
        return LIGHT_GROUND_BLOCKSTUN;
    case MoveClass::BlowAway:
        return BLOW_AWAY_GROUND_BLOCKSTUN;
    case MoveClass::Heavy:
    case MoveClass::Special:
    case MoveClass::Super:
        return HEAVY_GROUND_BLOCKSTUN;
    default:
        return UnknownMoveValue;
    }
}

} // namespace kof98
