#pragma once

#include <cstdint>

namespace kof98 {

constexpr int32_t UnknownMoveValue = -1;

// 同一個按鍵可能依遠近切換成不同普通技；Default 用於沒有遠近版本的招式。
enum class MoveVariant : int32_t {
    Default = 0,
    Near = 1,
    Far = 2,
};

enum class MoveClass : int32_t {
    None = 0,
    Light = 1,
    Heavy = 2,
    BlowAway = 3,
    Special = 4,
    Super = 5,
};

enum class MoveDataSource : int32_t {
    Unknown = 0,
    PublishedTable = 1,
    RuntimeMeasured = 2,
};

enum MoveFlags : uint32_t {
    MoveFlagNone = 0,
    MoveFlagMultiHit = 1u << 0,
    MoveFlagGuardPoint = 1u << 1,
    MoveFlagProjectile = 1u << 2,
    MoveFlagAirborne = 1u << 3,
};

// 欄位以遊戲邏輯 frame 與 Neo Geo 畫面座標表示。
// -1 代表來源資料未提供，不能拿 0 代替，否則會把未知值誤判成即時發生。
struct MoveData {
    int32_t action_id = -1;
    MoveVariant variant = MoveVariant::Default;
    MoveClass move_class = MoveClass::None;
    int32_t startup_frames = UnknownMoveValue;
    int32_t active_frames = UnknownMoveValue;
    int32_t recovery_frames = UnknownMoveValue;
    int32_t reach_front = UnknownMoveValue;
    int32_t reach_back = UnknownMoveValue;
    int32_t movement_forward = 0;
    int32_t attack_y_min = UnknownMoveValue;
    int32_t attack_y_max = UnknownMoveValue;
    int32_t anti_ground_small_jump_y = UnknownMoveValue;
    int32_t anti_ground_normal_jump_y = UnknownMoveValue;
    int32_t ground_blockstun_frames = UnknownMoveValue;
    int32_t air_blockstun_frames = UnknownMoveValue;
    uint32_t flags = MoveFlagNone;
    MoveDataSource source = MoveDataSource::Unknown;
};

const MoveData *findKyoMoveData(int32_t action_id,
                                MoveVariant variant = MoveVariant::Default);
// 依招式分類取得系統規則中的預期硬直；這不是 RAM 的 Hit/Guard Stop counter。
int32_t expectedBlockstunFrames(int32_t action_id, bool defender_airborne);

} // namespace kof98
