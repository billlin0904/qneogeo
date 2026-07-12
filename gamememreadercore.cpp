#include "gamememreadercore.h"

#include <algorithm>
#include <array>
#include <optional>

namespace game_memory {
namespace {

constexpr uint32_t KOF98_RAM_BASE = 0x100000;
constexpr uint32_t KOF98_OBJECT_STATE_OFFSET = 0x06;
constexpr uint32_t KOF98_PROJECTILE_LIST_SIZE = 0x200;
constexpr uint32_t KOF98_PROJECTILE_LIST_ENTRY_SIZE = 2;
constexpr uint32_t KOF98_HITBOX_ID_OFFSET = 0;
constexpr uint32_t KOF98_HITBOX_X_OFFSET = 1;
constexpr uint32_t KOF98_HITBOX_Y_OFFSET = 2;
constexpr uint32_t KOF98_HITBOX_RADIUS_X_OFFSET = 3;
constexpr uint32_t KOF98_HITBOX_RADIUS_Y_OFFSET = 4;
constexpr uint32_t KOF98_P1_PLAYER_BASE = 0x108100;
constexpr uint32_t KOF98_P2_PLAYER_BASE = 0x108300;
constexpr uint32_t KOF98_PLAYER_POS_X_OFFSET = 0x18;
constexpr uint32_t KOF98_PLAYER_POS_Y_OFFSET = 0x26;
constexpr uint32_t KOF98_PLAYER_FLIP_X_OFFSET = 0x31;
constexpr uint32_t KOF98_PLAYER_STATUS_OFFSET = 0x7C;
constexpr uint32_t KOF98_PLAYER_HEALTH_OFFSET = 0x138;
constexpr uint32_t KOF98_PLAYER_POWER_VALUE_OFFSET = 0xE9;
constexpr uint32_t KOF98_PLAYER_POWER_STATE_OFFSET = 0xEB;
constexpr uint32_t KOF98_PLAYER_STUN_OFFSET = 0x13E;
constexpr uint32_t KOF98_ROUND_TIME_ADDRESS = 0x10A83B;
constexpr uint32_t KOF98_GAME_PHASE_ADDRESS = 0x10B094;
constexpr uint32_t KOF98_CAMERA_LEFT_ADDRESS = 0x10B0CC;
constexpr uint32_t KOF98_PROJECTILE_LIST_ADDRESS = 0x10BF24;
constexpr uint32_t KOF98_P1_HEALTH_ALTERNATE_ADDRESS = 0x10A946;
constexpr uint32_t KOF98_P1_HEALTH_FALLBACK_ADDRESS = 0x10D946;
constexpr uint32_t KOF98_P2_HEALTH_ALTERNATE_ADDRESS = 0x10AA46;
constexpr uint32_t KOF98_P2_HEALTH_FALLBACK_ADDRESS = 0x10DA46;
constexpr uint32_t KOF98_P1_ADVANCED_POWER_VALUE_ADDRESS = 0x1081E9;
constexpr uint32_t KOF98_P1_ADVANCED_POWER_STOCKS_ADDRESS = 0x1082E2;
constexpr uint32_t KOF98_P2_ADVANCED_POWER_VALUE_ADDRESS = 0x1083E9;
constexpr uint32_t KOF98_P2_ADVANCED_POWER_STOCKS_ADDRESS = 0x1084E2;
constexpr uint32_t KOF98_P1_COMBO_COUNT_ADDRESS = 0x1082B1;
constexpr uint32_t KOF98_P2_COMBO_COUNT_ADDRESS = 0x1084B1;
constexpr int32_t KOF98_NATIVE_SCREEN_WIDTH = 320;
constexpr int32_t KOF98_GROUND_LEVEL = 16;
constexpr int32_t KOF98_HEALTH_CANDIDATE_MAX = 0x78;
constexpr uint16_t KOF98_BIOS_TEST_PATTERN_A = 0x5555;
constexpr uint16_t KOF98_BIOS_TEST_PATTERN_B = 0xAAAA;

class RamView {
public:
    RamView(const uint8_t *ram, size_t size)
        : ram_(ram)
        , size_(size) {
    }

    bool readU8(uint32_t address, uint8_t &value) const {
        if (address >= KOF98_RAM_BASE)
            address -= KOF98_RAM_BASE;

        return readRawU8(address ^ 1, value);
    }

    bool readRawU8(uint32_t address, uint8_t &value) const {
        if (!ram_)
            return false;
        if (address >= KOF98_RAM_BASE)
            address -= KOF98_RAM_BASE;
        if (address >= size_)
            return false;

        value = ram_[address];
        return true;
    }

    bool readS8(uint32_t address, int32_t &value) const {
        uint8_t byte = 0;
        if (!readU8(address, byte))
            return false;

        value = static_cast<int32_t>(static_cast<int8_t>(byte));
        return true;
    }

    bool readS16Be(uint32_t address, int32_t &value) const {
        uint8_t high = 0;
        uint8_t low = 0;
        if (!readU8(address, high) || !readU8(address + 1, low))
            return false;

        const uint16_t word = static_cast<uint16_t>((high << 8) | low);
        value = static_cast<int32_t>(static_cast<int16_t>(word));
        return true;
    }

    bool readU16Be(uint32_t address, uint16_t &value) const {
        uint8_t high = 0;
        uint8_t low = 0;
        if (!readU8(address, high) || !readU8(address + 1, low))
            return false;

        value = static_cast<uint16_t>((high << 8) | low);
        return true;
    }

private:
    const uint8_t *ram_ = nullptr;
    size_t size_ = 0;
};

struct BoxEntry {
    uint32_t offset = 0;
    int32_t type = HitboxUndefined;
    uint8_t active_bit = 0;
};

struct ObjectRef {
    uint32_t base = 0;
    bool projectile = false;
    int32_t owner = 0;
    int32_t pos_x = 0;
    int32_t pos_y = 0;
    int32_t flip_x = 1;
    uint8_t status = 0;
};

constexpr std::array<int32_t, 63> KOF98_BOX_TYPES {{
    HitboxVulnerability, HitboxVulnerability, HitboxVulnerability,
    HitboxVulnerability, HitboxVulnerability, HitboxVulnerability,
    HitboxVulnerability, HitboxVulnerability, HitboxVulnerability,
    HitboxGuard, HitboxGuard,
    HitboxAttack, HitboxAttack, HitboxAttack, HitboxAttack,
    HitboxAttack, HitboxAttack, HitboxAttack, HitboxAttack,
    HitboxAttack, HitboxAttack, HitboxAttack, HitboxAttack,
    HitboxAttack, HitboxAttack, HitboxAttack, HitboxAttack,
    HitboxAttack, HitboxAttack, HitboxAttack, HitboxAttack,
    HitboxAttack, HitboxAttack, HitboxAttack, HitboxAttack,
    HitboxAttack, HitboxAttack, HitboxAttack, HitboxAttack,
    HitboxAttack, HitboxAttack, HitboxAttack, HitboxAttack,
    HitboxAttack, HitboxAttack, HitboxAttack, HitboxAttack,
    HitboxAttack, HitboxAttack, HitboxAttack, HitboxAttack,
    HitboxAttack, HitboxAttack, HitboxAttack, HitboxAttack,
    HitboxGuard, HitboxGuard,
    HitboxProjectileVulnerability, HitboxProjectileVulnerability,
    HitboxProjectileVulnerability, HitboxProjectileVulnerability,
    HitboxProjectileVulnerability, HitboxProjectileVulnerability,
}};

constexpr std::array<BoxEntry, 5> KOF98_BOX_LIST {{
    { 0xA4, HitboxPush, 0 },
    { 0x9F, HitboxUndefined, 3 },
    { 0x9A, HitboxUndefined, 2 },
    { 0x95, HitboxUndefined, 1 },
    { 0x90, HitboxUndefined, 0 },
}};

bool hasBit(uint8_t value, uint8_t bit) {
    return (value & (1 << bit)) != 0;
}

int32_t boxTypeFromId(uint8_t id) {
    if (id == 0 || id > KOF98_BOX_TYPES.size())
        return HitboxUndefined;

    return KOF98_BOX_TYPES[id - 1];
}

bool biosTest(const RamView &ram, uint32_t address) {
    uint16_t value = 0;
    if (!ram.readU16Be(address, value))
        return true;

    return value == KOF98_BIOS_TEST_PATTERN_A ||
           value == KOF98_BIOS_TEST_PATTERN_B ||
           value == static_cast<uint16_t>(address & 0xffff);
}

int32_t readByteOrNegative(const RamView &ram, uint32_t address) {
    uint8_t value = 0;
    if (!ram.readU8(address, value))
        return -1;

    return static_cast<int32_t>(value);
}

int32_t readRawByteOrNegative(const RamView &ram, uint32_t address) {
    uint8_t value = 0;
    if (!ram.readRawU8(address, value))
        return -1;

    return static_cast<int32_t>(value);
}

int32_t readHealthByte(const RamView &ram,
                       uint32_t primary_address,
                       uint32_t alternate_address,
                       uint32_t fallback_address) {
    const std::array<uint32_t, 3> candidates {{
        primary_address,
        alternate_address,
        fallback_address,
    }};

    for (uint32_t address : candidates) {
        const int32_t value = readRawByteOrNegative(ram, address);
        if (value >= 0 && value <= KOF98_HEALTH_CANDIDATE_MAX)
            return value;
    }

    return -1;
}

bool defineBox(const RamView &ram, const ObjectRef &object, const BoxEntry &entry, HitboxRect &box) {
    const uint32_t address = object.base + entry.offset;
    uint8_t id = 0;
    if (!ram.readU8(address + KOF98_HITBOX_ID_OFFSET, id))
        return false;

    int32_t type = entry.type;
    if (type == HitboxUndefined) {
        if (!hasBit(object.status, entry.active_bit))
            return false;

        type = boxTypeFromId(id);
        if (type == HitboxAttack) {
            if (entry.active_bit == 1)
                return false;
            if (object.projectile)
                type = HitboxProjectileAttack;
        }
    } else if (type == HitboxPush) {
        if (id == 0xFF || object.projectile)
            return false;
    }

    uint8_t rad_x = 0;
    uint8_t rad_y = 0;
    int32_t val_x = 0;
    int32_t val_y = 0;
    if (!ram.readU8(address + KOF98_HITBOX_RADIUS_X_OFFSET, rad_x) ||
        !ram.readU8(address + KOF98_HITBOX_RADIUS_Y_OFFSET, rad_y) ||
        !ram.readS8(address + KOF98_HITBOX_X_OFFSET, val_x) ||
        !ram.readS8(address + KOF98_HITBOX_Y_OFFSET, val_y)) {
        return false;
    }

    if (rad_x == 0 && rad_y == 0 && val_x == 0 && val_y == 0)
        return false;

    const int32_t center_x = object.pos_x + val_x * object.flip_x;
    const int32_t center_y = object.pos_y + val_y;
    box.type = type;
    box.owner = object.owner;
    box.left = center_x - rad_x;
    box.top = center_y - rad_y;
    box.width = std::max<int32_t>(1, static_cast<int32_t>(rad_x) * 2);
    box.height = std::max<int32_t>(1, static_cast<int32_t>(rad_y) * 2);
    return true;
}

bool readPlayerPosition(const RamView &ram, uint32_t base, int32_t source_width, Point &position) {
    int32_t screen_left = 0;
    int32_t raw_x = 0;
    int32_t raw_y = 0;
    if (!ram.readS16Be(KOF98_CAMERA_LEFT_ADDRESS, screen_left) ||
        !ram.readS16Be(base + KOF98_PLAYER_POS_X_OFFSET, raw_x) ||
        !ram.readS16Be(base + KOF98_PLAYER_POS_Y_OFFSET, raw_y)) {
        return false;
    }

    if (source_width > 0 && source_width < KOF98_NATIVE_SCREEN_WIDTH)
        screen_left += (KOF98_NATIVE_SCREEN_WIDTH - source_width) / 2;

    position = { raw_x - screen_left, raw_y - KOF98_GROUND_LEVEL };
    return true;
}

} // namespace

GameMemReaderCore::GameMemReaderCore(const uint8_t *ram, size_t ram_size, int32_t source_width)
    : ram_(ram)
    , ram_size_(ram_size)
    , source_width_(source_width) {
}

HitboxOverlay GameMemReaderCore::getHitboxOverlay() const {
    HitboxOverlay result;
    const RamView ram(ram_, ram_size_);

    uint8_t game_phase = 0;
    if (biosTest(ram, KOF98_P1_PLAYER_BASE) ||
        !ram.readU8(KOF98_GAME_PHASE_ADDRESS, game_phase) ||
        game_phase == 0) {
        return result;
    }

    int32_t screen_left = 0;
    if (!ram.readS16Be(KOF98_CAMERA_LEFT_ADDRESS, screen_left))
        return result;

    if (source_width_ > 0 && source_width_ < KOF98_NATIVE_SCREEN_WIDTH)
        screen_left += (KOF98_NATIVE_SCREEN_WIDTH - source_width_) / 2;

    std::vector<ObjectRef> objects {
        { KOF98_P1_PLAYER_BASE, false, 1 },
        { KOF98_P2_PLAYER_BASE, false, 2 },
    };
    objects.reserve(KOF98_PROJECTILE_LIST_SIZE / KOF98_PROJECTILE_LIST_ENTRY_SIZE);

    for (uint32_t list_offset = 0; list_offset < KOF98_PROJECTILE_LIST_SIZE;
         list_offset += KOF98_PROJECTILE_LIST_ENTRY_SIZE) {
        uint16_t object_base_low = 0;
        if (!ram.readU16Be(KOF98_PROJECTILE_LIST_ADDRESS + list_offset, object_base_low) ||
            object_base_low == 0) {
            break;
        }

        const uint32_t object_base = KOF98_RAM_BASE | object_base_low;
        int32_t object_state = 0;
        if (!ram.readS16Be(object_base + KOF98_OBJECT_STATE_OFFSET, object_state) || object_state < 0)
            break;

        const auto duplicate = std::any_of(objects.cbegin(), objects.cend(),
            [object_base_low](const ObjectRef &object) {
                return (object.base & 0xffff) == object_base_low;
            });
        if (duplicate)
            break;

        objects.push_back({ object_base, true, 0 });
    }

    for (ObjectRef object : objects) {
        int32_t raw_x = 0;
        int32_t raw_y = 0;
        uint8_t flip_byte = 0;
        if (!ram.readS16Be(object.base + KOF98_PLAYER_POS_X_OFFSET, raw_x) ||
            !ram.readS16Be(object.base + KOF98_PLAYER_POS_Y_OFFSET, raw_y) ||
            !ram.readU8(object.base + KOF98_PLAYER_STATUS_OFFSET, object.status) ||
            !ram.readU8(object.base + KOF98_PLAYER_FLIP_X_OFFSET, flip_byte)) {
            continue;
        }

        object.pos_x = raw_x - screen_left;
        object.pos_y = raw_y - KOF98_GROUND_LEVEL;
        object.flip_x = (flip_byte & 0x01) != 0 ? -1 : 1;

        if (!object.projectile)
            result.axes.push_back({ object.pos_x, object.pos_y });

        for (const BoxEntry &entry : KOF98_BOX_LIST) {
            HitboxRect box;
            if (defineBox(ram, object, entry, box))
                result.boxes.push_back(box);
        }
    }

    return result;
}

int32_t GameMemReaderCore::readRoundTime() const {
    const RamView ram(ram_, ram_size_);
    return readByteOrNegative(ram, KOF98_ROUND_TIME_ADDRESS);
}

int32_t GameMemReaderCore::readP1Health() const {
    const RamView ram(ram_, ram_size_);
    return readHealthByte(ram,
                          KOF98_P1_PLAYER_BASE + KOF98_PLAYER_HEALTH_OFFSET,
                          KOF98_P1_HEALTH_ALTERNATE_ADDRESS,
                          KOF98_P1_HEALTH_FALLBACK_ADDRESS);
}

int32_t GameMemReaderCore::readP2Health() const {
    const RamView ram(ram_, ram_size_);
    return readHealthByte(ram,
                          KOF98_P2_PLAYER_BASE + KOF98_PLAYER_HEALTH_OFFSET,
                          KOF98_P2_HEALTH_ALTERNATE_ADDRESS,
                          KOF98_P2_HEALTH_FALLBACK_ADDRESS);
}

int32_t GameMemReaderCore::readP1Power() const {
    const RamView ram(ram_, ram_size_);
    return readByteOrNegative(ram, KOF98_P1_PLAYER_BASE + KOF98_PLAYER_POWER_VALUE_OFFSET);
}

int32_t GameMemReaderCore::readP2Power() const {
    const RamView ram(ram_, ram_size_);
    return readByteOrNegative(ram, KOF98_P2_PLAYER_BASE + KOF98_PLAYER_POWER_VALUE_OFFSET);
}

int32_t GameMemReaderCore::readP1PowerState() const {
    const RamView ram(ram_, ram_size_);
    return readByteOrNegative(ram, KOF98_P1_PLAYER_BASE + KOF98_PLAYER_POWER_STATE_OFFSET);
}

int32_t GameMemReaderCore::readP2PowerState() const {
    const RamView ram(ram_, ram_size_);
    return readByteOrNegative(ram, KOF98_P2_PLAYER_BASE + KOF98_PLAYER_POWER_STATE_OFFSET);
}

int32_t GameMemReaderCore::readP1AdvancedPowerValue() const {
    const RamView ram(ram_, ram_size_);
    return readRawByteOrNegative(ram, KOF98_P1_ADVANCED_POWER_VALUE_ADDRESS);
}

int32_t GameMemReaderCore::readP1AdvancedPowerStocks() const {
    const RamView ram(ram_, ram_size_);
    return readRawByteOrNegative(ram, KOF98_P1_ADVANCED_POWER_STOCKS_ADDRESS);
}

int32_t GameMemReaderCore::readP2AdvancedPowerValue() const {
    const RamView ram(ram_, ram_size_);
    return readRawByteOrNegative(ram, KOF98_P2_ADVANCED_POWER_VALUE_ADDRESS);
}

int32_t GameMemReaderCore::readP2AdvancedPowerStocks() const {
    const RamView ram(ram_, ram_size_);
    return readRawByteOrNegative(ram, KOF98_P2_ADVANCED_POWER_STOCKS_ADDRESS);
}

int32_t GameMemReaderCore::readP1Stun() const {
    const RamView ram(ram_, ram_size_);
    return readByteOrNegative(ram, KOF98_P1_PLAYER_BASE + KOF98_PLAYER_STUN_OFFSET);
}

int32_t GameMemReaderCore::readP2Stun() const {
    const RamView ram(ram_, ram_size_);
    return readByteOrNegative(ram, KOF98_P2_PLAYER_BASE + KOF98_PLAYER_STUN_OFFSET);
}

int32_t GameMemReaderCore::readP1ComboCount() const {
    const RamView ram(ram_, ram_size_);
    return readRawByteOrNegative(ram, KOF98_P1_COMBO_COUNT_ADDRESS);
}

int32_t GameMemReaderCore::readP2ComboCount() const {
    const RamView ram(ram_, ram_size_);
    return readRawByteOrNegative(ram, KOF98_P2_COMBO_COUNT_ADDRESS);
}

bool GameMemReaderCore::readP1Position(Point &position) const {
    const RamView ram(ram_, ram_size_);
    return readPlayerPosition(ram, KOF98_P1_PLAYER_BASE, source_width_, position);
}

bool GameMemReaderCore::readP2Position(Point &position) const {
    const RamView ram(ram_, ram_size_);
    return readPlayerPosition(ram, KOF98_P2_PLAYER_BASE, source_width_, position);
}

bool GameMemReaderCore::readP1FacingLeft(bool &facing_left) const {
    const RamView ram(ram_, ram_size_);
    uint8_t flip_byte = 0;
    if (!ram.readU8(KOF98_P1_PLAYER_BASE + KOF98_PLAYER_FLIP_X_OFFSET, flip_byte))
        return false;

    facing_left = (flip_byte & 0x01) == 0;
    return true;
}

bool GameMemReaderCore::p1ReadyForAction() const {
    if (readP1Health() <= 0)
        return false;

    const HitboxOverlay overlay = getHitboxOverlay();
    int32_t largest_area = 0;
    HitboxRect body_box;
    for (const HitboxRect &box : overlay.boxes) {
        if (box.owner != 1 || box.type != HitboxVulnerability)
            continue;

        const int32_t area = box.width * box.height;
        if (area > largest_area) {
            largest_area = area;
            body_box = box;
        }
    }

    if (largest_area <= 0)
        return false;

    return body_box.height >= body_box.width;
}

} // namespace game_memory
