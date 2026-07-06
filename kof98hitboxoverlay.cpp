#include "kof98hitboxoverlay.h"

#include <QColor>
#include <QPointF>
#include <QRectF>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <utility>

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
constexpr int KOF98_NATIVE_SCREEN_WIDTH = 320;
constexpr int KOF98_HEALTH_CANDIDATE_MAX = 0x78;
constexpr uint16_t KOF98_BIOS_TEST_PATTERN_A = 0x5555;
constexpr uint16_t KOF98_BIOS_TEST_PATTERN_B = 0xAAAA;
constexpr uint32_t KOF98_PLAYER_HEALTH_OFFSET = 0x138;
constexpr uint32_t KOF98_PLAYER_POWER_VALUE_OFFSET = 0xE9;
constexpr uint32_t KOF98_PLAYER_POWER_STATE_OFFSET = 0xEB;
constexpr uint32_t KOF98_PLAYER_STUN_OFFSET = 0x13E;
constexpr uint32_t KOF98_ROUND_TIME_ADDRESS = 0x10A83B;
constexpr uint32_t KOF98_P1_HEALTH_ALTERNATE_ADDRESS = 0x10A946;
constexpr uint32_t KOF98_P1_HEALTH_FALLBACK_ADDRESS = 0x10D946;
constexpr uint32_t KOF98_P2_HEALTH_ALTERNATE_ADDRESS = 0x10AA46;
constexpr uint32_t KOF98_P2_HEALTH_FALLBACK_ADDRESS = 0x10DA46;

enum class Kof98HitboxType {
    Undefined,
    Attack,                  // Lua: a = "attack"
    Vulnerability,           // Lua: v = "vulnerability"
    ProjectileVulnerability, // Lua: p = "proj. vulnerability"
    ProjectileAttack,        // Lua: "proj. attack"
    Push,                    // Lua: "push"
    Guard,                   // Lua: g = "guard"
    Throw,                   // Lua: t = "throw"
    AxisThrow,               // Lua: "axis throw"
    Throwable,               // Lua: "throwable"
};

struct Kof98Offset {
    uint32_t player_space = 0x200;
    uint32_t pos_x = 0x18;
    uint32_t pos_y = 0x26;
    uint32_t flip_x = 0x31;
    uint32_t status = 0x7C;
};

struct Kof98Address {
    uint32_t ground_level = 16;
    uint32_t player = 0x108100;
    uint32_t game_phase = 0x10B094;

    constexpr uint32_t screenLeft() const {
        return game_phase + 0x038;
    }

    constexpr uint32_t objectPointerList() const {
        return game_phase + 0xE90;
    }
};

struct Kof98BoxEntry {
    uint32_t offset = 0;
    Kof98HitboxType type = Kof98HitboxType::Undefined;
    uint8_t active_bit = 0;
    bool clear = false;
};

struct Kof98GameProfile {
    Kof98Address address;
    Kof98Offset offset;

    std::array<Kof98HitboxType, 63> box_types {{
        Kof98HitboxType::Vulnerability, Kof98HitboxType::Vulnerability, Kof98HitboxType::Vulnerability,
        Kof98HitboxType::Vulnerability, Kof98HitboxType::Vulnerability, Kof98HitboxType::Vulnerability,
        Kof98HitboxType::Vulnerability, Kof98HitboxType::Vulnerability, Kof98HitboxType::Vulnerability,
        Kof98HitboxType::Guard, Kof98HitboxType::Guard,
        Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack,
        Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack,
        Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack,
        Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack,
        Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack,
        Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack,
        Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack,
        Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack,
        Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack,
        Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack,
        Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack, Kof98HitboxType::Attack,
        Kof98HitboxType::Guard, Kof98HitboxType::Guard,
        Kof98HitboxType::ProjectileVulnerability, Kof98HitboxType::ProjectileVulnerability,
        Kof98HitboxType::ProjectileVulnerability, Kof98HitboxType::ProjectileVulnerability,
        Kof98HitboxType::ProjectileVulnerability, Kof98HitboxType::ProjectileVulnerability,
    }};

    std::array<Kof98BoxEntry, 5> box_list {{
        { 0xA4, Kof98HitboxType::Push, 0, false },
        { 0x9F, Kof98HitboxType::Undefined, 3, false },
        { 0x9A, Kof98HitboxType::Undefined, 2, false },
        { 0x95, Kof98HitboxType::Undefined, 1, false },
        { 0x90, Kof98HitboxType::Undefined, 0, false },
    }};

    Kof98HitboxType boxTypeFromId(uint8_t id) const {
        if (id == 0 || id > box_types.size())
            return Kof98HitboxType::Undefined;

        return box_types[id - 1];
    }
};

struct Kof98Object {
    uint32_t base = 0;
    bool projectile = false;
    int pos_x = 0;
    int pos_y = 0;
    int flip_x = 1;
    uint8_t status = 0;
};

struct Kof98Box {
    uint32_t address = 0;
    Kof98HitboxType type = Kof98HitboxType::Undefined;
    bool clear = false;
    uint8_t id = 0;
    int val_x = 0;
    int val_y = 0;
    uint8_t rad_x = 0;
    uint8_t rad_y = 0;
    int left = 0;
    int right = 0;
    int top = 0;
    int bottom = 0;
};

struct Kof98HitboxPalette {
    QColor fill;
    QColor outline;
};

class Kof98RamView {
public:
    explicit Kof98RamView(const QByteArray &ram)
        : ram_(ram) {
    }

    bool readU8(uint32_t address, uint8_t &value) const {
        if (address >= KOF98_RAM_BASE)
            address -= KOF98_RAM_BASE;

        return readRawU8(address ^ 1, value);
    }

    bool readRawByte(uint32_t address, uint8_t &value) const {
        return readRawU8(address, value);
    }

    bool readS8(uint32_t address, int &value) const {
        uint8_t byte = 0;
        if (!readU8(address, byte))
            return false;

        value = static_cast<int>(static_cast<int8_t>(byte));
        return true;
    }

    bool readS16Be(uint32_t address, int &value) const {
        uint8_t high = 0;
        uint8_t low = 0;
        if (!readU8(address, high) || !readU8(address + 1, low))
            return false;

        const uint16_t word = static_cast<uint16_t>((high << 8) | low);
        value = static_cast<int>(static_cast<int16_t>(word));
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
    bool readRawU8(uint32_t address, uint8_t &value) const {
        if (address >= KOF98_RAM_BASE)
            address -= KOF98_RAM_BASE;
        if (address >= static_cast<uint32_t>(ram_.size()))
            return false;

        value = static_cast<uint8_t>(ram_[static_cast<int>(address)]);
        return true;
    }

    const QByteArray &ram_;
};

constexpr Kof98GameProfile KOF98_GAME {};
constexpr uint32_t KOF98_P1_PLAYER_BASE = KOF98_GAME.address.player;
constexpr uint32_t KOF98_P2_PLAYER_BASE = KOF98_GAME.address.player + KOF98_GAME.offset.player_space;
constexpr uint32_t KOF98_GAME_PHASE_ADDRESS = KOF98_GAME.address.game_phase;
constexpr uint32_t KOF98_CAMERA_LEFT_ADDRESS = KOF98_GAME.address.screenLeft();
constexpr uint32_t KOF98_PROJECTILE_LIST_ADDRESS = KOF98_GAME.address.objectPointerList();

bool hasBit(uint8_t value, uint8_t bit) {
    return (value & (1 << bit)) != 0;
}

bool biosTest(const Kof98RamView &ram, uint32_t address) {
    uint16_t value = 0;
    if (!ram.readU16Be(address, value))
        return true;

    return value == KOF98_BIOS_TEST_PATTERN_A ||
           value == KOF98_BIOS_TEST_PATTERN_B ||
           value == static_cast<uint16_t>(address & 0xffff);
}

Kof98HitboxPalette paletteForKof98HitboxType(Kof98HitboxType type) {
    switch (type) {
    case Kof98HitboxType::Attack:
        return { QColor(255, 0, 0, 64), QColor(255, 0, 0, 255) };
    case Kof98HitboxType::ProjectileAttack:
        return { QColor(255, 128, 0, 64), QColor(255, 128, 0, 255) };
    case Kof98HitboxType::Vulnerability:
        return { QColor(0, 64, 255, 48), QColor(0, 64, 255, 255) };
    case Kof98HitboxType::ProjectileVulnerability:
        return { QColor(0, 220, 255, 48), QColor(0, 220, 255, 255) };
    case Kof98HitboxType::Push:
        return { QColor(0, 255, 0, 40), QColor(0, 255, 0, 255) };
    case Kof98HitboxType::Guard:
        return { QColor(255, 255, 0, 48), QColor(255, 255, 0, 255) };
    case Kof98HitboxType::Throw:
    case Kof98HitboxType::AxisThrow:
        return { QColor(255, 0, 255, 56), QColor(255, 0, 255, 255) };
    case Kof98HitboxType::Throwable:
        return { QColor(255, 255, 255, 40), QColor(255, 255, 255, 255) };
    case Kof98HitboxType::Undefined:
        return { QColor(160, 160, 160, 40), QColor(160, 160, 160, 255) };
    }

    return { QColor(160, 160, 160, 40), QColor(160, 160, 160, 255) };
}

std::optional<Kof98Box> defineBox(const Kof98RamView &ram, const Kof98Object &object, const Kof98BoxEntry &box_entry) {
    Kof98Box box;
    box.address = object.base + box_entry.offset;
    box.type = box_entry.type;
    box.clear = box_entry.clear;

    if (!ram.readU8(box.address + KOF98_HITBOX_ID_OFFSET, box.id))
        return std::nullopt;

    if (box.type == Kof98HitboxType::Undefined) {
        if (!hasBit(object.status, box_entry.active_bit))
            return std::nullopt;

        box.type = KOF98_GAME.boxTypeFromId(box.id);
        if (box.type == Kof98HitboxType::Attack) {
            // KOF hitbox Lua intentionally skips active_bit 1 attack ghost boxes.
            if (box_entry.active_bit == 1)
                return std::nullopt;
            if (object.projectile)
                box.type = Kof98HitboxType::ProjectileAttack;
        }
    } else if (box.type == Kof98HitboxType::Push) {
        if (box.id == 0xFF || object.projectile)
            return std::nullopt;
    }

    if (!ram.readU8(box.address + KOF98_HITBOX_RADIUS_X_OFFSET, box.rad_x) ||
        !ram.readU8(box.address + KOF98_HITBOX_RADIUS_Y_OFFSET, box.rad_y) ||
        !ram.readS8(box.address + KOF98_HITBOX_X_OFFSET, box.val_x) ||
        !ram.readS8(box.address + KOF98_HITBOX_Y_OFFSET, box.val_y)) {
        return std::nullopt;
    }

    if (box.rad_x == 0 && box.rad_y == 0 && box.val_x == 0 && box.val_y == 0)
        return std::nullopt;

    box.val_x = object.pos_x + box.val_x * object.flip_x;
    box.val_y = object.pos_y + box.val_y;
    box.left = box.val_x - box.rad_x;
    box.right = box.val_x + box.rad_x - 1;
    box.top = box.val_y - box.rad_y;
    box.bottom = box.val_y + box.rad_y - 1;
    return box;
}

int readByteOrNegative(const Kof98RamView &ram, uint32_t address) {
    uint8_t value = 0;
    if (!ram.readU8(address, value))
        return -1;

    return static_cast<int>(value);
}

int readRawByteOrNegative(const Kof98RamView &ram, uint32_t address) {
    uint8_t value = 0;
    if (!ram.readRawByte(address, value))
        return -1;

    return static_cast<int>(value);
}

int readHealthByte(const Kof98RamView &ram, uint32_t primary_address, uint32_t alternate_address, uint32_t fallback_address) {
    const std::array<uint32_t, 3> candidates {{
        primary_address,
        alternate_address,
        fallback_address,
    }};

    for (uint32_t address : candidates) {
        const int value = readRawByteOrNegative(ram, address);
        if (value >= 0 && value <= KOF98_HEALTH_CANDIDATE_MAX)
            return value;
    }

    return -1;
}

bool readPlayerPosition(const Kof98RamView &ram, uint32_t base, QSize source_size, QPoint &position) {
    int screen_left = 0;
    int raw_x = 0;
    int raw_y = 0;
    if (!ram.readS16Be(KOF98_CAMERA_LEFT_ADDRESS, screen_left) ||
        !ram.readS16Be(base + KOF98_GAME.offset.pos_x, raw_x) ||
        !ram.readS16Be(base + KOF98_GAME.offset.pos_y, raw_y)) {
        return false;
    }

    if (source_size.width() > 0 && source_size.width() < KOF98_NATIVE_SCREEN_WIDTH)
        screen_left += (KOF98_NATIVE_SCREEN_WIDTH - source_size.width()) / 2;

    position = QPoint(raw_x - screen_left, raw_y - static_cast<int>(KOF98_GAME.address.ground_level));
    return true;
}
} // namespace

KofGameMemReader::KofGameMemReader(QByteArray ram, QSize sourceSize)
    : ram_(std::move(ram))
    , source_size_(sourceSize) {
}

HitboxOverlay KofGameMemReader::getHitboxOverlay() const {
    HitboxOverlay result;
    const Kof98RamView ram(ram_);

    uint8_t game_phase = 0;
    if (biosTest(ram, KOF98_P1_PLAYER_BASE) ||
        !ram.readU8(KOF98_GAME_PHASE_ADDRESS, game_phase) ||
        game_phase == 0) {
        return result;
    }

    int screen_left = 0;
    if (!ram.readS16Be(KOF98_CAMERA_LEFT_ADDRESS, screen_left))
        return result;

    if (source_size_.width() > 0 && source_size_.width() < KOF98_NATIVE_SCREEN_WIDTH)
        screen_left += (KOF98_NATIVE_SCREEN_WIDTH - source_size_.width()) / 2;

    auto appendObjectOverlay = [&](Kof98Object object) {
        int raw_x = 0;
        int raw_y = 0;
        uint8_t flip_byte = 0;
        if (!ram.readS16Be(object.base + KOF98_GAME.offset.pos_x, raw_x) ||
            !ram.readS16Be(object.base + KOF98_GAME.offset.pos_y, raw_y) ||
            !ram.readU8(object.base + KOF98_GAME.offset.status, object.status) ||
            !ram.readU8(object.base + KOF98_GAME.offset.flip_x, flip_byte)) {
            return;
        }

        object.pos_x = raw_x - screen_left;
        object.pos_y = raw_y - static_cast<int>(KOF98_GAME.address.ground_level);
        object.flip_x = (flip_byte & 0x01) != 0 ? -1 : 1;

        if (!object.projectile)
            result.axes.push_back({ QPointF(object.pos_x, object.pos_y), QColor(255, 255, 255, 255) });

        for (const Kof98BoxEntry &entry : KOF98_GAME.box_list) {
            const std::optional<Kof98Box> maybe_box = defineBox(ram, object, entry);
            if (!maybe_box)
                continue;

            const Kof98Box &box = *maybe_box;
            const int width = std::max(1, box.right - box.left + 1);
            const int height = std::max(1, box.bottom - box.top + 1);
            const Kof98HitboxPalette palette = paletteForKof98HitboxType(box.type);

            result.boxes.push_back({
                QRectF(box.left, box.top, width, height),
                palette.fill,
                palette.outline,
            });
        }
    };

    std::vector<Kof98Object> objects {
        { KOF98_P1_PLAYER_BASE, false },
        { KOF98_P2_PLAYER_BASE, false },
    };

    objects.reserve(KOF98_PROJECTILE_LIST_SIZE / KOF98_PROJECTILE_LIST_ENTRY_SIZE);

    for (uint32_t list_offset = 0; list_offset < KOF98_PROJECTILE_LIST_SIZE;
         list_offset += KOF98_PROJECTILE_LIST_ENTRY_SIZE) {
        uint16_t object_base_low = 0;
        if (!ram.readU16Be(KOF98_PROJECTILE_LIST_ADDRESS + list_offset, object_base_low) || object_base_low == 0)
            break;

        const uint32_t object_base = KOF98_RAM_BASE | object_base_low;
        int object_state = 0;
        if (!ram.readS16Be(object_base + KOF98_OBJECT_STATE_OFFSET, object_state) || object_state < 0)
            break;

        const auto duplicate = std::any_of(objects.cbegin(), objects.cend(),
            [object_base_low](const Kof98Object &object) {
            return (object.base & 0xffff) == object_base_low;
        });
        if (duplicate)
            break;

        objects.push_back({ object_base, true });
    }

    for (const Kof98Object &object : objects)
        appendObjectOverlay(object);

    return result;
}

int KofGameMemReader::readRoundTime() const {
    const Kof98RamView ram(ram_);
    return readByteOrNegative(ram, KOF98_ROUND_TIME_ADDRESS);
}

int KofGameMemReader::readP1Health() const {
    const Kof98RamView ram(ram_);
    return readHealthByte(ram,
                          KOF98_P1_PLAYER_BASE + KOF98_PLAYER_HEALTH_OFFSET,
                          KOF98_P1_HEALTH_ALTERNATE_ADDRESS,
                          KOF98_P1_HEALTH_FALLBACK_ADDRESS);
}

int KofGameMemReader::readP2Health() const {
    const Kof98RamView ram(ram_);
    return readHealthByte(ram,
                          KOF98_P2_PLAYER_BASE + KOF98_PLAYER_HEALTH_OFFSET,
                          KOF98_P2_HEALTH_ALTERNATE_ADDRESS,
                          KOF98_P2_HEALTH_FALLBACK_ADDRESS);
}

int KofGameMemReader::readP1Power() const {
    const Kof98RamView ram(ram_);
    return readByteOrNegative(ram, KOF98_P1_PLAYER_BASE + KOF98_PLAYER_POWER_VALUE_OFFSET);
}

int KofGameMemReader::readP2Power() const {
    const Kof98RamView ram(ram_);
    return readByteOrNegative(ram, KOF98_P2_PLAYER_BASE + KOF98_PLAYER_POWER_VALUE_OFFSET);
}

int KofGameMemReader::readP1PowerState() const {
    const Kof98RamView ram(ram_);
    return readByteOrNegative(ram, KOF98_P1_PLAYER_BASE + KOF98_PLAYER_POWER_STATE_OFFSET);
}

int KofGameMemReader::readP2PowerState() const {
    const Kof98RamView ram(ram_);
    return readByteOrNegative(ram, KOF98_P2_PLAYER_BASE + KOF98_PLAYER_POWER_STATE_OFFSET);
}

int KofGameMemReader::readP1Stun() const {
    const Kof98RamView ram(ram_);
    return readByteOrNegative(ram, KOF98_P1_PLAYER_BASE + KOF98_PLAYER_STUN_OFFSET);
}

int KofGameMemReader::readP2Stun() const {
    const Kof98RamView ram(ram_);
    return readByteOrNegative(ram, KOF98_P2_PLAYER_BASE + KOF98_PLAYER_STUN_OFFSET);
}

bool KofGameMemReader::readP1Position(QPoint &position) const {
    const Kof98RamView ram(ram_);
    return readPlayerPosition(ram, KOF98_P1_PLAYER_BASE, source_size_, position);
}

bool KofGameMemReader::readP2Position(QPoint &position) const {
    const Kof98RamView ram(ram_);
    return readPlayerPosition(ram, KOF98_P2_PLAYER_BASE, source_size_, position);
}
