#include <array>
#include "bios.h"
#include "memory.h"
#include "endian.h"

static constexpr std::array<uint8_t, 4>
VALIDITY_SEARCH_PATTERN_DATA
{
    0x00, 0x10, 0xF3, 0x00
};

static constexpr std::array<uint8_t, 4>
FRONT_LOADER_SEARCH_PATTERN_DATA
{
    0x00, 0xC0, 0xC8, 0x5E
};

static constexpr std::array<uint8_t, 4>
TOP_LOADER_SEARCH_PATTERN_DATA
{
    0x00, 0xC0, 0xC2, 0x22
};

static constexpr std::array<uint8_t, 4>
CDZ_SEARCH_PATTERN_DATA
{
    0x00, 0xC0, 0xA3, 0xE8
};

static constexpr std::array<uint8_t, 4>
SMKDAN_FRONT_SEARCH_PATTERN_DATA
{
    0x00, 0xC2, 0x33, 0x00
};

static constexpr std::array<uint8_t, 4>
SMKDAN_TOP_SEARCH_PATTERN_DATA
{
    0x00, 0xC2, 0x34, 0x00
};

static constexpr std::array<uint8_t, 4>
SMKDAN_CDZ_SEARCH_PATTERN_DATA
{
    0x00, 0xC6, 0x20, 0x00
};

static constexpr std::array<uint8_t, 4>
UNIVERSE32_SEARCH_PATTERN_DATA
{
    0x1C, 0xCA, 0x85, 0x8A
};

static constexpr std::array<uint8_t, 4>
UNIVERSE33_SEARCH_PATTERN_DATA
{
    0xA4, 0x4B, 0x15, 0x2F
};

static constexpr std::array<uint8_t, 2>
CD_REC_SEARCH_PATTERN_DATA_A
{
    0x66, 0x10
};

static constexpr std::array<uint8_t, 2>
CD_REC_SEARCH_PATTERN_DATA_B
{
    0x66, 0x74
};

static constexpr std::array<uint8_t, 2>
CD_REC_SEARCH_PATTERN_DATA_C
{
    0x66, 0x04
};

static constexpr std::array<uint8_t, 6>
SPEEDHACK_SEARCH_PATTERN_DATA_A
{
    0x53, 0x81, 0x67, 0x00, 0xFE, 0xF4
};

static constexpr std::array<uint8_t, 6>
SPEEDHACK_SEARCH_PATTERN_DATA_B
{
    0x53, 0x81, 0x67, 0x00, 0x00, 0x0E
};

static constexpr std::array<uint8_t, 6>
SPEEDHACK_SEARCH_PATTERN_DATA_C
{
    0x53, 0x81, 0x67, 0x00, 0xFE, 0x70
};

static constexpr std::array<uint8_t, 6>
SPEEDHACK_SEARCH_PATTERN_DATA_D
{
    0x53, 0x81, 0x67, 0x00, 0xFF, 0x46
};

static constexpr std::array<uint8_t, 6>
SPEEDHACK_SEARCH_PATTERN_DATA_E
{
    0x53, 0x81, 0x67, 0x00, 0xFE, 0xC4
};

static constexpr std::array<uint8_t, 6>
SPEEDHACK_SEARCH_PATTERN_DATA_F
{
    0x53, 0x81, 0x67, 0x00, 0xFF, 0x2A
};

static constexpr std::array<uint8_t, 6>
SPEEDHACK_SEARCH_PATTERN_DATA_G
{
    0x53, 0x81, 0x67, 0x00, 0xFE, 0xA6
};

static constexpr std::array<uint8_t, 2>
UNIBIOS33_CHECKSUM_SEARCH_PATTERN_DATA
{
    0x67, 0x32
};

static constexpr std::array<uint8_t, 6>
SMKDAN_CHECKSUM_SEARCH_PATTERN_DATA_A
{
    0x22, 0x39, 0x00, 0xC6,  0xFF,  0xF4
};

static constexpr std::array<uint8_t, 6>
SMKDAN_CHECKSUM_SEARCH_PATTERN_DATA_B
{
    0x22, 0x39, 0x00, 0xC2,  0xFF,  0xF4
};

static constexpr std::array<uint8_t, 2>
REPLACE_DATA_NOP
{
    0x4E, 0x71
};

static constexpr std::array<uint8_t, 6>
REPLACE_DATA_SPEEDHACK
{
    0xFA, 0xBE, 0x4E, 0x71, 0x4E, 0x71
};

static constexpr std::array<uint8_t, 2>
REPLACE_DATA_UNIBIOS33_CHECKSUM
{
    0x60, 0x32
};

static constexpr std::array<uint8_t, 6>
REPLACE_DATA_SMKDAN_CHECKSUM
{
    0x22, 0x00, 0x4E, 0x71, 0x4E, 0x71
};

static const Pattern VALIDITY_SEARCH_PATTERN
{
    0xC00000,
    VALIDITY_SEARCH_PATTERN_DATA.data(),
    VALIDITY_SEARCH_PATTERN_DATA.size()
};

static const Pattern FRONT_LOADER_SEARCH_PATTERN
{
    0xC0006C,
    FRONT_LOADER_SEARCH_PATTERN_DATA.data(),
    FRONT_LOADER_SEARCH_PATTERN_DATA.size()
};

static const Pattern TOP_LOADER_SEARCH_PATTERN
{
    0xC0006C,
    TOP_LOADER_SEARCH_PATTERN_DATA.data(),
    TOP_LOADER_SEARCH_PATTERN_DATA.size()
};

static const Pattern CDZ_SEARCH_PATTERN
{
    0xC0006C,
    CDZ_SEARCH_PATTERN_DATA.data(),
    CDZ_SEARCH_PATTERN_DATA.size()
};

static const Pattern SMKDAN_FRONT_SEARCH_PATTERN
{
    0xC00004,
    SMKDAN_FRONT_SEARCH_PATTERN_DATA.data(),
    SMKDAN_FRONT_SEARCH_PATTERN_DATA.size()
};

static const Pattern SMKDAN_TOP_SEARCH_PATTERN
{
    0xC00004,
    SMKDAN_TOP_SEARCH_PATTERN_DATA.data(),
    SMKDAN_TOP_SEARCH_PATTERN_DATA.size()
};

static const Pattern SMKDAN_CDZ_SEARCH_PATTERN
{
    0xC00004,
    SMKDAN_CDZ_SEARCH_PATTERN_DATA.data(),
    SMKDAN_CDZ_SEARCH_PATTERN_DATA.size()
};

static const Pattern UNIVERSE32_SEARCH_PATTERN
{
    0xC00150,
    UNIVERSE32_SEARCH_PATTERN_DATA.data(),
    UNIVERSE32_SEARCH_PATTERN_DATA.size()
};

static const Pattern UNIVERSE33_SEARCH_PATTERN
{
    0xC00150,
    UNIVERSE33_SEARCH_PATTERN_DATA.data(),
    UNIVERSE33_SEARCH_PATTERN_DATA.size()
};

static const std::array<ReplacePattern, 3> CDZ_CD_RECOG_REPLACE
{
    ReplacePattern { 0xC0EB82, CD_REC_SEARCH_PATTERN_DATA_A.data(), REPLACE_DATA_NOP.data(), CD_REC_SEARCH_PATTERN_DATA_A.size() },
    ReplacePattern { 0xC0D280, CD_REC_SEARCH_PATTERN_DATA_B.data(), REPLACE_DATA_NOP.data(), CD_REC_SEARCH_PATTERN_DATA_B.size() },
    ReplacePattern { 0x000000, nullptr, nullptr, 0 }
};

static const std::array<ReplacePattern, 6> CDZ_SPEEDHACK_REPLACE
{
    ReplacePattern { 0xC0E6E0, SPEEDHACK_SEARCH_PATTERN_DATA_A.data(), REPLACE_DATA_SPEEDHACK.data(), SPEEDHACK_SEARCH_PATTERN_DATA_A.size() },
    ReplacePattern { 0xC0E724, SPEEDHACK_SEARCH_PATTERN_DATA_B.data(), REPLACE_DATA_SPEEDHACK.data(), SPEEDHACK_SEARCH_PATTERN_DATA_B.size() },
    ReplacePattern { 0xC0E764, SPEEDHACK_SEARCH_PATTERN_DATA_C.data(), REPLACE_DATA_SPEEDHACK.data(), SPEEDHACK_SEARCH_PATTERN_DATA_C.size() },
    ReplacePattern { 0xC0E836, SPEEDHACK_SEARCH_PATTERN_DATA_B.data(), REPLACE_DATA_SPEEDHACK.data(), SPEEDHACK_SEARCH_PATTERN_DATA_B.size() },
    ReplacePattern { 0xC0E860, SPEEDHACK_SEARCH_PATTERN_DATA_B.data(), REPLACE_DATA_SPEEDHACK.data(), SPEEDHACK_SEARCH_PATTERN_DATA_B.size() },
    ReplacePattern { 0x000000, nullptr, nullptr, 0 }
};

static const std::array<ReplacePattern, 2> CDZ_UNIVERSE33_CHECKSUM_REPLACE
{
    ReplacePattern { 0xC1D3EC, UNIBIOS33_CHECKSUM_SEARCH_PATTERN_DATA.data(), REPLACE_DATA_UNIBIOS33_CHECKSUM.data(), UNIBIOS33_CHECKSUM_SEARCH_PATTERN_DATA.size() },
    ReplacePattern { 0x000000, nullptr, nullptr, 0 }
};

static const std::array<ReplacePattern, 2> CDZ_SMKDAN_CHECKSUM_REPLACE
 {
    ReplacePattern { 0xC62BF4, SMKDAN_CHECKSUM_SEARCH_PATTERN_DATA_A.data(), REPLACE_DATA_SMKDAN_CHECKSUM.data(), SMKDAN_CHECKSUM_SEARCH_PATTERN_DATA_A.size() },
    ReplacePattern { 0x000000, nullptr, nullptr, 0 }
};

static const std::array<ReplacePattern, 2> FRONT_CD_RECOG_REPLACE
{
    ReplacePattern { 0xC10B64, CD_REC_SEARCH_PATTERN_DATA_C.data(), REPLACE_DATA_NOP.data(), CD_REC_SEARCH_PATTERN_DATA_C.size() },
    ReplacePattern { 0x000000, nullptr, nullptr, 0 }
};

static const std::array<ReplacePattern, 5> FRONT_SPEEDHACK_REPLACE
{
    ReplacePattern { 0xC10716, SPEEDHACK_SEARCH_PATTERN_DATA_D.data(), REPLACE_DATA_SPEEDHACK.data(), SPEEDHACK_SEARCH_PATTERN_DATA_D.size() },
    ReplacePattern { 0xC10758, SPEEDHACK_SEARCH_PATTERN_DATA_B.data(), REPLACE_DATA_SPEEDHACK.data(), SPEEDHACK_SEARCH_PATTERN_DATA_B.size() },
    ReplacePattern { 0xC10798, SPEEDHACK_SEARCH_PATTERN_DATA_E.data(), REPLACE_DATA_SPEEDHACK.data(), SPEEDHACK_SEARCH_PATTERN_DATA_E.size() },
    ReplacePattern { 0xC10864, SPEEDHACK_SEARCH_PATTERN_DATA_B.data(), REPLACE_DATA_SPEEDHACK.data(), SPEEDHACK_SEARCH_PATTERN_DATA_B.size() },
    ReplacePattern { 0x000000, nullptr, nullptr, 0 }
};

static const std::array<ReplacePattern, 2> FRONT_SMKDAN_CHECKSUM_REPLACE
{
    ReplacePattern { 0xC23EBE, SMKDAN_CHECKSUM_SEARCH_PATTERN_DATA_B.data(), REPLACE_DATA_SMKDAN_CHECKSUM.data(), SMKDAN_CHECKSUM_SEARCH_PATTERN_DATA_B.size() },
    ReplacePattern { 0x000000, nullptr, nullptr, 0 }
};

static const std::array<ReplacePattern, 2> TOP_CD_RECOG_REPLACE
{
    ReplacePattern { 0xC10436, CD_REC_SEARCH_PATTERN_DATA_C.data(), REPLACE_DATA_NOP.data(), CD_REC_SEARCH_PATTERN_DATA_C.size() },
    ReplacePattern { 0x000000, nullptr, nullptr, 0 }
};

static const std::array<ReplacePattern, 6> TOP_SPEEDHACK_REPLACE
{
    ReplacePattern { 0xC0FFCA, SPEEDHACK_SEARCH_PATTERN_DATA_F.data(), REPLACE_DATA_SPEEDHACK.data(), SPEEDHACK_SEARCH_PATTERN_DATA_F.size() },
    ReplacePattern { 0xC1000E, SPEEDHACK_SEARCH_PATTERN_DATA_B.data(), REPLACE_DATA_SPEEDHACK.data(), SPEEDHACK_SEARCH_PATTERN_DATA_B.size() },
    ReplacePattern { 0xC1004E, SPEEDHACK_SEARCH_PATTERN_DATA_G.data(), REPLACE_DATA_SPEEDHACK.data(), SPEEDHACK_SEARCH_PATTERN_DATA_G.size() },
    ReplacePattern { 0xC10120, SPEEDHACK_SEARCH_PATTERN_DATA_B.data(), REPLACE_DATA_SPEEDHACK.data(), SPEEDHACK_SEARCH_PATTERN_DATA_B.size() },
    ReplacePattern { 0x000000, nullptr, nullptr, 0 }
};

static const std::array<ReplacePattern, 2> TOP_SMKDAN_CHECKSUM_REPLACE
{
    ReplacePattern { 0xC23FBE, SMKDAN_CHECKSUM_SEARCH_PATTERN_DATA_B.data(), REPLACE_DATA_SMKDAN_CHECKSUM.data(), SMKDAN_CHECKSUM_SEARCH_PATTERN_DATA_B.size() },
    ReplacePattern { 0x000000, nullptr, nullptr, 0 }
};

bool Bios::isPatternPresent(const Pattern &pattern) {
    return (memcmp(pattern.data, rom_.data() + pattern.address - 0xc00000, pattern.length) == 0);
}

void Bios::autoByteSwap() {
    if (rom_[0] != QNEOGEO_LITTLE_ENDIAN_WORD(0x0010)) {
        return;
    }
    std::for_each(rom_.begin(), rom_.end(), [](auto& data) {
        data = QNEOGEO_BYTE_SWAP_16(data);
    });
}

Type Bios::identify() {
    auto family = Family::Invalid;
    auto mod = Mod::None;

    if (isPatternPresent(VALIDITY_SEARCH_PATTERN)) {
        family = Family::Unknown;
    }

    if (isPatternPresent(FRONT_LOADER_SEARCH_PATTERN)) {
        family = Family::FrontLoader;

        if (isPatternPresent(SMKDAN_FRONT_SEARCH_PATTERN))
            mod = Mod::SMKDan;
    } else if (isPatternPresent(TOP_LOADER_SEARCH_PATTERN)) {
        family = Family::TopLoader;

        if (isPatternPresent(SMKDAN_TOP_SEARCH_PATTERN))
            mod = Mod::SMKDan;
    } else if (isPatternPresent(CDZ_SEARCH_PATTERN)) {
        family = Family::CDZ;

        if (isPatternPresent(SMKDAN_CDZ_SEARCH_PATTERN))
            mod = Mod::SMKDan;
        else if (isPatternPresent(UNIVERSE32_SEARCH_PATTERN))
            mod = Mod::Universe32;
        else if (isPatternPresent(UNIVERSE33_SEARCH_PATTERN))
            mod = Mod::Universe33;
    }

    return std::make_pair(family, mod);
}

bool Bios::replacePattern(const ReplacePattern *replacePattern) {
    const auto* ptr = replacePattern;

    while(ptr->address) {
        if (memcmp(ptr->originalData, rom_.data() + ptr->address - 0xC00000, ptr->length) != 0) {
            return false;
        }
        ++ptr;
    }

    ptr = replacePattern;

    while(ptr->address) {
        memcpy(rom_.data() + ptr->address - 0xC00000, ptr->modifiedData, ptr->length);
        ++ptr;
    }
    return true;
}

void Bios::patch(const Type biosType, bool speedHackEnabled) {
    if (biosType.first == Family::CDZ) {
        if (!replacePattern(CDZ_CD_RECOG_REPLACE.data())) {
            throw Exception("CD recognition patch failed.");
        }

        if (speedHackEnabled) {
            if (!replacePattern(CDZ_SPEEDHACK_REPLACE.data())) {
                throw Exception("CD recognition patch failed.");
            }
        }

        if (biosType.second == Mod::SMKDan) {
            if (!replacePattern(CDZ_SMKDAN_CHECKSUM_REPLACE.data())) {
                throw Exception("CD recognition patch failed.");
            }
        }

        if (biosType.second == Mod::Universe33) {
            if (!replacePattern(CDZ_SMKDAN_CHECKSUM_REPLACE.data())) {
                throw Exception("CD recognition patch failed.");
            }
        }
    }
    else if (biosType.first == Family::FrontLoader) {

    }
    else if (biosType.first == Family::TopLoader) {

    }
}


