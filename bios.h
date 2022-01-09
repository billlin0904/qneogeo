#pragma once

#include <cstdint>
#include <utility>
#include <array>

#include "memory.h"

enum class Family : uint8_t {
    FrontLoader = 0,
    TopLoader = 1,
    CDZ = 2,
    Invalid = 254,
    Unknown = 255
};

enum class Mod : uint8_t {
    None,
    SMKDan,
    Universe32,
    Universe33
};

using Type = std::pair<Family, Mod>;

struct Pattern {
    uint32_t address;
    const uint8_t* data;
    uint32_t length;
};

struct ReplacePattern {
    uint32_t address;
    const uint8_t* originalData;
    const uint8_t* modifiedData;
    uint32_t length;
};

class Exception : public std::exception {
public:
    Exception(const char *str);
};

class Bios {
public:

    void autoByteSwap();

    Type identify();

    void patch(const Type biosType, bool speedHackEnabled);
private:
    bool isPatternPresent(const Pattern &pattern);

    bool replacePattern(const ReplacePattern *replacePattern);

    std::array<uint16_t, Memory::ROM_SIZE> rom_;
};

void patch(uint8_t* biosData, const Type biosType, bool speedHackEnabled);

std::string description(Type type);



