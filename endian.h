#pragma once

#include <cstdint>

#if defined (__BYTE_ORDER__) && defined (__ORDER_BIG_ENDIAN__) &&  __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define Q_NEOGEO_BIG_ENDIAN
#else
#define Q_NEOGEO_LITTLE_ENDIAN
#endif

#if defined(__clang__)
#define QNEOGEO_BYTE_SWAP_16(x) __builtin_bswap16(x)
#define QNEOGEO_BYTE_SWAP_32(x) __builtin_bswap32(x)
#elif defined(_MSC_VER)
#include <cstdlib>
#define QNEOGEO_BYTE_SWAP_16(x) _byteswap_ushort(x)
#define QNEOGEO_BYTE_SWAP_32(x) _byteswap_ulong(x)
#endif

#ifdef Q_NEOGEO_BIG_ENDIAN
#define QNEOGEO_BIG_ENDIAN_WORD(x) (x)
#define QNEOGEO_BIG_ENDIAN_DWORD(x) (x)
#define QNEOGEO_LITTLE_ENDIAN_WORD(x) BYTE_SWAP_16(x)
#define QNEOGEO_LITTLE_ENDIAN_DWORD(x) BYTE_SWAP_32(x)
#else
#define QNEOGEO_BIG_ENDIAN_WORD(x) BYTE_SWAP_16(x)
#define QNEOGEO_BIG_ENDIAN_DWORD(x) BYTE_SWAP_32(x)
#define QNEOGEO_LITTLE_ENDIAN_WORD(x) (x)
#define QNEOGEO_LITTLE_ENDIAN_DWORD(x) (x)
#endif
