// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
#ifndef DWT_COMMON_H_
#define DWT_COMMON_H_

const int APRON = 4;
const int APRON_HALF = APRON / 2;
const int BLOCK_SIZE = 32;
const int BLOCK_SIZE_HALF = BLOCK_SIZE >> 1;

#if !FP16 && PRECISION == 0
#undef PRECISION
#define PRECISION 1
#endif

#if PRECISION == 2
#define FLOAT float
#define VEC2 vec2
#define VEC4 vec4
#define SHARED_VEC2 vec2
#elif PRECISION == 1
#define FLOAT float
#define VEC2 vec2
#define VEC4 vec4
#if FP16
#define SHARED_VEC2 f16vec2
#else
#define SHARED_VEC2 uint
#endif
#else
#define FLOAT float16_t
#define VEC2 f16vec2
#define VEC4 f16vec4
#define SHARED_VEC2 f16vec2
#endif

const FLOAT ALPHA = FLOAT(-1.586134342059924);
const FLOAT BETA = FLOAT(-0.052980118572961);
const FLOAT GAMMA = FLOAT(0.882911075530934);
const FLOAT DELTA = FLOAT(0.443506852043971);
const FLOAT K = FLOAT(1.230174104914001);
const FLOAT inv_K = FLOAT(1.0 / 1.230174104914001);

// Stride 42 with bit-0 XOR from bit 2 of y avoids 32-bank conflicts across
// horizontal/vertical 8x2 loads and transpose stores on 32-bank GPUs.
// In PRECISION == 2, split FP32 arrays keep 32-bit shared accesses conflict-free
// (vec2 LDS.64 forces a 2-way conflict on x-strides of 16) within 6720 bytes.
uint swizzle_shared_x(uint y, uint x)
{
	return x ^ ((y >> 2u) & 1u);
}

#if PRECISION == 2
shared FLOAT shared_block_x[(BLOCK_SIZE + 2 * APRON) / 2][(BLOCK_SIZE + 2 * APRON) + 2];
shared FLOAT shared_block_y[(BLOCK_SIZE + 2 * APRON) / 2][(BLOCK_SIZE + 2 * APRON) + 2];
VEC2 load_shared(uint y, uint x)
{
	uint sx = swizzle_shared_x(y, x);
	return VEC2(shared_block_x[y][sx], shared_block_y[y][sx]);
}
void store_shared(uint y, uint x, VEC2 v)
{
	uint sx = swizzle_shared_x(y, x);
	shared_block_x[y][sx] = v.x;
	shared_block_y[y][sx] = v.y;
}
#else
shared SHARED_VEC2 shared_block[(BLOCK_SIZE + 2 * APRON) / 2][(BLOCK_SIZE + 2 * APRON) + 2];
#if !FP16 && PRECISION == 1
VEC2 load_shared(uint y, uint x) { return unpackHalf2x16(shared_block[y][swizzle_shared_x(y, x)]); }
void store_shared(uint y, uint x, VEC2 v) { shared_block[y][swizzle_shared_x(y, x)] = packHalf2x16(v); }
#else
VEC2 load_shared(uint y, uint x) { return VEC2(shared_block[y][swizzle_shared_x(y, x)]); }
void store_shared(uint y, uint x, VEC2 v) { shared_block[y][swizzle_shared_x(y, x)] = SHARED_VEC2(v); }
#endif
#endif

bvec2 band(bvec2 a, bvec2 b)
{
	return bvec2(a.x && b.x, a.y && b.y);
}

#include "dwt_swizzle.h"

#endif
