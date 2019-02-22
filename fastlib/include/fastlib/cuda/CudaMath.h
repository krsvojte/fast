/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 *  This file implements common mathematical operations on vector types
 *  (float3, float4 etc.) since these are not provided as standard by CUDA.
 *
 *  The syntax is modeled on the Cg standard library.
 *
 *  This is part of the Helper library includes
 *
 *    Thanks to Linh Hah for additions and fixes.
 */

#ifndef HELPER_MATH_H
#define HELPER_MATH_H

#include "cuda_runtime.h"

typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef __CUDACC__
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// host implementations of CUDA functions
////////////////////////////////////////////////////////////////////////////////

inline float fminf(float a, float b)
{
    return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
    return a > b ? a : b;
}

inline int max(int a, int b)
{
    return a > b ? a : b;
}

inline int min(int a, int b)
{
    return a < b ? a : b;
}

inline float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 make_float2(float s)
{
    return make_float2(s, s);
}
inline __host__ __device__ float2 make_float2(float3 a)
{
    return make_float2(a.x, a.y);
}
inline __host__ __device__ float2 make_float2(int2 a)
{
    return make_float2(float(a.x), float(a.y));
}
inline __host__ __device__ float2 make_float2(uint2 a)
{
    return make_float2(float(a.x), float(a.y));
}

inline __host__ __device__ int2 make_int2(int s)
{
    return make_int2(s, s);
}
inline __host__ __device__ int2 make_int2(int3 a)
{
    return make_int2(a.x, a.y);
}
inline __host__ __device__ int2 make_int2(uint2 a)
{
    return make_int2(int(a.x), int(a.y));
}
inline __host__ __device__ int2 make_int2(float2 a)
{
    return make_int2(int(a.x), int(a.y));
}

inline __host__ __device__ uint2 make_uint2(uint s)
{
    return make_uint2(s, s);
}
inline __host__ __device__ uint2 make_uint2(uint3 a)
{
    return make_uint2(a.x, a.y);
}
inline __host__ __device__ uint2 make_uint2(int2 a)
{
    return make_uint2(uint(a.x), uint(a.y));
}

inline __host__ __device__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}
inline __host__ __device__ float3 make_float3(float2 a)
{
    return make_float3(a.x, a.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(float2 a, float s)
{
    return make_float3(a.x, a.y, s);
}
inline __host__ __device__ float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z);
}
inline __host__ __device__ float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}
inline __host__ __device__ float3 make_float3(uint3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

inline __host__ __device__ int3 make_int3(int s)
{
    return make_int3(s, s, s);
}
inline __host__ __device__ int3 make_int3(int2 a)
{
    return make_int3(a.x, a.y, 0);
}
inline __host__ __device__ int3 make_int3(int2 a, int s)
{
    return make_int3(a.x, a.y, s);
}
inline __host__ __device__ int3 make_int3(uint3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}
inline __host__ __device__ int3 make_int3(float3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}

inline __host__ __device__ uint3 make_uint3(uint s)
{
    return make_uint3(s, s, s);
}
inline __host__ __device__ uint3 make_uint3(uint2 a)
{
    return make_uint3(a.x, a.y, 0);
}
inline __host__ __device__ uint3 make_uint3(uint2 a, uint s)
{
    return make_uint3(a.x, a.y, s);
}
inline __host__ __device__ uint3 make_uint3(uint4 a)
{
    return make_uint3(a.x, a.y, a.z);
}
inline __host__ __device__ uint3 make_uint3(int3 a)
{
    return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

inline __host__ __device__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}
inline __host__ __device__ float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}
inline __host__ __device__ float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}
inline __host__ __device__ float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
inline __host__ __device__ float4 make_float4(uint4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline __host__ __device__ int4 make_int4(int s)
{
    return make_int4(s, s, s, s);
}
inline __host__ __device__ int4 make_int4(int3 a)
{
    return make_int4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ int4 make_int4(int3 a, int w)
{
    return make_int4(a.x, a.y, a.z, w);
}
inline __host__ __device__ int4 make_int4(uint4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}
inline __host__ __device__ int4 make_int4(float4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}


inline __host__ __device__ uint4 make_uint4(uint s)
{
    return make_uint4(s, s, s, s);
}
inline __host__ __device__ uint4 make_uint4(uint3 a)
{
    return make_uint4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ uint4 make_uint4(uint3 a, uint w)
{
    return make_uint4(a.x, a.y, a.z, w);
}
inline __host__ __device__ uint4 make_uint4(int4 a)
{
    return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w));
}

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 &a)
{
    return make_float2(-a.x, -a.y);
}
inline __host__ __device__ int2 operator-(int2 &a)
{
    return make_int2(-a.x, -a.y);
}
inline __host__ __device__ float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ int3 operator-(int3 &a)
{
    return make_int3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ float4 operator-(float4 &a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}
inline __host__ __device__ int4 operator-(int4 &a)
{
    return make_int4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ float2 operator+(float2 a, float b)
{
    return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ float2 operator+(float b, float2 a)
{
    return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(float2 &a, float b)
{
    a.x += b;
    a.y += b;
}

inline __host__ __device__ int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(int2 &a, int2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ int2 operator+(int2 a, int b)
{
    return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ int2 operator+(int b, int2 a)
{
    return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(int2 &a, int b)
{
    a.x += b;
    a.y += b;
}

inline __host__ __device__ uint2 operator+(uint2 a, uint2 b)
{
    return make_uint2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(uint2 &a, uint2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ uint2 operator+(uint2 a, uint b)
{
    return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ uint2 operator+(uint b, uint2 a)
{
    return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(uint2 &a, uint b)
{
    a.x += b;
    a.y += b;
}


inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(int3 &a, int3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ int3 operator+(int3 a, int b)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(int3 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ uint3 operator+(uint3 a, uint3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(uint3 &a, uint3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ uint3 operator+(uint3 a, uint b)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(uint3 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ int3 operator+(int b, int3 a)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ uint3 operator+(uint b, uint3 a)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ float3 operator+(float b, float3 a)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ float4 operator+(float4 a, float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ float4 operator+(float b, float4 a)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(float4 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ int4 operator+(int4 a, int4 b)
{
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(int4 &a, int4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ int4 operator+(int4 a, int b)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ int4 operator+(int b, int4 a)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ void operator+=(int4 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ uint4 operator+(uint4 a, uint4 b)
{
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(uint4 &a, uint4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ uint4 operator+(uint4 a, uint b)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ uint4 operator+(uint b, uint4 a)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ void operator+=(uint4 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ float2 operator-(float2 a, float b)
{
    return make_float2(a.x - b, a.y - b);
}
inline __host__ __device__ float2 operator-(float b, float2 a)
{
    return make_float2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(float2 &a, float b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(int2 &a, int2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ int2 operator-(int2 a, int b)
{
    return make_int2(a.x - b, a.y - b);
}
inline __host__ __device__ int2 operator-(int b, int2 a)
{
    return make_int2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(int2 &a, int b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ uint2 operator-(uint2 a, uint2 b)
{
    return make_uint2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(uint2 &a, uint2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ uint2 operator-(uint2 a, uint b)
{
    return make_uint2(a.x - b, a.y - b);
}
inline __host__ __device__ uint2 operator-(uint b, uint2 a)
{
    return make_uint2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(uint2 &a, uint b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ float3 operator-(float b, float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(float3 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ int3 operator-(int3 a, int3 b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(int3 &a, int3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ int3 operator-(int3 a, int b)
{
    return make_int3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ int3 operator-(int b, int3 a)
{
    return make_int3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(int3 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ uint3 operator-(uint3 a, uint3 b)
{
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(uint3 &a, uint3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ uint3 operator-(uint3 a, uint b)
{
    return make_uint3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ uint3 operator-(uint b, uint3 a)
{
    return make_uint3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(uint3 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ float4 operator-(float4 a, float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ void operator-=(float4 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ int4 operator-(int4 a, int4 b)
{
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(int4 &a, int4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ int4 operator-(int4 a, int b)
{
    return make_int4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ int4 operator-(int b, int4 a)
{
    return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(int4 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ uint4 operator-(uint4 a, uint4 b)
{
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(uint4 &a, uint4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ uint4 operator-(uint4 a, uint b)
{
    return make_uint4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ uint4 operator-(uint b, uint4 a)
{
    return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(uint4 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(float2 &a, float2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ float2 operator*(float b, float2 a)
{
    return make_float2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(float2 &a, float b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ int2 operator*(int2 a, int2 b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(int2 &a, int2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ int2 operator*(int2 a, int b)
{
    return make_int2(a.x * b, a.y * b);
}
inline __host__ __device__ int2 operator*(int b, int2 a)
{
    return make_int2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(int2 &a, int b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ uint2 operator*(uint2 a, uint2 b)
{
    return make_uint2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ uint2 operator*(uint2 a, uint b)
{
    return make_uint2(a.x * b, a.y * b);
}
inline __host__ __device__ uint2 operator*(uint b, uint2 a)
{
    return make_uint2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(float3 &a, float3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(float3 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ int3 operator*(int3 a, int3 b)
{
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(int3 &a, int3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ int3 operator*(int3 a, int b)
{
    return make_int3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ int3 operator*(int b, int3 a)
{
    return make_int3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(int3 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ uint3 operator*(uint3 a, uint3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ uint3 operator*(uint3 a, uint b)
{
    return make_uint3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ uint3 operator*(uint b, uint3 a)
{
    return make_uint3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(float4 &a, float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ float4 operator*(float b, float4 a)
{
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(float4 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ int4 operator*(int4 a, int4 b)
{
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(int4 &a, int4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ int4 operator*(int4 a, int b)
{
    return make_int4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ int4 operator*(int b, int4 a)
{
    return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(int4 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ uint4 operator*(uint4 a, uint4 b)
{
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ uint4 operator*(uint4 a, uint b)
{
    return make_uint4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ uint4 operator*(uint b, uint4 a)
{
    return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ void operator/=(float2 &a, float2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}
inline __host__ __device__ float2 operator/(float2 a, float b)
{
    return make_float2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(float2 &a, float b)
{
    a.x /= b;
    a.y /= b;
}
inline __host__ __device__ float2 operator/(float b, float2 a)
{
    return make_float2(b / a.x, b / a.y);
}

inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(float3 &a, float3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __host__ __device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(float3 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline __host__ __device__ float3 operator/(float b, float3 a)
{
    return make_float3(b / a.x, b / a.y, b / a.z);
}

inline __host__ __device__ float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}
inline __host__ __device__ void operator/=(float4 &a, float4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
inline __host__ __device__ float4 operator/(float4 a, float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);
}
inline __host__ __device__ void operator/=(float4 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}
inline __host__ __device__ float4 operator/(float b, float4 a)
{
    return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

inline  __host__ __device__ float2 fminf(float2 a, float2 b)
{
    return make_float2(fminf(a.x,b.x), fminf(a.y,b.y));
}
inline __host__ __device__ float3 fminf(float3 a, float3 b)
{
    return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
inline  __host__ __device__ float4 fminf(float4 a, float4 b)
{
    return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}

inline __host__ __device__ int2 min(int2 a, int2 b)
{
    return make_int2(min(a.x,b.x), min(a.y,b.y));
}
inline __host__ __device__ int3 min(int3 a, int3 b)
{
    return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}
inline __host__ __device__ int4 min(int4 a, int4 b)
{
    return make_int4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

inline __host__ __device__ uint2 min(uint2 a, uint2 b)
{
    return make_uint2(min(a.x,b.x), min(a.y,b.y));
}
inline __host__ __device__ uint3 min(uint3 a, uint3 b)
{
    return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}
inline __host__ __device__ uint4 min(uint4 a, uint4 b)
{
    return make_uint4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fmaxf(float2 a, float2 b)
{
    return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y));
}
inline __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
inline __host__ __device__ float4 fmaxf(float4 a, float4 b)
{
    return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

inline __host__ __device__ int2 max(int2 a, int2 b)
{
    return make_int2(max(a.x,b.x), max(a.y,b.y));
}
inline __host__ __device__ int3 max(int3 a, int3 b)
{
    return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}
inline __host__ __device__ int4 max(int4 a, int4 b)
{
    return make_int4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

inline __host__ __device__ uint2 max(uint2 a, uint2 b)
{
    return make_uint2(max(a.x,b.x), max(a.y,b.y));
}
inline __host__ __device__ uint3 max(uint3 a, uint3 b)
{
    return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}
inline __host__ __device__ uint4 max(uint4 a, uint4 b)
{
    return make_uint4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float2 lerp(float2 a, float2 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float4 lerp(float4 a, float4 b, float t)
{
    return a + t*(b-a);
}

////////////////////////////////////////////////////////////////////////////////
// clamp
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}
inline __device__ __host__ int clamp(int f, int a, int b)
{
    return max(a, min(f, b));
}
inline __device__ __host__ uint clamp(uint f, uint a, uint b)
{
    return max(a, min(f, b));
}

inline __device__ __host__ float2 clamp(float2 v, float a, float b)
{
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ float2 clamp(float2 v, float2 a, float2 b)
{
    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ float4 clamp(float4 v, float a, float b)
{
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ float4 clamp(float4 v, float4 a, float4 b)
{
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ int2 clamp(int2 v, int a, int b)
{
    return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ int2 clamp(int2 v, int2 a, int2 b)
{
    return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ int3 clamp(int3 v, int a, int b)
{
    return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ int3 clamp(int3 v, int3 a, int3 b)
{
    return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ int4 clamp(int4 v, int a, int b)
{
    return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ int4 clamp(int4 v, int4 a, int4 b)
{
    return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ uint2 clamp(uint2 v, uint a, uint b)
{
    return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ uint2 clamp(uint2 v, uint2 a, uint2 b)
{
    return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint a, uint b)
{
    return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint3 a, uint3 b)
{
    return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint a, uint b)
{
    return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint4 a, uint4 b)
{
    return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ int dot(int2 a, int2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ int dot(int3 a, int3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ int dot(int4 a, int4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ uint dot(uint2 a, uint2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ uint dot(uint3 a, uint3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ uint dot(uint4 a, uint4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float length(float2 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float4 v)
{
    return sqrtf(dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 normalize(float2 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float4 normalize(float4 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 floorf(float2 v)
{
    return make_float2(floorf(v.x), floorf(v.y));
}
inline __host__ __device__ float3 floorf(float3 v)
{
    return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}
inline __host__ __device__ float4 floorf(float4 v)
{
    return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float fracf(float v)
{
    return v - floorf(v);
}
inline __host__ __device__ float2 fracf(float2 v)
{
    return make_float2(fracf(v.x), fracf(v.y));
}
inline __host__ __device__ float3 fracf(float3 v)
{
    return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
inline __host__ __device__ float4 fracf(float4 v)
{
    return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fmodf(float2 a, float2 b)
{
    return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
}
inline __host__ __device__ float3 fmodf(float3 a, float3 b)
{
    return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}
inline __host__ __device__ float4 fmodf(float4 a, float4 b)
{
    return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fabs(float2 v)
{
    return make_float2(fabs(v.x), fabs(v.y));
}
inline __host__ __device__ float3 fabs(float3 v)
{
    return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}
inline __host__ __device__ float4 fabs(float4 v)
{
    return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

inline __host__ __device__ int2 abs(int2 v)
{
    return make_int2(abs(v.x), abs(v.y));
}
inline __host__ __device__ int3 abs(int3 v)
{
    return make_int3(abs(v.x), abs(v.y), abs(v.z));
}
inline __host__ __device__ int4 abs(int4 v)
{
    return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 reflect(float3 i, float3 n)
{
    return i - 2.0f * n * dot(n,i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

////////////////////////////////////////////////////////////////////////////////
// smoothstep
// - returns 0 if x < a
// - returns 1 if x > b
// - otherwise returns smooth interpolation between 0 and 1 based on x
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float smoothstep(float a, float b, float x)
{
    float y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(3.0f - (2.0f*y)));
}
inline __device__ __host__ float2 smoothstep(float2 a, float2 b, float2 x)
{
    float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float2(3.0f) - (make_float2(2.0f)*y)));
}
inline __device__ __host__ float3 smoothstep(float3 a, float3 b, float3 x)
{
    float3 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float3(3.0f) - (make_float3(2.0f)*y)));
}
inline __device__ __host__ float4 smoothstep(float4 a, float4 b, float4 x)
{
    float4 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float4(3.0f) - (make_float4(2.0f)*y)));
}












//----------
// mat3x3
template<class T>
class tensor3x3
{
public:
	__host__ __device__
		tensor3x3()
	{
		clear();
	}

	__host__ __device__
		void make_identity()
	{
		for (int r = 0; r < 3; r++)
			for (int c = 0; c < 3; c++)
				element(r, c) = (r == c) ? 1.0 : 0.0f;
	}

	__host__ __device__
		void clear()
	{
		for (int r = 0; r < 3; r++)
			for (int c = 0; c < 3; c++)
				element(r, c) = 0.0f;
	}

	__host__ __device__
		float frobeniusNorm()
	{
		float f = 0.0f;

		for (int r = 0; r < 3; r++)
		{
			for (int c = 0; c < 3; c++)
			{
				f += pow(element(r, c), 2.0f);
			}
		}

		return sqrtf(f);
	}

	__host__ __device__
		float magnitude()
	{
		return frobeniusNorm();
	}

	__host__ __device__
		float trace()
	{
		float t = 0.0f;

		for (int i = 0; i < 3; i++)
		{
			t += _array[i][i];
		}

		return t;
	}

	__host__ __device__
		tensor3x3<T> getTraceless()
	{
		float mean = trace() / 3.0;

		tensor3x3<T> tl;
		for (int r = 0; r < 3; r++)
		{
			for (int c = 0; c < 3; c++)
			{
				tl.element(r, c) = element(r, c);

				if (r == c)
				{
					tl.element(c, c) -= mean;
				}
			}
		}

		return tl;
	}

	__host__ __device__
		tensor3x3<T> transpose()
	{
		tensor3x3<T> transposed;

		for (int r = 0; r < 3; r++)
		{
			for (int c = 0; c < 3; c++)
			{
				transposed._array[c][r] += _array[r][c];
			}
		}

		return transposed;
	}

	__host__ __device__
		T &operator()(int row, int col)
	{
		return element(row, col);
	}

	__host__ __device__
		const T &operator()(int row, int col) const
	{
		return element(row, col);
	}

	__host__ __device__
		T &element(int row, int col)
	{
		return _array[row][col];
	}

	__host__ __device__
		const T &element(int row, int col) const
	{
		return _array[row][col];
	}

	// operators
	__host__ __device__
		tensor3x3<T> operator*(const float &s)
	{
		tensor3x3<T> t;

		for (int r = 0; r < 3; r++)
			for (int c = 0; c < 3; c++)
				t._array[r][c] = _array[r][c] * s;

		return t;
	}

	__host__ __device__
		tensor3x3<T> operator/(const float &s)
	{
		tensor3x3<T> t;

		for (int r = 0; r < 3; r++)
		{
			for (int c = 0; c < 3; c++)
			{
				if (s != 0.0f) t._array[r][c] = _array[r][c] / s;
			}
		}
		return t;
	}

	__host__ __device__
		tensor3x3<T> operator/(const tensor3x3<T> &t)
	{
		tensor3x3<T> t_result;

		for (int r = 0; r < 3; r++)
		{
			for (int c = 0; c < 3; c++)
			{
				if (t._array[r][c] != 0.0f) t_result._array[r][c] = _array[r][c] / t._array[r][c];
			}
		}
		return t_result;
	}

	__host__ __device__
		tensor3x3<T> operator-(const float &s)
	{
		tensor3x3<T> t;

		for (int r = 0; r < 3; r++)
			for (int c = 0; c < 3; c++)
				t._array[r][c] = _array[r][c] - s;

		return t;
	}

	__host__ __device__
		tensor3x3<T> operator+(tensor3x3<T> &t_in)
	{
		tensor3x3<T> t;

		for (int r = 0; r < 3; r++)
			for (int c = 0; c < 3; c++)
				t._array[r][c] = _array[r][c] + t_in._array[r][c];

		return t;
	}

	__host__ __device__
		tensor3x3<T> operator-(tensor3x3<T> &t_in)
	{
		tensor3x3<T> t;

		for (int r = 0; r < 3; r++)
			for (int c = 0; c < 3; c++)
				t._array[r][c] = _array[r][c] - t_in._array[r][c];

		return t;
	}

	__host__ __device__
		float3 operator*(const float3 &v)
	{
		float3 r = make_float3(0.0f);

		r.x = _array[0][0] * v.x + _array[0][1] * v.y + _array[0][2] * v.z;
		r.y = _array[1][0] * v.x + _array[1][1] * v.y + _array[1][2] * v.z;
		r.z = _array[2][0] * v.x + _array[2][1] * v.y + _array[2][2] * v.z;

		return r;
	}

	// data
	T _array[3][3];
};





// MATRIX
class matrix4x4
{
public:
	__host__ __device__
		matrix4x4()
	{
		make_identity();
	}

	__host__ __device__
		matrix4x4(float v)
	{
		for (int r = 0; r < 4; r++)
		{
			for (int c = 0; c < 4; c++)
			{
				if (r == c) element(r, c) = v;
				else element(r, c) = 0.0f;
			}
		}
	}

	__host__ __device__
		void make_identity()
	{
		for (int r = 0; r < 4; r++)
		{
			for (int c = 0; c < 4; c++)
			{
				if (r == c) element(r, c) = 1.0f;
				else element(r, c) = 0.0f;
			}
		}
	}

	__host__ __device__
		void clear()
	{
		for (int r = 0; r < 4; r++)
		{
			for (int c = 0; c < 4; c++)
			{
				element(r, c) = 0.0f;
			}
		}
	}

	__host__ __device__
		void rotationFromAxis(float3 axis, float degRad)
	{
		make_identity();

		element(0, 0) = cos(degRad) + axis.x*axis.x * (1.0f - cos(degRad));
		element(1, 0) = axis.y*axis.x * (1.0f - cos(degRad)) + axis.z * sin(degRad);
		element(2, 0) = axis.z*axis.x * (1.0f - cos(degRad)) - axis.y * sin(degRad);

		element(0, 1) = axis.x*axis.y * (1.0f - cos(degRad)) - axis.z * sin(degRad);
		element(1, 1) = cos(degRad) + axis.y*axis.y * (1.0f - cos(degRad));
		element(2, 1) = axis.z*axis.y * (1.0f - cos(degRad)) + axis.x * sin(degRad);

		element(0, 2) = axis.x*axis.z * (1.0f - cos(degRad)) + axis.y * sin(degRad);
		element(1, 2) = axis.y*axis.z * (1.0f - cos(degRad)) - axis.x * sin(degRad);
		element(2, 2) = cos(degRad) + axis.z*axis.z * (1.0f - cos(degRad));
	}

	__host__ __device__
		float &element(int row, int col)
	{
		return _array[row | (col << 2)];
	}

	__host__ __device__
		const float &element(int row, int col) const
	{
		return _array[row | (col << 2)];
	}

	__host__ __device__
		const float &operator()(int row, int col) const
	{
		return element(row, col);
	}

	__host__ __device__
		friend matrix4x4 operator*(const matrix4x4 &lhs, const matrix4x4 &rhs)
	{
		matrix4x4 r;
		r.clear();

		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				for (int c = 0; c < 4; c++)
				{
					r.element(i, j) += lhs(i, c) * rhs(c, j);
				}

		return r;
	}

	__host__ __device__
		void transpose()
	{
		matrix4x4 mcopy;
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
			{
				mcopy.element(i, j) = element(i, j);
			}

		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
			{
				element(i, j) = mcopy.element(j, i);
			}
	}

	__host__ __device__
		float4 operator *(const float4 &in) const
	{
		float src[] = { in.x, in.y, in.z, in.w };
		float r[4];

		for (int i = 0; i < 4; i++)
			r[i] = (src[0] * element(i, 0) + src[1] * element(i, 1) + src[2] * element(i, 2) + src[3] * element(i, 3));

		return make_float4(r[0], r[1], r[2], r[3]);
	}

	__host__ __device__
		float3 mulVec(const float3 & in) const
	{
		float src[] = { in.x, in.y, in.z, 0.0f };
		float r[3];

		for (int i = 0; i < 3; i++)
			r[i] = (src[0] * element(i, 0) + src[1] * element(i, 1) + src[2] * element(i, 2) + src[3] * element(i, 3));

		return make_float3(r[0], r[1], r[2]);
	}

	__host__ __device__
		float3 mulPt(const float3 & in) const
	{
		float src[] = { in.x, in.y, in.z, 1.0f };
		float r[3];

		for (int i = 0; i < 3; i++)
			r[i] = (src[0] * element(i, 0) + src[1] * element(i, 1) + src[2] * element(i, 2) + src[3] * element(i, 3));

		return make_float3(r[0], r[1], r[2]);
	}


	union
	{
		struct
		{
			float _11, _12, _13, _14;
			float _21, _22, _23, _24;
			float _31, _32, _33, _34;
			float _41, _42, _43, _44;
		};
		float _array[16];
	};
};



#endif