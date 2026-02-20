/*
Copyright (c) 2017 landave

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#ifndef KERNEL_H_
#define KERNEL_H_

#include <string>
#include "TSHasherContext.h"

const char* TSHasherContext::KERNEL_CODE = R"(
#if VENDOR_ID == (1 << 0)
#define IS_AMD
#elif VENDOR_ID == (1 << 5)
#define IS_NV
#else
#define IS_GENERIC
#endif

#ifdef IS_AMD
#pragma OPENCL EXTENSION cl_amd_media_ops  : enable
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
#endif

// the following basic SHA1 implementation has been
// taken from the hashcat project
// see https://hashcat.net/hashcat/
// and https://github.com/hashcat/hashcat

typedef enum sha1_constants
{
  SHA1M_A=0x67452301,
  SHA1M_B=0xefcdab89,
  SHA1M_C=0x98badcfe,
  SHA1M_D=0x10325476,
  SHA1M_E=0xc3d2e1f0,

  SHA1C00=0x5a827999,
  SHA1C01=0x6ed9eba1,
  SHA1C02=0x8f1bbcdc,
  SHA1C03=0xca62c1d6u

} sha1_constants_t;

#ifdef IS_NV
#define SHA1_F0(x,y,z)  ((z) ^ ((x) & ((y) ^ (z))))
#define SHA1_F1(x,y,z)  ((x) ^ (y) ^ (z))
#define SHA1_F2(x,y,z)  (((x) & (y)) | ((z) & ((x) ^ (y))))
#define SHA1_F0o(x,y,z) (bitselect ((z), (y), (x)))
#define SHA1_F2o(x,y,z) (bitselect ((x), (y), ((x) ^ (z))))
#endif

#ifdef IS_AMD
#define SHA1_F0(x,y,z)  ((z) ^ ((x) & ((y) ^ (z))))
#define SHA1_F1(x,y,z)  ((x) ^ (y) ^ (z))
#define SHA1_F2(x,y,z)  (((x) & (y)) | ((z) & ((x) ^ (y))))
#define SHA1_F0o(x,y,z) (bitselect ((z), (y), (x)))
#define SHA1_F2o(x,y,z) (bitselect ((x), (y), ((x) ^ (z))))
#endif

#ifdef IS_GENERIC
#define SHA1_F0(x,y,z)  ((z) ^ ((x) & ((y) ^ (z))))
#define SHA1_F1(x,y,z)  ((x) ^ (y) ^ (z))
#define SHA1_F2(x,y,z)  (((x) & (y)) | ((z) & ((x) ^ (y))))
#define SHA1_F0o(x,y,z) (SHA1_F0 ((x), (y), (z)))
#define SHA1_F2o(x,y,z) (SHA1_F2 ((x), (y), (z)))
#endif


#define SHA1_STEP(f,a,b,c,d,e,x)    \
{                                   \
  e += K;                           \
  e += x;                           \
  e += f (b, c, d);                 \
  e += rotate (a,  5u);             \
  b  = rotate (b, 30u);             \
}

// Spezielles Makro fuer die uint4 Vektorisierung
#define SHA1_STEP_VEC4(f,a,b,c,d,e,x)        \
{                                             \
  e += (uint4)(K);                            \
  e += x;                                     \
  e += f (b, c, d);                           \
  e += rotate (a, (uint4)(5u));               \
  b  = rotate (b, (uint4)(30u));              \
}

typedef uchar  u8;
typedef ushort u16;
typedef uint   u32;
typedef ulong  u64;


void sha1_64 (u32 block[16], u32 digest[5])
{

  u32 a = digest[0];
  u32 b = digest[1];
  u32 c = digest[2];
  u32 d = digest[3];
  u32 e = digest[4];

  u32 w0_t = block[ 0];
  u32 w1_t = block[ 1];
  u32 w2_t = block[ 2];
  u32 w3_t = block[ 3];
  u32 w4_t = block[ 4];
  u32 w5_t = block[ 5];
  u32 w6_t = block[ 6];
  u32 w7_t = block[ 7];
  u32 w8_t = block[ 8];
  u32 w9_t = block[ 9];
  u32 wa_t = block[10];
  u32 wb_t = block[11];
  u32 wc_t = block[12];
  u32 wd_t = block[13];
  u32 we_t = block[14];
  u32 wf_t = block[15];

  #undef K
  #define K SHA1C00

  SHA1_STEP (SHA1_F0o, a, b, c, d, e, w0_t);
  SHA1_STEP (SHA1_F0o, e, a, b, c, d, w1_t);
  SHA1_STEP (SHA1_F0o, d, e, a, b, c, w2_t);
  SHA1_STEP (SHA1_F0o, c, d, e, a, b, w3_t);
  SHA1_STEP (SHA1_F0o, b, c, d, e, a, w4_t);
  SHA1_STEP (SHA1_F0o, a, b, c, d, e, w5_t);
  SHA1_STEP (SHA1_F0o, e, a, b, c, d, w6_t);
  SHA1_STEP (SHA1_F0o, d, e, a, b, c, w7_t);
  SHA1_STEP (SHA1_F0o, c, d, e, a, b, w8_t);
  SHA1_STEP (SHA1_F0o, b, c, d, e, a, w9_t);
  SHA1_STEP (SHA1_F0o, a, b, c, d, e, wa_t);
  SHA1_STEP (SHA1_F0o, e, a, b, c, d, wb_t);
  SHA1_STEP (SHA1_F0o, d, e, a, b, c, wc_t);
  SHA1_STEP (SHA1_F0o, c, d, e, a, b, wd_t);
  SHA1_STEP (SHA1_F0o, b, c, d, e, a, we_t);
  SHA1_STEP (SHA1_F0o, a, b, c, d, e, wf_t);
 
  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP (SHA1_F0o, e, a, b, c, d, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP (SHA1_F0o, d, e, a, b, c, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP (SHA1_F0o, c, d, e, a, b, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP (SHA1_F0o, b, c, d, e, a, w3_t);

  #undef K
  #define K SHA1C01

  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, w7_t);
  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, wb_t);
  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, wf_t);
  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, w3_t);
  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, w7_t);

  #undef K
  #define K SHA1C02

  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP (SHA1_F2o, a, b, c, d, e, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP (SHA1_F2o, e, a, b, c, d, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP (SHA1_F2o, d, e, a, b, c, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP (SHA1_F2o, c, d, e, a, b, wb_t);
  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP (SHA1_F2o, b, c, d, e, a, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP (SHA1_F2o, a, b, c, d, e, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP (SHA1_F2o, e, a, b, c, d, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP (SHA1_F2o, d, e, a, b, c, wf_t);
  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP (SHA1_F2o, c, d, e, a, b, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP (SHA1_F2o, b, c, d, e, a, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP (SHA1_F2o, a, b, c, d, e, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP (SHA1_F2o, e, a, b, c, d, w3_t);
  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP (SHA1_F2o, d, e, a, b, c, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP (SHA1_F2o, c, d, e, a, b, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP (SHA1_F2o, b, c, d, e, a, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP (SHA1_F2o, a, b, c, d, e, w7_t);
  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP (SHA1_F2o, e, a, b, c, d, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP (SHA1_F2o, d, e, a, b, c, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP (SHA1_F2o, c, d, e, a, b, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP (SHA1_F2o, b, c, d, e, a, wb_t);

  #undef K
  #define K SHA1C03

  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, wf_t);
  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, w3_t);
  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, w7_t);
  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, wb_t);
  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, wf_t);

  digest[0] += a;
  digest[1] += b;
  digest[2] += c;
  digest[3] += d;
  digest[4] += e;
}
)"

R"(
// Vektorisierte volle SHA-1 Kompression (4 Lanes parallel, fuer Block 1 im Double-Block-Kernel)
void sha1_64_vec4 (uint4 block[16], uint4 digest[5])
{
  uint4 a = digest[0];
  uint4 b = digest[1];
  uint4 c = digest[2];
  uint4 d = digest[3];
  uint4 e = digest[4];

  uint4 w0_t = block[ 0];
  uint4 w1_t = block[ 1];
  uint4 w2_t = block[ 2];
  uint4 w3_t = block[ 3];
  uint4 w4_t = block[ 4];
  uint4 w5_t = block[ 5];
  uint4 w6_t = block[ 6];
  uint4 w7_t = block[ 7];
  uint4 w8_t = block[ 8];
  uint4 w9_t = block[ 9];
  uint4 wa_t = block[10];
  uint4 wb_t = block[11];
  uint4 wc_t = block[12];
  uint4 wd_t = block[13];
  uint4 we_t = block[14];
  uint4 wf_t = block[15];

  #undef K
  #define K SHA1C00

  SHA1_STEP_VEC4 (SHA1_F0o, a, b, c, d, e, w0_t);
  SHA1_STEP_VEC4 (SHA1_F0o, e, a, b, c, d, w1_t);
  SHA1_STEP_VEC4 (SHA1_F0o, d, e, a, b, c, w2_t);
  SHA1_STEP_VEC4 (SHA1_F0o, c, d, e, a, b, w3_t);
  SHA1_STEP_VEC4 (SHA1_F0o, b, c, d, e, a, w4_t);
  SHA1_STEP_VEC4 (SHA1_F0o, a, b, c, d, e, w5_t);
  SHA1_STEP_VEC4 (SHA1_F0o, e, a, b, c, d, w6_t);
  SHA1_STEP_VEC4 (SHA1_F0o, d, e, a, b, c, w7_t);
  SHA1_STEP_VEC4 (SHA1_F0o, c, d, e, a, b, w8_t);
  SHA1_STEP_VEC4 (SHA1_F0o, b, c, d, e, a, w9_t);
  SHA1_STEP_VEC4 (SHA1_F0o, a, b, c, d, e, wa_t);
  SHA1_STEP_VEC4 (SHA1_F0o, e, a, b, c, d, wb_t);
  SHA1_STEP_VEC4 (SHA1_F0o, d, e, a, b, c, wc_t);
  SHA1_STEP_VEC4 (SHA1_F0o, c, d, e, a, b, wd_t);
  SHA1_STEP_VEC4 (SHA1_F0o, b, c, d, e, a, we_t);
  SHA1_STEP_VEC4 (SHA1_F0o, a, b, c, d, e, wf_t);

  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, e, a, b, c, d, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, d, e, a, b, c, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, c, d, e, a, b, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, b, c, d, e, a, w3_t);

  #undef K
  #define K SHA1C01

  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w7_t);
  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, wb_t);
  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wf_t);
  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w3_t);
  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w7_t);

  #undef K
  #define K SHA1C02

  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, wb_t);
  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, wf_t);
  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, w3_t);
  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, w7_t);
  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, wb_t);

  #undef K
  #define K SHA1C03

  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, wf_t);
  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w3_t);
  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w7_t);
  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, wb_t);
  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, wf_t);

  digest[0] += a;
  digest[1] += b;
  digest[2] += c;
  digest[3] += d;
  digest[4] += e;
}
)"

R"(
bool sha1_64_check (u32 block[16], u32 digest[5], u32 td0, u32 td1, u32 td2, u32 td3, u32 td4) {
  u32 a = digest[0];
  u32 b = digest[1];
  u32 c = digest[2];
  u32 d = digest[3];
  u32 e = digest[4];

  u32 w0_t = block[ 0];
  u32 w1_t = block[ 1];
  u32 w2_t = block[ 2];
  u32 w3_t = block[ 3];
  u32 w4_t = block[ 4];
  u32 w5_t = block[ 5];
  u32 w6_t = block[ 6];
  u32 w7_t = block[ 7];
  u32 w8_t = block[ 8];
  u32 w9_t = block[ 9];
  u32 wa_t = block[10];
  u32 wb_t = block[11];
  u32 wc_t = block[12];
  u32 wd_t = block[13];
  u32 we_t = block[14];
  u32 wf_t = block[15];

  #undef K
  #define K SHA1C00

  SHA1_STEP (SHA1_F0o, a, b, c, d, e, w0_t);
  SHA1_STEP (SHA1_F0o, e, a, b, c, d, w1_t);
  SHA1_STEP (SHA1_F0o, d, e, a, b, c, w2_t);
  SHA1_STEP (SHA1_F0o, c, d, e, a, b, w3_t);
  SHA1_STEP (SHA1_F0o, b, c, d, e, a, w4_t);
  SHA1_STEP (SHA1_F0o, a, b, c, d, e, w5_t);
  SHA1_STEP (SHA1_F0o, e, a, b, c, d, w6_t);
  SHA1_STEP (SHA1_F0o, d, e, a, b, c, w7_t);
  SHA1_STEP (SHA1_F0o, c, d, e, a, b, w8_t);
  SHA1_STEP (SHA1_F0o, b, c, d, e, a, w9_t);
  SHA1_STEP (SHA1_F0o, a, b, c, d, e, wa_t);
  SHA1_STEP (SHA1_F0o, e, a, b, c, d, wb_t);
  SHA1_STEP (SHA1_F0o, d, e, a, b, c, wc_t);
  SHA1_STEP (SHA1_F0o, c, d, e, a, b, wd_t);
  SHA1_STEP (SHA1_F0o, b, c, d, e, a, we_t);
  SHA1_STEP (SHA1_F0o, a, b, c, d, e, wf_t);

  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP (SHA1_F0o, e, a, b, c, d, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP (SHA1_F0o, d, e, a, b, c, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP (SHA1_F0o, c, d, e, a, b, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP (SHA1_F0o, b, c, d, e, a, w3_t);

  #undef K
  #define K SHA1C01

  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, w7_t);
  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, wb_t);
  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, wf_t);
  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, w3_t);
  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, w7_t);

  #undef K
  #define K SHA1C02

  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP (SHA1_F2o, a, b, c, d, e, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP (SHA1_F2o, e, a, b, c, d, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP (SHA1_F2o, d, e, a, b, c, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP (SHA1_F2o, c, d, e, a, b, wb_t);
  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP (SHA1_F2o, b, c, d, e, a, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP (SHA1_F2o, a, b, c, d, e, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP (SHA1_F2o, e, a, b, c, d, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP (SHA1_F2o, d, e, a, b, c, wf_t);
  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP (SHA1_F2o, c, d, e, a, b, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP (SHA1_F2o, b, c, d, e, a, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP (SHA1_F2o, a, b, c, d, e, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP (SHA1_F2o, e, a, b, c, d, w3_t);
  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP (SHA1_F2o, d, e, a, b, c, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP (SHA1_F2o, c, d, e, a, b, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP (SHA1_F2o, b, c, d, e, a, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP (SHA1_F2o, a, b, c, d, e, w7_t);
  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP (SHA1_F2o, e, a, b, c, d, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP (SHA1_F2o, d, e, a, b, c, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP (SHA1_F2o, c, d, e, a, b, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP (SHA1_F2o, b, c, d, e, a, wb_t);

  #undef K
  #define K SHA1C03

  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, wf_t);
  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, w3_t);
  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, w7_t);
  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, wb_t);
  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, wf_t);

  // ---- EARLY EXIT LOGIK ----
  u32 final_h0 = digest[0] + a;

  // Da TeamSpeak Level >= 32 gesucht werden, muss der erste Block zwingend 0 sein
  if (final_h0 != 0) return false;

  // Nur bei einem potentiellen Treffer berechnen wir den Rest
  u32 final_h1 = digest[1] + b;
  u32 final_h2 = digest[2] + c;
  u32 final_h3 = digest[3] + d;
  u32 final_h4 = digest[4] + e;

  return 0 == ((final_h0 & td0) |
               (final_h1 & td1) |
               (final_h2 & td2) |
               (final_h3 & td3) |
               (final_h4 & td4));
}
)"

R"(
// Vektorisierte Early-Exit SHA-1 Funktion (4 Hashes parallel)
bool sha1_64_check_vec4 (uint4 block[16], uint4 digest[5], uchar targetdifficulty) {
  uint4 a = digest[0];
  uint4 b = digest[1];
  uint4 c = digest[2];
  uint4 d = digest[3];
  uint4 e = digest[4];

  uint4 w0_t = block[ 0];
  uint4 w1_t = block[ 1];
  uint4 w2_t = block[ 2];
  uint4 w3_t = block[ 3];
  uint4 w4_t = block[ 4];
  uint4 w5_t = block[ 5];
  uint4 w6_t = block[ 6];
  uint4 w7_t = block[ 7];
  uint4 w8_t = block[ 8];
  uint4 w9_t = block[ 9];
  uint4 wa_t = block[10];
  uint4 wb_t = block[11];
  uint4 wc_t = block[12];
  uint4 wd_t = block[13];
  uint4 we_t = block[14];
  uint4 wf_t = block[15];

  #undef K
  #define K SHA1C00

  SHA1_STEP_VEC4 (SHA1_F0o, a, b, c, d, e, w0_t);
  SHA1_STEP_VEC4 (SHA1_F0o, e, a, b, c, d, w1_t);
  SHA1_STEP_VEC4 (SHA1_F0o, d, e, a, b, c, w2_t);
  SHA1_STEP_VEC4 (SHA1_F0o, c, d, e, a, b, w3_t);
  SHA1_STEP_VEC4 (SHA1_F0o, b, c, d, e, a, w4_t);
  SHA1_STEP_VEC4 (SHA1_F0o, a, b, c, d, e, w5_t);
  SHA1_STEP_VEC4 (SHA1_F0o, e, a, b, c, d, w6_t);
  SHA1_STEP_VEC4 (SHA1_F0o, d, e, a, b, c, w7_t);
  SHA1_STEP_VEC4 (SHA1_F0o, c, d, e, a, b, w8_t);
  SHA1_STEP_VEC4 (SHA1_F0o, b, c, d, e, a, w9_t);
  SHA1_STEP_VEC4 (SHA1_F0o, a, b, c, d, e, wa_t);
  SHA1_STEP_VEC4 (SHA1_F0o, e, a, b, c, d, wb_t);
  SHA1_STEP_VEC4 (SHA1_F0o, d, e, a, b, c, wc_t);
  SHA1_STEP_VEC4 (SHA1_F0o, c, d, e, a, b, wd_t);
  SHA1_STEP_VEC4 (SHA1_F0o, b, c, d, e, a, we_t);
  SHA1_STEP_VEC4 (SHA1_F0o, a, b, c, d, e, wf_t);

  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, e, a, b, c, d, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, d, e, a, b, c, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, c, d, e, a, b, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, b, c, d, e, a, w3_t);

  #undef K
  #define K SHA1C01

  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w7_t);
  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, wb_t);
  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wf_t);
  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w3_t);
  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w7_t);

  #undef K
  #define K SHA1C02

  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, wb_t);
  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, wf_t);
  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, w3_t);
  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, w7_t);
  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, wb_t);

  #undef K
  #define K SHA1C03

  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, wf_t);
  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w3_t);
  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w7_t);
  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, wb_t);
  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, wf_t);

  // ---- CLZ()-BASIERTE EARLY EXIT LOGIK FUER 4 LANES ----
  uint4 final_h0 = digest[0] + a;

  // Fuehrende Nullbits pro Lane zaehlen (ein Taktzyklus per clz!)
  uint z0 = clz(final_h0.s0);
  uint z1 = clz(final_h0.s1);
  uint z2 = clz(final_h0.s2);
  uint z3 = clz(final_h0.s3);

  // Quick-Reject: Wenn kein Lane genug Nullen im ersten Wort hat
  uint td_cap = min((uint)targetdifficulty, 32u);
  if (max(max(z0, z1), max(z2, z3)) < td_cap) return false;

  // Vielversprechend - zweites Wort pruefen wenn noetig
  uint4 final_h1 = digest[1] + b;
  if (z0 == 32) z0 += clz(final_h1.s0);
  if (z1 == 32) z1 += clz(final_h1.s1);
  if (z2 == 32) z2 += clz(final_h1.s2);
  if (z3 == 32) z3 += clz(final_h1.s3);

  return (z0 >= targetdifficulty) || (z1 >= targetdifficulty) ||
         (z2 >= targetdifficulty) || (z3 >= targetdifficulty);
}
)"

R"(
// Vektorisierte SHA-1 Funktion mit Partial Round Precomputation (Start ab Runde 8)
bool sha1_64_check_vec4_r8 (
    uint4 block[16], uint4 digest[5],
    uint4 pre_a, uint4 pre_b, uint4 pre_c, uint4 pre_d, uint4 pre_e,
    uchar targetdifficulty) {

  // Vorgespulten Zustand aus Runde 7 uebernehmen
  uint4 a = pre_a;
  uint4 b = pre_b;
  uint4 c = pre_c;
  uint4 d = pre_d;
  uint4 e = pre_e;

  // Alle 16 Woerter laden (w0-w7 fuer Message Expansion noetig)
  uint4 w0_t = block[ 0];
  uint4 w1_t = block[ 1];
  uint4 w2_t = block[ 2];
  uint4 w3_t = block[ 3];
  uint4 w4_t = block[ 4];
  uint4 w5_t = block[ 5];
  uint4 w6_t = block[ 6];
  uint4 w7_t = block[ 7];
  uint4 w8_t = block[ 8];
  uint4 w9_t = block[ 9];
  uint4 wa_t = block[10];
  uint4 wb_t = block[11];
  uint4 wc_t = block[12];
  uint4 wd_t = block[13];
  uint4 we_t = block[14];
  uint4 wf_t = block[15];

  #undef K
  #define K SHA1C00

  // RUNDEN 0-7 UEBERSPRUNGEN (vorberechnet)

  // Ab Runde 8
  SHA1_STEP_VEC4 (SHA1_F0o, c, d, e, a, b, w8_t);
  SHA1_STEP_VEC4 (SHA1_F0o, b, c, d, e, a, w9_t);
  SHA1_STEP_VEC4 (SHA1_F0o, a, b, c, d, e, wa_t);
  SHA1_STEP_VEC4 (SHA1_F0o, e, a, b, c, d, wb_t);
  SHA1_STEP_VEC4 (SHA1_F0o, d, e, a, b, c, wc_t);
  SHA1_STEP_VEC4 (SHA1_F0o, c, d, e, a, b, wd_t);
  SHA1_STEP_VEC4 (SHA1_F0o, b, c, d, e, a, we_t);
  SHA1_STEP_VEC4 (SHA1_F0o, a, b, c, d, e, wf_t);

  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F0o, e, a, b, c, d, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F0o, d, e, a, b, c, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F0o, c, d, e, a, b, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F0o, b, c, d, e, a, w3_t);

  #undef K
  #define K SHA1C01

  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w7_t);
  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, wb_t);
  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wf_t);
  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w3_t);
  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w7_t);

  #undef K
  #define K SHA1C02

  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, wb_t);
  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, wf_t);
  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, w3_t);
  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, w7_t);
  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, wb_t);

  #undef K
  #define K SHA1C03

  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, wf_t);
  w0_t = rotate ((wd_t ^ w8_t ^ w2_t ^ w0_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w0_t);
  w1_t = rotate ((we_t ^ w9_t ^ w3_t ^ w1_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w1_t);
  w2_t = rotate ((wf_t ^ wa_t ^ w4_t ^ w2_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w2_t);
  w3_t = rotate ((w0_t ^ wb_t ^ w5_t ^ w3_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w3_t);
  w4_t = rotate ((w1_t ^ wc_t ^ w6_t ^ w4_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w4_t);
  w5_t = rotate ((w2_t ^ wd_t ^ w7_t ^ w5_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w5_t);
  w6_t = rotate ((w3_t ^ we_t ^ w8_t ^ w6_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w6_t);
  w7_t = rotate ((w4_t ^ wf_t ^ w9_t ^ w7_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w7_t);
  w8_t = rotate ((w5_t ^ w0_t ^ wa_t ^ w8_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w8_t);
  w9_t = rotate ((w6_t ^ w1_t ^ wb_t ^ w9_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w9_t);
  wa_t = rotate ((w7_t ^ w2_t ^ wc_t ^ wa_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, wa_t);
  wb_t = rotate ((w8_t ^ w3_t ^ wd_t ^ wb_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, wb_t);
  wc_t = rotate ((w9_t ^ w4_t ^ we_t ^ wc_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wc_t);
  wd_t = rotate ((wa_t ^ w5_t ^ wf_t ^ wd_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, wd_t);
  we_t = rotate ((wb_t ^ w6_t ^ w0_t ^ we_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, we_t);
  wf_t = rotate ((wc_t ^ w7_t ^ w1_t ^ wf_t), (uint4)(1u));
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, wf_t);

  // ---- CLZ()-BASIERTE EARLY EXIT LOGIK FUER 4 LANES ----
  uint4 final_h0 = digest[0] + a;

  // Fuehrende Nullbits pro Lane zaehlen (ein Taktzyklus per clz!)
  uint z0 = clz(final_h0.s0);
  uint z1 = clz(final_h0.s1);
  uint z2 = clz(final_h0.s2);
  uint z3 = clz(final_h0.s3);

  // Quick-Reject: Wenn kein Lane genug Nullen im ersten Wort hat
  uint td_cap = min((uint)targetdifficulty, 32u);
  if (max(max(z0, z1), max(z2, z3)) < td_cap) return false;

  // Vielversprechend - zweites Wort pruefen wenn noetig
  uint4 final_h1 = digest[1] + b;
  if (z0 == 32) z0 += clz(final_h1.s0);
  if (z1 == 32) z1 += clz(final_h1.s1);
  if (z2 == 32) z2 += clz(final_h1.s2);
  if (z3 == 32) z3 += clz(final_h1.s3);

  return (z0 >= targetdifficulty) || (z1 >= targetdifficulty) ||
         (z2 >= targetdifficulty) || (z3 >= targetdifficulty);
}
)"

R"(
ulong countertostring(char* dst, ulong c) {
  if (c==0) { *dst = '0'; return 1; }
  char* dst0 = dst;
  char* dst1 = dst;
  ulong counterlength=0;
  while (c) {
    uchar currentdigit = c%10;
    *dst1 = '0'+currentdigit;
    dst1++;
    counterlength++;
    c = c/10;
  }
  // invert string
  
  dst1--;
  while (dst0 < dst1) {
    uchar tmp = *dst0;
    *dst0 = *dst1;
    *dst1 = tmp;
    dst0++;
    dst1--;
  }
  
  return counterlength;
}


inline uint swap_uint(uint val) {
  #ifdef IS_NV
    u32 r;
    asm ("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(r) : "r"(val));
    return r;
  #else
    return (as_uint (as_uchar4 (val).s3210));
  #endif
}
    
inline ulong swap_ulong(ulong val)
{
    #ifdef IS_NV
      u32 il;
      u32 ir;

      asm ("mov.b64 {%0, %1}, %2;" : "=r"(il), "=r"(ir) : "l"(val));

      u32 tl;
      u32 tr;

      asm ("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(tl) : "r"(il));
      asm ("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(tr) : "r"(ir));

      u64 r;

      asm ("mov.b64 %0, {%1, %2};" : "=l"(r) : "r"(tr), "r"(tl));

      return r;
    #else
      return (as_ulong (as_uchar8 (val).s76543210));
    #endif  
}

inline uchar getDifficulty(uchar* hash) {
  // we push all difficulty levels up to 8 to zero
  // this increases performance
  if (hash[0]!=0) { return 0; }

  
  uchar zerobytes = 1;
  while (zerobytes < 20 && hash[zerobytes] == 0) {
    zerobytes++;
  }
  uchar zerobits = 0;
  if (zerobytes < 20) {
    uchar lastbyte = hash[zerobytes];
    while (!(lastbyte & 1)) {
      zerobits++;
      lastbyte >>= 1;
    }
  }

  return 8 * zerobytes + zerobits;
}

// increases a decimal string counter
// counterend should point to the least significant digit of the counter
inline void increaseStringCounter(char *counterend) {
  bool add = 1;
  
  while (add) {
    uchar currentdigit = *counterend - '0';
    add = currentdigit == 9;
    *counterend = (currentdigit+1)%10 + '0';
    counterend--;
  }
}

// increases a decimal string counter
// counterend should point to the least significant digit of the counter
inline void increaseStringCounterSwapped(char *str, int endpos) {
  bool add = 1;

  while (add) {
    uchar currentdigit = str[endpos] - '0';
    add = currentdigit == 9;
    str[endpos] = (currentdigit+1)%10 + '0';
    endpos--;
  }
}

// Erhöht den String direkt im Big-Endian umgewandelten Array
inline void increaseStringCounterBE(uchar* hashstring_bytes, int counter_end_idx) {
  bool add = 1;
  int current_idx = counter_end_idx;
  while(add) {
    // Wandle den logischen String-Index in den physischen Big-Endian Speicher-Index um
    int phys_idx = (current_idx & ~3) | (3 - (current_idx & 3));
    uchar currentdigit = hashstring_bytes[phys_idx];
    if (currentdigit == '9') {
      hashstring_bytes[phys_idx] = '0';
      current_idx--;
    } else {
      hashstring_bytes[phys_idx] = currentdigit + 1;
      add = 0;
    }
  }
}

inline void compute_targetdigest(u32 targetdigest[5], uchar targetdifficulty) {
  int i=0;
  for (; i<targetdifficulty/32; i++) {
    targetdigest[i] = 0xFFFFFFFF;
  }
  if (i < 5) { targetdigest[i] = (((u32)1)<<(targetdifficulty%32))-1; i++; };
  for (; i<5; i++) {
    targetdigest[i] = 0;
  }
}
)"

R"(
__kernel void TeamSpeakHasher (const ulong startcounter,
                               const uint iterations,
                               const uchar targetdifficulty,
                               const __global uchar* identity,
                               const uint identity_length,
                               __global uchar *results)
{
  const int gid = get_global_id(0);
  const uint identity_length_snd_block = identity_length-64;

  u32 hashstring[16];
  
  for (int i=0; i < 64; i++) {
    ((uchar*)hashstring)[i] = identity[i];
  }

  //we hash the first block 
  u32 digest1[5];
  digest1[0] = SHA1M_A;
  digest1[1] = SHA1M_B;
  digest1[2] = SHA1M_C;
  digest1[3] = SHA1M_D;
  digest1[4] = SHA1M_E;
  for (int j = 0; j<16; j++) {
    hashstring[j] = swap_uint(((u32*)hashstring)[j]);
  }
  sha1_64(hashstring, digest1);

  for (int i=0; i<identity_length_snd_block; i++) {
    ((uchar*)hashstring)[i] = identity[i+64];
  }

  for (int i=identity_length_snd_block; i<64; i++) {
    ((uchar*)hashstring)[i] = 0;
  }

  for (int j = 0; j<identity_length_snd_block/4; j++) {
    hashstring[j] = swap_uint(hashstring[j]);
  }

  const int swapendianness_start = identity_length_snd_block/4;

  // --- 4-Lane Vektorisierung ---
  ulong chunk = iterations / 4;

  ulong currentcounter0 = startcounter + gid * iterations;
  ulong currentcounter1 = startcounter + gid * iterations + chunk;
  ulong currentcounter2 = startcounter + gid * iterations + 2 * chunk;
  ulong currentcounter3 = startcounter + gid * iterations + 3 * chunk;

  u32 hashstring0[16]; u32 hashstring1[16]; u32 hashstring2[16]; u32 hashstring3[16];

  for(int i=0; i<16; i++) {
    hashstring0[i] = hashstring[i]; hashstring1[i] = hashstring[i];
    hashstring2[i] = hashstring[i]; hashstring3[i] = hashstring[i];
  }

  // Counter-Strings und Padding fuer alle 4 Lanes generieren
  ulong clen0 = countertostring(((uchar*)hashstring0)+identity_length_snd_block, currentcounter0);
  ((uchar*)hashstring0)[identity_length_snd_block+clen0] = 0x80;
  *((ulong*)(((uchar*)hashstring0)+56)) = swap_ulong(8*((ulong)identity_length+clen0));

  ulong clen1 = countertostring(((uchar*)hashstring1)+identity_length_snd_block, currentcounter1);
  ((uchar*)hashstring1)[identity_length_snd_block+clen1] = 0x80;
  *((ulong*)(((uchar*)hashstring1)+56)) = swap_ulong(8*((ulong)identity_length+clen1));

  ulong clen2 = countertostring(((uchar*)hashstring2)+identity_length_snd_block, currentcounter2);
  ((uchar*)hashstring2)[identity_length_snd_block+clen2] = 0x80;
  *((ulong*)(((uchar*)hashstring2)+56)) = swap_ulong(8*((ulong)identity_length+clen2));

  ulong clen3 = countertostring(((uchar*)hashstring3)+identity_length_snd_block, currentcounter3);
  ((uchar*)hashstring3)[identity_length_snd_block+clen3] = 0x80;
  *((ulong*)(((uchar*)hashstring3)+56)) = swap_ulong(8*((ulong)identity_length+clen3));

  // Laenge einmalig zu Big-Endian
  hashstring0[14] = swap_uint(hashstring0[14]); hashstring0[15] = swap_uint(hashstring0[15]);
  hashstring1[14] = swap_uint(hashstring1[14]); hashstring1[15] = swap_uint(hashstring1[15]);
  hashstring2[14] = swap_uint(hashstring2[14]); hashstring2[15] = swap_uint(hashstring2[15]);
  hashstring3[14] = swap_uint(hashstring3[14]); hashstring3[15] = swap_uint(hashstring3[15]);

  // Endianness-Swap fuer Counter-Bereich
  for(int j=swapendianness_start; j<14; j++) {
    hashstring0[j] = swap_uint(hashstring0[j]); hashstring1[j] = swap_uint(hashstring1[j]);
    hashstring2[j] = swap_uint(hashstring2[j]); hashstring3[j] = swap_uint(hashstring3[j]);
  }

  // Berechne das letzte Wort, das sich durch den Counter aendert
  int max_clen = (int)max(max(clen0, clen1), max(clen2, clen3));
  int counter_word_end = (identity_length_snd_block + max_clen - 1) / 4;

  // Vektor-Arrays vorbereiten
  uint4 hashstring_vec[16];
  uint4 digest1_vec[5];

  for(int j=0; j<5; j++) {
    digest1_vec[j] = (uint4)(digest1[j], digest1[j], digest1[j], digest1[j]);
  }

  // Statische Woerter einmalig vor der Schleife in den Vektor packen
  for(int j=0; j < swapendianness_start; j++) {
    hashstring_vec[j] = (uint4)(hashstring0[j], hashstring1[j], hashstring2[j], hashstring3[j]);
  }

  // Statische Woerter NACH dem Counter einmalig packen (0x80, Nullen, Laenge)
  for(int j=counter_word_end + 1; j < 16; j++) {
    hashstring_vec[j] = (uint4)(hashstring0[j], hashstring1[j], hashstring2[j], hashstring3[j]);
  }

  // --- PARTIAL ROUND PRECOMPUTATION ---
  // Die ersten 8 SHA-1 Runden sind komplett statisch (W0-W7 aendern sich nie)
  // und werden hier einmalig pro Thread vorberechnet.
  uint4 pre_a = digest1_vec[0];
  uint4 pre_b = digest1_vec[1];
  uint4 pre_c = digest1_vec[2];
  uint4 pre_d = digest1_vec[3];
  uint4 pre_e = digest1_vec[4];

  uint4 pre_w0 = hashstring_vec[0];
  uint4 pre_w1 = hashstring_vec[1];
  uint4 pre_w2 = hashstring_vec[2];
  uint4 pre_w3 = hashstring_vec[3];
  uint4 pre_w4 = hashstring_vec[4];
  uint4 pre_w5 = hashstring_vec[5];
  uint4 pre_w6 = hashstring_vec[6];
  uint4 pre_w7 = hashstring_vec[7];

  #undef K
  #define K SHA1C00

  SHA1_STEP_VEC4 (SHA1_F0o, pre_a, pre_b, pre_c, pre_d, pre_e, pre_w0);
  SHA1_STEP_VEC4 (SHA1_F0o, pre_e, pre_a, pre_b, pre_c, pre_d, pre_w1);
  SHA1_STEP_VEC4 (SHA1_F0o, pre_d, pre_e, pre_a, pre_b, pre_c, pre_w2);
  SHA1_STEP_VEC4 (SHA1_F0o, pre_c, pre_d, pre_e, pre_a, pre_b, pre_w3);
  SHA1_STEP_VEC4 (SHA1_F0o, pre_b, pre_c, pre_d, pre_e, pre_a, pre_w4);
  SHA1_STEP_VEC4 (SHA1_F0o, pre_a, pre_b, pre_c, pre_d, pre_e, pre_w5);
  SHA1_STEP_VEC4 (SHA1_F0o, pre_e, pre_a, pre_b, pre_c, pre_d, pre_w6);
  SHA1_STEP_VEC4 (SHA1_F0o, pre_d, pre_e, pre_a, pre_b, pre_c, pre_w7);
  // -------------------------------------

  bool target_found = false;

  for (ulong it = 0; it < chunk; it++) {
    // NUR die Woerter packen, in denen der Counter tatsaechlich liegt
    for(int j = swapendianness_start; j <= counter_word_end; j++) {
      hashstring_vec[j] = (uint4)(hashstring0[j], hashstring1[j], hashstring2[j], hashstring3[j]);
    }

    // Vorberechneten Zustand und rohen Mid-State uebergeben
    if (sha1_64_check_vec4_r8(hashstring_vec, digest1_vec, pre_a, pre_b, pre_c, pre_d, pre_e, targetdifficulty)) {
      target_found = true;
    }

    // Counter separat erhoehen
    increaseStringCounterBE((uchar*)hashstring0, identity_length_snd_block + clen0 - 1);
    increaseStringCounterBE((uchar*)hashstring1, identity_length_snd_block + clen1 - 1);
    increaseStringCounterBE((uchar*)hashstring2, identity_length_snd_block + clen2 - 1);
    increaseStringCounterBE((uchar*)hashstring3, identity_length_snd_block + clen3 - 1);
  }

  results[gid] = target_found;
}
)"


R"(
__kernel void TeamSpeakHasher2 (const ulong startcounter,
                                const uint iterations,
                                const uchar targetdifficulty,
                                const __global uchar* identity,
                                const uint identity_length,
                                __global uchar *results)
{
  const int gid = get_global_id(0);
  const uint identity_length_snd_block = identity_length-64;

  u32 hashstring[32];
  
  for (int i=0; i<64; i++) {
    ((uchar*)hashstring)[i] = identity[i];
  }

  // we hash the first block 
  u32 digest1[5];
  digest1[0] = SHA1M_A;
  digest1[1] = SHA1M_B;
  digest1[2] = SHA1M_C;
  digest1[3] = SHA1M_D;
  digest1[4] = SHA1M_E;
  for (int j = 0; j<16; j++) {
    hashstring[j] = swap_uint(((u32*)hashstring)[j]);
  }
  sha1_64(hashstring, digest1);

  for (int i=0; i<identity_length_snd_block; i++) {
    ((uchar*)hashstring)[i] = identity[i+64];
  }

  for (int i=identity_length_snd_block; i<128; i++) {
    ((uchar*)hashstring)[i] = 0;
  }
   
  for (int j = 0; j<identity_length_snd_block/4; j++) {
    hashstring[j] = swap_uint(hashstring[j]);
  }

  const int swapendianness_start = identity_length_snd_block/4;

  // --- 4-Lane Vektorisierung ---
  ulong chunk = iterations / 4;

  ulong currentcounter0 = startcounter + gid * iterations;
  ulong currentcounter1 = startcounter + gid * iterations + chunk;
  ulong currentcounter2 = startcounter + gid * iterations + 2 * chunk;
  ulong currentcounter3 = startcounter + gid * iterations + 3 * chunk;

  u32 hashstring0[32]; u32 hashstring1[32]; u32 hashstring2[32]; u32 hashstring3[32];

  for(int i=0; i<32; i++) {
    hashstring0[i] = hashstring[i]; hashstring1[i] = hashstring[i];
    hashstring2[i] = hashstring[i]; hashstring3[i] = hashstring[i];
  }

  // Counter-Strings und Padding fuer alle 4 Lanes generieren
  ulong clen0 = countertostring(((uchar*)hashstring0)+identity_length_snd_block, currentcounter0);
  ((uchar*)hashstring0)[identity_length_snd_block+clen0] = 0x80;
  *((ulong*)(((uchar*)hashstring0)+64+56)) = swap_ulong(8*((ulong)identity_length+clen0));

  ulong clen1 = countertostring(((uchar*)hashstring1)+identity_length_snd_block, currentcounter1);
  ((uchar*)hashstring1)[identity_length_snd_block+clen1] = 0x80;
  *((ulong*)(((uchar*)hashstring1)+64+56)) = swap_ulong(8*((ulong)identity_length+clen1));

  ulong clen2 = countertostring(((uchar*)hashstring2)+identity_length_snd_block, currentcounter2);
  ((uchar*)hashstring2)[identity_length_snd_block+clen2] = 0x80;
  *((ulong*)(((uchar*)hashstring2)+64+56)) = swap_ulong(8*((ulong)identity_length+clen2));

  ulong clen3 = countertostring(((uchar*)hashstring3)+identity_length_snd_block, currentcounter3);
  ((uchar*)hashstring3)[identity_length_snd_block+clen3] = 0x80;
  *((ulong*)(((uchar*)hashstring3)+64+56)) = swap_ulong(8*((ulong)identity_length+clen3));

  // Erstes Wort des zweiten Blocks zu Big-Endian (nur 0x00000000 oder 0x80000000)
  hashstring0[16] = swap_uint(hashstring0[16]); hashstring1[16] = swap_uint(hashstring1[16]);
  hashstring2[16] = swap_uint(hashstring2[16]); hashstring3[16] = swap_uint(hashstring3[16]);

  // Laenge einmalig zu Big-Endian
  hashstring0[30] = swap_uint(hashstring0[30]); hashstring0[31] = swap_uint(hashstring0[31]);
  hashstring1[30] = swap_uint(hashstring1[30]); hashstring1[31] = swap_uint(hashstring1[31]);
  hashstring2[30] = swap_uint(hashstring2[30]); hashstring2[31] = swap_uint(hashstring2[31]);
  hashstring3[30] = swap_uint(hashstring3[30]); hashstring3[31] = swap_uint(hashstring3[31]);

  // Endianness-Swap fuer Counter-Bereich im ersten Block
  for(int j=swapendianness_start; j<16; j++) {
    hashstring0[j] = swap_uint(hashstring0[j]); hashstring1[j] = swap_uint(hashstring1[j]);
    hashstring2[j] = swap_uint(hashstring2[j]); hashstring3[j] = swap_uint(hashstring3[j]);
  }

  // Berechne das letzte Wort in Block 1, das sich durch den Counter aendert
  int max_clen = (int)max(max(clen0, clen1), max(clen2, clen3));
  int counter_word_end = min((int)((identity_length_snd_block + max_clen - 1) / 4), 15);

  // Vektor-Arrays vorbereiten (Block 1: 16 Woerter, Block 2: 16 Woerter)
  uint4 hashstring_vec_b1[16];
  uint4 hashstring_vec_b2[16];
  uint4 digest1_vec[5];

  for(int j=0; j<5; j++) {
    digest1_vec[j] = (uint4)(digest1[j], digest1[j], digest1[j], digest1[j]);
  }

  // Statische Woerter von Block 1 einmalig packen
  for(int j=0; j < swapendianness_start; j++) {
    hashstring_vec_b1[j] = (uint4)(hashstring0[j], hashstring1[j], hashstring2[j], hashstring3[j]);
  }

  // Statische Woerter NACH dem Counter in Block 1 einmalig packen
  for(int j=counter_word_end + 1; j < 16; j++) {
    hashstring_vec_b1[j] = (uint4)(hashstring0[j], hashstring1[j], hashstring2[j], hashstring3[j]);
  }

  // Block 2 statische Woerter packen (Woerter 16-31 -> Index 0-15 im Block2-Vektor)
  // Wort 16 (0x80-Padding) und Woerter 17-29 (Nullen) und 30-31 (Laenge) aendern sich nicht
  for(int j=0; j<16; j++) {
    hashstring_vec_b2[j] = (uint4)(hashstring0[16+j], hashstring1[16+j], hashstring2[16+j], hashstring3[16+j]);
  }

  bool target_found = false;

  for (ulong it = 0; it < chunk; it++) {
    uint4 digest2_vec[5];
    for (int j=0; j<5; j++) { digest2_vec[j] = digest1_vec[j]; }

    // Block 1: NUR die Woerter packen, in denen der Counter tatsaechlich liegt
    for(int j=swapendianness_start; j <= counter_word_end; j++) {
      hashstring_vec_b1[j] = (uint4)(hashstring0[j], hashstring1[j], hashstring2[j], hashstring3[j]);
    }

    // Volle SHA1-Kompression fuer Block 1 (liefert Mid-State)
    sha1_64_vec4(hashstring_vec_b1, digest2_vec);

    // Early-Exit Check auf Block 2
    if (sha1_64_check_vec4(hashstring_vec_b2, digest2_vec, targetdifficulty)) {
      target_found = true;
    }

    // Counter separat erhoehen
    increaseStringCounterBE((uchar*)hashstring0, identity_length_snd_block + clen0 - 1);
    increaseStringCounterBE((uchar*)hashstring1, identity_length_snd_block + clen1 - 1);
    increaseStringCounterBE((uchar*)hashstring2, identity_length_snd_block + clen2 - 1);
    increaseStringCounterBE((uchar*)hashstring3, identity_length_snd_block + clen3 - 1);
  }

  results[gid] = target_found;
}
)";

#endif
