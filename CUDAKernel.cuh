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
#ifndef CUDAKERNEL_CUH_
#define CUDAKERNEL_CUH_

// ============================================================================
// Custom uint4 type for CUDA (tsh_uint4)
// ============================================================================

struct tsh_uint4 {
  unsigned int x, y, z, w;
};

__device__ __forceinline__ tsh_uint4 make_tsh_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{
    tsh_uint4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r;
}

__device__ __forceinline__ tsh_uint4 operator+(const tsh_uint4& a, const tsh_uint4& b) {
  tsh_uint4 r;
  r.x = a.x + b.x; r.y = a.y + b.y; r.z = a.z + b.z; r.w = a.w + b.w;
  return r;
}

__device__ __forceinline__ tsh_uint4 operator^(const tsh_uint4& a, const tsh_uint4& b) {
  tsh_uint4 r;
  r.x = a.x ^ b.x; r.y = a.y ^ b.y; r.z = a.z ^ b.z; r.w = a.w ^ b.w;
  return r;
}

__device__ __forceinline__ tsh_uint4 operator&(const tsh_uint4& a, const tsh_uint4& b) {
  tsh_uint4 r;
  r.x = a.x & b.x; r.y = a.y & b.y; r.z = a.z & b.z; r.w = a.w & b.w;
  return r;
}

__device__ __forceinline__ tsh_uint4 operator|(const tsh_uint4& a, const tsh_uint4& b) {
  tsh_uint4 r;
  r.x = a.x | b.x; r.y = a.y | b.y; r.z = a.z | b.z; r.w = a.w | b.w;
  return r;
}

__device__ __forceinline__ tsh_uint4 operator~(const tsh_uint4& a) {
  tsh_uint4 r;
  r.x = ~a.x; r.y = ~a.y; r.z = ~a.z; r.w = ~a.w;
  return r;
}

__device__ __forceinline__ tsh_uint4& operator+=(tsh_uint4& a, const tsh_uint4& b) {
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
  return a;
}

__device__ __forceinline__ tsh_uint4 make_tsh_uint4(unsigned int v) {
  tsh_uint4 r;
  r.x = v; r.y = v; r.z = v; r.w = v;
  return r;
}

// ============================================================================
// Rotation primitives
// ============================================================================

__device__ __forceinline__ unsigned int rotl32(unsigned int val, unsigned int n) {
  return __funnelshift_l(val, val, n);
}

__device__ __forceinline__ tsh_uint4 rotl32_vec4(tsh_uint4 v, unsigned int n) {
  tsh_uint4 r;
  r.x = rotl32(v.x, n);
  r.y = rotl32(v.y, n);
  r.z = rotl32(v.z, n);
  r.w = rotl32(v.w, n);
  return r;
}

// ============================================================================
// SHA-1 constants
// ============================================================================

typedef enum sha1_constants {
  SHA1M_A  = 0x67452301,
  SHA1M_B  = 0xefcdab89,
  SHA1M_C  = 0x98badcfe,
  SHA1M_D  = 0x10325476,
  SHA1M_E  = 0xc3d2e1f0,

  SHA1C00  = 0x5a827999,
  SHA1C01  = 0x6ed9eba1,
  SHA1C02  = 0x8f1bbcdc,
  SHA1C03  = 0xca62c1d6u
} sha1_constants_t;

// ============================================================================
// LOP3.LUT PTX inline functions for SHA-1 round functions
// ============================================================================

// Ch(x,y,z) = z ^ (x & (y ^ z))  -- truth table 0xCA
__device__ __forceinline__ unsigned int lop3_0xCA(unsigned int a, unsigned int b, unsigned int c) {
  unsigned int r;
  asm("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

// Parity(x,y,z) = x ^ y ^ z  -- truth table 0x96
__device__ __forceinline__ unsigned int lop3_0x96(unsigned int a, unsigned int b, unsigned int c) {
  unsigned int r;
  asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

// Maj(x,y,z) = (x & y) | (z & (x ^ y))  -- truth table 0xE8
__device__ __forceinline__ unsigned int lop3_0xE8(unsigned int a, unsigned int b, unsigned int c) {
  unsigned int r;
  asm("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

// ============================================================================
// Vectorized LOP3 wrappers for tsh_uint4 (component-wise)
// ============================================================================

__device__ __forceinline__ tsh_uint4 SHA1_F0_vec4(tsh_uint4 x, tsh_uint4 y, tsh_uint4 z) {
  tsh_uint4 r;
  r.x = lop3_0xCA(x.x, y.x, z.x);
  r.y = lop3_0xCA(x.y, y.y, z.y);
  r.z = lop3_0xCA(x.z, y.z, z.z);
  r.w = lop3_0xCA(x.w, y.w, z.w);
  return r;
}

__device__ __forceinline__ tsh_uint4 SHA1_F1_vec4(tsh_uint4 x, tsh_uint4 y, tsh_uint4 z) {
  tsh_uint4 r;
  r.x = lop3_0x96(x.x, y.x, z.x);
  r.y = lop3_0x96(x.y, y.y, z.y);
  r.z = lop3_0x96(x.z, y.z, z.z);
  r.w = lop3_0x96(x.w, y.w, z.w);
  return r;
}

__device__ __forceinline__ tsh_uint4 SHA1_F2_vec4(tsh_uint4 x, tsh_uint4 y, tsh_uint4 z) {
  tsh_uint4 r;
  r.x = lop3_0xE8(x.x, y.x, z.x);
  r.y = lop3_0xE8(x.y, y.y, z.y);
  r.z = lop3_0xE8(x.z, y.z, z.z);
  r.w = lop3_0xE8(x.w, y.w, z.w);
  return r;
}

// ============================================================================
// Overloaded SHA-1 round functions (scalar + vector dispatch)
// ============================================================================

// Scalar overloads
__device__ __forceinline__ unsigned int SHA1_F0o(unsigned int x, unsigned int y, unsigned int z) {
  return lop3_0xCA(x, y, z);
}

__device__ __forceinline__ unsigned int SHA1_F1(unsigned int x, unsigned int y, unsigned int z) {
  return lop3_0x96(x, y, z);
}

__device__ __forceinline__ unsigned int SHA1_F2o(unsigned int x, unsigned int y, unsigned int z) {
  return lop3_0xE8(x, y, z);
}

// Vector overloads
__device__ __forceinline__ tsh_uint4 SHA1_F0o(tsh_uint4 x, tsh_uint4 y, tsh_uint4 z) {
  return SHA1_F0_vec4(x, y, z);
}

__device__ __forceinline__ tsh_uint4 SHA1_F1(tsh_uint4 x, tsh_uint4 y, tsh_uint4 z) {
  return SHA1_F1_vec4(x, y, z);
}

__device__ __forceinline__ tsh_uint4 SHA1_F2o(tsh_uint4 x, tsh_uint4 y, tsh_uint4 z) {
  return SHA1_F2_vec4(x, y, z);
}

// ============================================================================
// SHA-1 step macros
// ============================================================================

#define SHA1_STEP(f, a, b, c, d, e, x)  \
{                                        \
  e += K;                                \
  e += x;                                \
  e += f(b, c, d);                       \
  e += rotl32(a, 5u);                    \
  b  = rotl32(b, 30u);                   \
}

#define SHA1_STEP_VEC4(f, a, b, c, d, e, x)  \
{                                             \
  e += make_tsh_uint4(K);                     \
  e += x;                                     \
  e += f(b, c, d);                            \
  e += rotl32_vec4(a, 5u);                    \
  b  = rotl32_vec4(b, 30u);                   \
}

// ============================================================================
// Byte swap functions
// ============================================================================

__device__ __forceinline__ unsigned int swap_uint(unsigned int val) {
  return __byte_perm(val, 0, 0x0123);
}

__device__ __forceinline__ unsigned long long swap_ulong(unsigned long long val) {
  unsigned int lo = (unsigned int)(val);
  unsigned int hi = (unsigned int)(val >> 32);
  unsigned int tlo = __byte_perm(lo, 0, 0x0123);
  unsigned int thi = __byte_perm(hi, 0, 0x0123);
  return ((unsigned long long)tlo << 32) | (unsigned long long)thi;
}

// ============================================================================
// Counter to string conversion
// ============================================================================

__device__ __forceinline__ unsigned long long countertostring(unsigned char* dst, unsigned long long c) {
  if (c == 0) { *dst = '0'; return 1; }
  unsigned char* dst0 = dst;
  unsigned char* dst1 = dst;
  unsigned long long counterlength = 0;
  while (c) {
    unsigned char currentdigit = (unsigned char)(c % 10);
    *dst1 = '0' + currentdigit;
    dst1++;
    counterlength++;
    c = c / 10;
  }
  // invert string
  dst1--;
  while (dst0 < dst1) {
    unsigned char tmp = *dst0;
    *dst0 = *dst1;
    *dst1 = tmp;
    dst0++;
    dst1--;
  }
  return counterlength;
}

// ============================================================================
// Increment counter string directly in big-endian converted array
// ============================================================================

__device__ __forceinline__ void increaseStringCounterBE(unsigned char* hashstring_bytes, int counter_end_idx) {
  bool add = 1;
  int current_idx = counter_end_idx;
  while (add) {
    // Map logical string index to physical big-endian memory index
    int phys_idx = (current_idx & ~3) | (3 - (current_idx & 3));
    unsigned char currentdigit = hashstring_bytes[phys_idx];
    if (currentdigit == '9') {
      hashstring_bytes[phys_idx] = '0';
      current_idx--;
    } else {
      hashstring_bytes[phys_idx] = currentdigit + 1;
      add = 0;
    }
  }
}

__device__ __forceinline__ void inc_ascii_reg(
    unsigned int& w8, unsigned int& w9, unsigned int& wa, unsigned int& wb,
    unsigned int& wc, unsigned int& wd, unsigned int& we, unsigned int& wf,
    int last_idx)
{
    bool carry = true;

    // Wir wandern durch die 8 Zähler-Wörter (von hinten nach vorne)
#pragma unroll
    for (int i = 15; i >= 8; i--)
    {
        // Wort nur anfassen, wenn der Zähler überhaupt so weit reicht
        if (i * 4 <= last_idx && carry)
        {
            unsigned int w;
            if (i == 15) w = wf;
            else if (i == 14) w = we;
            else if (i == 13) w = wd;
            else if (i == 12) w = wc;
            else if (i == 11) w = wb;
            else if (i == 10) w = wa;
            else if (i == 9)  w = w9;
            else if (i == 8)  w = w8;

            // Die 4 Bytes innerhalb des 32-Bit Wortes von hinten nach vorne prüfen
        #pragma unroll
            for (int b = 3; b >= 0; b--)
            {
                int global_b = i * 4 + b; // Berechnet den absoluten String-Index

                if (global_b <= last_idx && carry)
                {
                    unsigned int shift = b * 8;
                    unsigned int val = (w >> shift) & 0xFF; // Isoliert das Byte

                    if (val == '9')
                    {
                        // Overflow auf '0', carry bleibt true für das nächste Byte
                        w = (w & ~(0xFF << shift)) | ('0' << shift);
                    }
                    else
                    {
                        // Zähler um 1 erhöhen, carry beenden
                        w += (1 << shift);
                        carry = false;
                    }
                }
            }

            // Zurück in das richtige Register schreiben
            if (i == 15) wf = w;
            else if (i == 14) we = w;
            else if (i == 13) wd = w;
            else if (i == 12) wc = w;
            else if (i == 11) wb = w;
            else if (i == 10) wa = w;
            else if (i == 9)  w9 = w;
            else if (i == 8)  w8 = w;
        }
    }
}

// ============================================================================
// Scalar SHA-1 compression (single block, 80 rounds)
// ============================================================================

__device__ __forceinline__ void sha1_64(unsigned int block[16], unsigned int digest[5]) {
  unsigned int a = digest[0];
  unsigned int b = digest[1];
  unsigned int c = digest[2];
  unsigned int d = digest[3];
  unsigned int e = digest[4];

  unsigned int w0_t = block[ 0];
  unsigned int w1_t = block[ 1];
  unsigned int w2_t = block[ 2];
  unsigned int w3_t = block[ 3];
  unsigned int w4_t = block[ 4];
  unsigned int w5_t = block[ 5];
  unsigned int w6_t = block[ 6];
  unsigned int w7_t = block[ 7];
  unsigned int w8_t = block[ 8];
  unsigned int w9_t = block[ 9];
  unsigned int wa_t = block[10];
  unsigned int wb_t = block[11];
  unsigned int wc_t = block[12];
  unsigned int wd_t = block[13];
  unsigned int we_t = block[14];
  unsigned int wf_t = block[15];

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

  w0_t = rotl32((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP (SHA1_F0o, e, a, b, c, d, w0_t);
  w1_t = rotl32((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP (SHA1_F0o, d, e, a, b, c, w1_t);
  w2_t = rotl32((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP (SHA1_F0o, c, d, e, a, b, w2_t);
  w3_t = rotl32((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP (SHA1_F0o, b, c, d, e, a, w3_t);

  #undef K
  #define K SHA1C01

  w4_t = rotl32((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, w4_t);
  w5_t = rotl32((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, w5_t);
  w6_t = rotl32((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, w6_t);
  w7_t = rotl32((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, w7_t);
  w8_t = rotl32((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, w8_t);
  w9_t = rotl32((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, w9_t);
  wa_t = rotl32((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, wa_t);
  wb_t = rotl32((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, wb_t);
  wc_t = rotl32((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, wc_t);
  wd_t = rotl32((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, wd_t);
  we_t = rotl32((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, we_t);
  wf_t = rotl32((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, wf_t);
  w0_t = rotl32((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, w0_t);
  w1_t = rotl32((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, w1_t);
  w2_t = rotl32((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, w2_t);
  w3_t = rotl32((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, w3_t);
  w4_t = rotl32((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, w4_t);
  w5_t = rotl32((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, w5_t);
  w6_t = rotl32((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, w6_t);
  w7_t = rotl32((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, w7_t);

  #undef K
  #define K SHA1C02

  w8_t = rotl32((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP (SHA1_F2o, a, b, c, d, e, w8_t);
  w9_t = rotl32((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP (SHA1_F2o, e, a, b, c, d, w9_t);
  wa_t = rotl32((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP (SHA1_F2o, d, e, a, b, c, wa_t);
  wb_t = rotl32((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP (SHA1_F2o, c, d, e, a, b, wb_t);
  wc_t = rotl32((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP (SHA1_F2o, b, c, d, e, a, wc_t);
  wd_t = rotl32((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP (SHA1_F2o, a, b, c, d, e, wd_t);
  we_t = rotl32((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP (SHA1_F2o, e, a, b, c, d, we_t);
  wf_t = rotl32((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP (SHA1_F2o, d, e, a, b, c, wf_t);
  w0_t = rotl32((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP (SHA1_F2o, c, d, e, a, b, w0_t);
  w1_t = rotl32((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP (SHA1_F2o, b, c, d, e, a, w1_t);
  w2_t = rotl32((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP (SHA1_F2o, a, b, c, d, e, w2_t);
  w3_t = rotl32((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP (SHA1_F2o, e, a, b, c, d, w3_t);
  w4_t = rotl32((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP (SHA1_F2o, d, e, a, b, c, w4_t);
  w5_t = rotl32((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP (SHA1_F2o, c, d, e, a, b, w5_t);
  w6_t = rotl32((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP (SHA1_F2o, b, c, d, e, a, w6_t);
  w7_t = rotl32((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP (SHA1_F2o, a, b, c, d, e, w7_t);
  w8_t = rotl32((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP (SHA1_F2o, e, a, b, c, d, w8_t);
  w9_t = rotl32((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP (SHA1_F2o, d, e, a, b, c, w9_t);
  wa_t = rotl32((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP (SHA1_F2o, c, d, e, a, b, wa_t);
  wb_t = rotl32((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP (SHA1_F2o, b, c, d, e, a, wb_t);

  #undef K
  #define K SHA1C03

  wc_t = rotl32((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, wc_t);
  wd_t = rotl32((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, wd_t);
  we_t = rotl32((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, we_t);
  wf_t = rotl32((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, wf_t);
  w0_t = rotl32((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, w0_t);
  w1_t = rotl32((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, w1_t);
  w2_t = rotl32((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, w2_t);
  w3_t = rotl32((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, w3_t);
  w4_t = rotl32((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, w4_t);
  w5_t = rotl32((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, w5_t);
  w6_t = rotl32((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, w6_t);
  w7_t = rotl32((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, w7_t);
  w8_t = rotl32((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, w8_t);
  w9_t = rotl32((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, w9_t);
  wa_t = rotl32((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, wa_t);
  wb_t = rotl32((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP (SHA1_F1, a, b, c, d, e, wb_t);
  wc_t = rotl32((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP (SHA1_F1, e, a, b, c, d, wc_t);
  wd_t = rotl32((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP (SHA1_F1, d, e, a, b, c, wd_t);
  we_t = rotl32((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP (SHA1_F1, c, d, e, a, b, we_t);
  wf_t = rotl32((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP (SHA1_F1, b, c, d, e, a, wf_t);

  digest[0] += a;
  digest[1] += b;
  digest[2] += c;
  digest[3] += d;
  digest[4] += e;
}

// ============================================================================
// Vectorized full SHA-1 compression (4 lanes parallel, for block 1 in double-block kernel)
// ============================================================================

__device__ __forceinline__ void sha1_64_vec4(
    tsh_uint4 w0_t, tsh_uint4 w1_t, tsh_uint4 w2_t, tsh_uint4 w3_t,
    tsh_uint4 w4_t, tsh_uint4 w5_t, tsh_uint4 w6_t, tsh_uint4 w7_t,
    tsh_uint4 w8_t, tsh_uint4 w9_t, tsh_uint4 wa_t, tsh_uint4 wb_t,
    tsh_uint4 wc_t, tsh_uint4 wd_t, tsh_uint4 we_t, tsh_uint4 wf_t,
    tsh_uint4& out_a, tsh_uint4& out_b, tsh_uint4& out_c, tsh_uint4& out_d, tsh_uint4& out_e) 
{
    tsh_uint4 a = out_a;
    tsh_uint4 b = out_b;
    tsh_uint4 c = out_c;
    tsh_uint4 d = out_d;
    tsh_uint4 e = out_e;

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

  w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, e, a, b, c, d, w0_t);
  w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, d, e, a, b, c, w1_t);
  w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, c, d, e, a, b, w2_t);
  w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, b, c, d, e, a, w3_t);

  #undef K
  #define K SHA1C01

  w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w4_t);
  w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w5_t);
  w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w6_t);
  w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w7_t);
  w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w8_t);
  w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w9_t);
  wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wa_t);
  wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, wb_t);
  wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, wc_t);
  wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, wd_t);
  we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, we_t);
  wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wf_t);
  w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w0_t);
  w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w1_t);
  w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w2_t);
  w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w3_t);
  w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w4_t);
  w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w5_t);
  w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w6_t);
  w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w7_t);

  #undef K
  #define K SHA1C02

  w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, w8_t);
  w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, w9_t);
  wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, wa_t);
  wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, wb_t);
  wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, wc_t);
  wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, wd_t);
  we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, we_t);
  wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, wf_t);
  w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, w0_t);
  w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, w1_t);
  w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, w2_t);
  w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, w3_t);
  w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, w4_t);
  w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, w5_t);
  w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, w6_t);
  w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, w7_t);
  w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, w8_t);
  w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, w9_t);
  wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, wa_t);
  wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, wb_t);

  #undef K
  #define K SHA1C03

  wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, wc_t);
  wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wd_t);
  we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, we_t);
  wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, wf_t);
  w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w0_t);
  w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w1_t);
  w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w2_t);
  w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w3_t);
  w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w4_t);
  w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w5_t);
  w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w6_t);
  w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w7_t);
  w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w8_t);
  w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w9_t);
  wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, wa_t);
  wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, wb_t);
  wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wc_t);
  wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, wd_t);
  we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, we_t);
  wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, wf_t);

  out_a += a;
  out_b += b;
  out_c += c;
  out_d += d;
  out_e += e;
}

// ============================================================================
// Vectorized early-exit SHA-1 (4 hashes parallel)
// ============================================================================

__device__ __forceinline__ bool sha1_64_check_vec4(
    tsh_uint4 w0_t, tsh_uint4 w1_t, tsh_uint4 w2_t, tsh_uint4 w3_t,
    tsh_uint4 w4_t, tsh_uint4 w5_t, tsh_uint4 w6_t, tsh_uint4 w7_t,
    tsh_uint4 w8_t, tsh_uint4 w9_t, tsh_uint4 wa_t, tsh_uint4 wb_t,
    tsh_uint4 wc_t, tsh_uint4 wd_t, tsh_uint4 we_t, tsh_uint4 wf_t,
    tsh_uint4 dig0, tsh_uint4 dig1, tsh_uint4 dig2, tsh_uint4 dig3, tsh_uint4 dig4, 
    unsigned char targetdifficulty) 
{
    tsh_uint4 a = dig0;
    tsh_uint4 b = dig1;
    tsh_uint4 c = dig2;
    tsh_uint4 d = dig3;
    tsh_uint4 e = dig4;

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

  w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, e, a, b, c, d, w0_t);
  w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, d, e, a, b, c, w1_t);
  w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, c, d, e, a, b, w2_t);
  w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, b, c, d, e, a, w3_t);

  #undef K
  #define K SHA1C01

  w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w4_t);
  w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w5_t);
  w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w6_t);
  w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w7_t);
  w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w8_t);
  w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w9_t);
  wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wa_t);
  wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, wb_t);
  wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, wc_t);
  wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, wd_t);
  we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, we_t);
  wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wf_t);
  w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w0_t);
  w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w1_t);
  w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w2_t);
  w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w3_t);
  w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w4_t);
  w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w5_t);
  w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w6_t);
  w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w7_t);

  #undef K
  #define K SHA1C02

  w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, w8_t);
  w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, w9_t);
  wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, wa_t);
  wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, wb_t);
  wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, wc_t);
  wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, wd_t);
  we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, we_t);
  wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, wf_t);
  w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, w0_t);
  w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, w1_t);
  w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, w2_t);
  w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, w3_t);
  w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, w4_t);
  w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, w5_t);
  w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, w6_t);
  w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, w7_t);
  w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, w8_t);
  w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, w9_t);
  wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, wa_t);
  wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, wb_t);

  #undef K
  #define K SHA1C03

  wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, wc_t);
  wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wd_t);
  we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, we_t);
  wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, wf_t);
  w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w0_t);
  w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w1_t);
  w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w2_t);
  w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w3_t);
  w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w4_t);
  w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w5_t);
  w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w6_t);
  w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w7_t);
  w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w8_t);
  w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w9_t);
  wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, wa_t);
  wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, wb_t);
  wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wc_t);
  wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, wd_t);
  we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, we_t);
  wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, wf_t);

  // ---- CLZ-based early exit logic for 4 lanes ----
  tsh_uint4 final_h0 = dig0 + a;

  unsigned int z0 = __clz(final_h0.x);
  unsigned int z1 = __clz(final_h0.y);
  unsigned int z2 = __clz(final_h0.z);
  unsigned int z3 = __clz(final_h0.w);

  // Quick reject: if no lane has enough leading zeros in the first word
  unsigned int td_cap = min((unsigned int)targetdifficulty, 32u);
  if (max(max(z0, z1), max(z2, z3)) < td_cap) return false;

  // Promising -- check second word if needed
  tsh_uint4 final_h1 = dig1 + b;
  if (z0 == 32) z0 += __clz(final_h1.x);
  if (z1 == 32) z1 += __clz(final_h1.y);
  if (z2 == 32) z2 += __clz(final_h1.z);
  if (z3 == 32) z3 += __clz(final_h1.w);

  return (z0 >= targetdifficulty) || (z1 >= targetdifficulty) ||
         (z2 >= targetdifficulty) || (z3 >= targetdifficulty);
}

// ============================================================================
// Vectorized SHA-1 with partial round precomputation (start from round 8)
// ============================================================================

__device__ __forceinline__ bool sha1_64_check_vec4_r8(
    tsh_uint4 w8_t, tsh_uint4 w9_t, tsh_uint4 wa_t, tsh_uint4 wb_t,
    tsh_uint4 wc_t, tsh_uint4 wd_t, tsh_uint4 we_t, tsh_uint4 wf_t,
    tsh_uint4 pre_w0_t, tsh_uint4 pre_w1_t, tsh_uint4 pre_w2_t, tsh_uint4 pre_w3_t,
    tsh_uint4 pre_w4_t, tsh_uint4 pre_w5_t, tsh_uint4 pre_w6_t, tsh_uint4 pre_w7_t,
    tsh_uint4 dig0, tsh_uint4 dig1, tsh_uint4 dig2, tsh_uint4 dig3, tsh_uint4 dig4,
    tsh_uint4 pre_a, tsh_uint4 pre_b, tsh_uint4 pre_c, tsh_uint4 pre_d, tsh_uint4 pre_e,
    unsigned char targetdifficulty) {

  // Take pre-computed state from round 7
  tsh_uint4 a = pre_a;
  tsh_uint4 b = pre_b;
  tsh_uint4 c = pre_c;
  tsh_uint4 d = pre_d;
  tsh_uint4 e = pre_e;

  // Load all 16 words (w0-w7 needed for message expansion)
  tsh_uint4 w0_t = pre_w0_t;
  tsh_uint4 w1_t = pre_w1_t;
  tsh_uint4 w2_t = pre_w2_t;
  tsh_uint4 w3_t = pre_w3_t;
  tsh_uint4 w4_t = pre_w4_t;
  tsh_uint4 w5_t = pre_w5_t;
  tsh_uint4 w6_t = pre_w6_t;
  tsh_uint4 w7_t = pre_w7_t;

  #undef K
  #define K SHA1C00

  // ROUNDS 0-7 SKIPPED (pre-computed)

  // From round 8
  SHA1_STEP_VEC4 (SHA1_F0o, c, d, e, a, b, w8_t);
  SHA1_STEP_VEC4 (SHA1_F0o, b, c, d, e, a, w9_t);
  SHA1_STEP_VEC4 (SHA1_F0o, a, b, c, d, e, wa_t);
  SHA1_STEP_VEC4 (SHA1_F0o, e, a, b, c, d, wb_t);
  SHA1_STEP_VEC4 (SHA1_F0o, d, e, a, b, c, wc_t);
  SHA1_STEP_VEC4 (SHA1_F0o, c, d, e, a, b, wd_t);
  SHA1_STEP_VEC4 (SHA1_F0o, b, c, d, e, a, we_t);
  SHA1_STEP_VEC4 (SHA1_F0o, a, b, c, d, e, wf_t);

  w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, e, a, b, c, d, w0_t);
  w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, d, e, a, b, c, w1_t);
  w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, c, d, e, a, b, w2_t);
  w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F0o, b, c, d, e, a, w3_t);

  #undef K
  #define K SHA1C01

  w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w4_t);
  w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w5_t);
  w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w6_t);
  w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w7_t);
  w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w8_t);
  w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w9_t);
  wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wa_t);
  wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, wb_t);
  wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, wc_t);
  wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, wd_t);
  we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, we_t);
  wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wf_t);
  w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w0_t);
  w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w1_t);
  w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w2_t);
  w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w3_t);
  w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w4_t);
  w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w5_t);
  w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w6_t);
  w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w7_t);

  #undef K
  #define K SHA1C02

  w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, w8_t);
  w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, w9_t);
  wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, wa_t);
  wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, wb_t);
  wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, wc_t);
  wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, wd_t);
  we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, we_t);
  wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, wf_t);
  w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, w0_t);
  w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, w1_t);
  w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, w2_t);
  w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, w3_t);
  w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, w4_t);
  w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, w5_t);
  w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, w6_t);
  w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, a, b, c, d, e, w7_t);
  w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, e, a, b, c, d, w8_t);
  w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, d, e, a, b, c, w9_t);
  wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, c, d, e, a, b, wa_t);
  wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F2o, b, c, d, e, a, wb_t);

  #undef K
  #define K SHA1C03

  wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, wc_t);
  wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wd_t);
  we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, we_t);
  wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, wf_t);
  w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w0_t);
  w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w1_t);
  w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w2_t);
  w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w3_t);
  w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w4_t);
  w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, w5_t);
  w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, w6_t);
  w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, w7_t);
  w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, w8_t);
  w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, w9_t);
  wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, wa_t);
  wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, a, b, c, d, e, wb_t);
  wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, e, a, b, c, d, wc_t);
  wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, d, e, a, b, c, wd_t);
  we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, c, d, e, a, b, we_t);
  wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
  SHA1_STEP_VEC4 (SHA1_F1, b, c, d, e, a, wf_t);

  // ---- CLZ-based early exit logic for 4 lanes ----
  tsh_uint4 final_h0 = dig0 + a;

  unsigned int z0 = __clz(final_h0.x);
  unsigned int z1 = __clz(final_h0.y);
  unsigned int z2 = __clz(final_h0.z);
  unsigned int z3 = __clz(final_h0.w);

  // Quick reject: if no lane has enough leading zeros in the first word
  unsigned int td_cap = min((unsigned int)targetdifficulty, 32u);
  if (max(max(z0, z1), max(z2, z3)) < td_cap) return false;

  // Promising -- check second word if needed
  tsh_uint4 final_h1 = dig1 + b;
  if (z0 == 32) z0 += __clz(final_h1.x);
  if (z1 == 32) z1 += __clz(final_h1.y);
  if (z2 == 32) z2 += __clz(final_h1.z);
  if (z3 == 32) z3 += __clz(final_h1.w);

  return (z0 >= targetdifficulty) || (z1 >= targetdifficulty) ||
         (z2 >= targetdifficulty) || (z3 >= targetdifficulty);
}

// ============================================================================
// CUDA kernel: single SHA-1 block (fast phase) __launch_bounds__(256) 
// ============================================================================

__global__ void TeamSpeakHasher_cuda(
    unsigned long long startcounter,
    unsigned int iterations,
    unsigned char targetdifficulty,
    const unsigned char* __restrict__ identity,
    unsigned int identity_length,
    unsigned char* __restrict__ results)
{
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int identity_length_snd_block = identity_length - 64;

  unsigned int hashstring[16];

  for (int i = 0; i < 64; i++) {
    ((unsigned char*)hashstring)[i] = identity[i];
  }

  // Hash the first block
  unsigned int digest1[5];
  digest1[0] = SHA1M_A;
  digest1[1] = SHA1M_B;
  digest1[2] = SHA1M_C;
  digest1[3] = SHA1M_D;
  digest1[4] = SHA1M_E;
  for (int j = 0; j < 16; j++) {
    hashstring[j] = swap_uint(((unsigned int*)hashstring)[j]);
  }
  sha1_64(hashstring, digest1);

  for (int i = 0; i < (int)identity_length_snd_block; i++) {
    ((unsigned char*)hashstring)[i] = identity[i + 64];
  }

  for (int i = (int)identity_length_snd_block; i < 64; i++) {
    ((unsigned char*)hashstring)[i] = 0;
  }

  for (int j = 0; j < (int)identity_length_snd_block / 4; j++) {
    hashstring[j] = swap_uint(hashstring[j]);
  }

  const int swapendianness_start = identity_length_snd_block / 4;

  // --- 4-lane vectorization ---
  unsigned long long chunk = iterations / 4;

  unsigned long long currentcounter0 = startcounter + (unsigned long long)gid * iterations;
  unsigned long long currentcounter1 = startcounter + (unsigned long long)gid * iterations + chunk;
  unsigned long long currentcounter2 = startcounter + (unsigned long long)gid * iterations + 2 * chunk;
  unsigned long long currentcounter3 = startcounter + (unsigned long long)gid * iterations + 3 * chunk;

  unsigned int hashstring0[16]; unsigned int hashstring1[16];
  unsigned int hashstring2[16]; unsigned int hashstring3[16];

  for (int i = 0; i < 16; i++) {
    hashstring0[i] = hashstring[i]; hashstring1[i] = hashstring[i];
    hashstring2[i] = hashstring[i]; hashstring3[i] = hashstring[i];
  }

  // Counter strings and padding for all 4 lanes
  unsigned long long clen0 = countertostring(((unsigned char*)hashstring0) + identity_length_snd_block, currentcounter0);
  ((unsigned char*)hashstring0)[identity_length_snd_block + clen0] = 0x80;
  *((unsigned long long*)(((unsigned char*)hashstring0) + 56)) = swap_ulong(8 * ((unsigned long long)identity_length + clen0));

  unsigned long long clen1 = countertostring(((unsigned char*)hashstring1) + identity_length_snd_block, currentcounter1);
  ((unsigned char*)hashstring1)[identity_length_snd_block + clen1] = 0x80;
  *((unsigned long long*)(((unsigned char*)hashstring1) + 56)) = swap_ulong(8 * ((unsigned long long)identity_length + clen1));

  unsigned long long clen2 = countertostring(((unsigned char*)hashstring2) + identity_length_snd_block, currentcounter2);
  ((unsigned char*)hashstring2)[identity_length_snd_block + clen2] = 0x80;
  *((unsigned long long*)(((unsigned char*)hashstring2) + 56)) = swap_ulong(8 * ((unsigned long long)identity_length + clen2));

  unsigned long long clen3 = countertostring(((unsigned char*)hashstring3) + identity_length_snd_block, currentcounter3);
  ((unsigned char*)hashstring3)[identity_length_snd_block + clen3] = 0x80;
  *((unsigned long long*)(((unsigned char*)hashstring3) + 56)) = swap_ulong(8 * ((unsigned long long)identity_length + clen3));

  // Length to big-endian (once)
  hashstring0[14] = swap_uint(hashstring0[14]); hashstring0[15] = swap_uint(hashstring0[15]);
  hashstring1[14] = swap_uint(hashstring1[14]); hashstring1[15] = swap_uint(hashstring1[15]);
  hashstring2[14] = swap_uint(hashstring2[14]); hashstring2[15] = swap_uint(hashstring2[15]);
  hashstring3[14] = swap_uint(hashstring3[14]); hashstring3[15] = swap_uint(hashstring3[15]);

  // Endianness swap for counter region
  for (int j = swapendianness_start; j < 14; j++) {
    hashstring0[j] = swap_uint(hashstring0[j]); hashstring1[j] = swap_uint(hashstring1[j]);
    hashstring2[j] = swap_uint(hashstring2[j]); hashstring3[j] = swap_uint(hashstring3[j]);
  }

  // Compute last word that changes due to counter
  int max_clen = (int)max(max(clen0, clen1), max(clen2, clen3));
  int counter_word_end = (identity_length_snd_block + max_clen - 1) / 4;

  tsh_uint4 dig0 = make_tsh_uint4(digest1[0]);
  tsh_uint4 dig1 = make_tsh_uint4(digest1[1]);
  tsh_uint4 dig2 = make_tsh_uint4(digest1[2]);
  tsh_uint4 dig3 = make_tsh_uint4(digest1[3]);
  tsh_uint4 dig4 = make_tsh_uint4(digest1[4]);

  tsh_uint4 pre_a = dig0;
  tsh_uint4 pre_b = dig1;
  tsh_uint4 pre_c = dig2;
  tsh_uint4 pre_d = dig3;
  tsh_uint4 pre_e = dig4;

  // Statische Wörter (W0-W7) einmalig packen
  tsh_uint4 pre_w0 = make_tsh_uint4(hashstring0[0], hashstring1[0], hashstring2[0], hashstring3[0]);
  tsh_uint4 pre_w1 = make_tsh_uint4(hashstring0[1], hashstring1[1], hashstring2[1], hashstring3[1]);
  tsh_uint4 pre_w2 = make_tsh_uint4(hashstring0[2], hashstring1[2], hashstring2[2], hashstring3[2]);
  tsh_uint4 pre_w3 = make_tsh_uint4(hashstring0[3], hashstring1[3], hashstring2[3], hashstring3[3]);
  tsh_uint4 pre_w4 = make_tsh_uint4(hashstring0[4], hashstring1[4], hashstring2[4], hashstring3[4]);
  tsh_uint4 pre_w5 = make_tsh_uint4(hashstring0[5], hashstring1[5], hashstring2[5], hashstring3[5]);
  tsh_uint4 pre_w6 = make_tsh_uint4(hashstring0[6], hashstring1[6], hashstring2[6], hashstring3[6]);
  tsh_uint4 pre_w7 = make_tsh_uint4(hashstring0[7], hashstring1[7], hashstring2[7], hashstring3[7]);

  // Wir legen die hinteren Woerter (W8-W15) hier einmal komplett an
  // (Der statische Teil davon wird hier schon initialisiert)
  tsh_uint4 w8 = make_tsh_uint4(hashstring0[8], hashstring1[8], hashstring2[8], hashstring3[8]);
  tsh_uint4 w9 = make_tsh_uint4(hashstring0[9], hashstring1[9], hashstring2[9], hashstring3[9]);
  tsh_uint4 wa = make_tsh_uint4(hashstring0[10], hashstring1[10], hashstring2[10], hashstring3[10]);
  tsh_uint4 wb = make_tsh_uint4(hashstring0[11], hashstring1[11], hashstring2[11], hashstring3[11]);
  tsh_uint4 wc = make_tsh_uint4(hashstring0[12], hashstring1[12], hashstring2[12], hashstring3[12]);
  tsh_uint4 wd = make_tsh_uint4(hashstring0[13], hashstring1[13], hashstring2[13], hashstring3[13]);
  tsh_uint4 we = make_tsh_uint4(hashstring0[14], hashstring1[14], hashstring2[14], hashstring3[14]);
  tsh_uint4 wf = make_tsh_uint4(hashstring0[15], hashstring1[15], hashstring2[15], hashstring3[15]);

  // --- PARTIAL ROUND PRECOMPUTATION ---
  #undef K
  #define K SHA1C00

  SHA1_STEP_VEC4(SHA1_F0o, pre_a, pre_b, pre_c, pre_d, pre_e, pre_w0);
  SHA1_STEP_VEC4(SHA1_F0o, pre_e, pre_a, pre_b, pre_c, pre_d, pre_w1);
  SHA1_STEP_VEC4(SHA1_F0o, pre_d, pre_e, pre_a, pre_b, pre_c, pre_w2);
  SHA1_STEP_VEC4(SHA1_F0o, pre_c, pre_d, pre_e, pre_a, pre_b, pre_w3);
  SHA1_STEP_VEC4(SHA1_F0o, pre_b, pre_c, pre_d, pre_e, pre_a, pre_w4);
  SHA1_STEP_VEC4(SHA1_F0o, pre_a, pre_b, pre_c, pre_d, pre_e, pre_w5);
  SHA1_STEP_VEC4(SHA1_F0o, pre_e, pre_a, pre_b, pre_c, pre_d, pre_w6);
  SHA1_STEP_VEC4(SHA1_F0o, pre_d, pre_e, pre_a, pre_b, pre_c, pre_w7);

  bool target_found = false;

  // ====================================================================
  // 2. DIE HAUPTSCHLEIFE (Pure Register, kein Local Memory Spilling!)
  // ====================================================================
  #pragma unroll
  for (unsigned long long it = 0; it < chunk; it++)
  {

      // Wir packen in jedem Schleifendurchlauf NUR die Register neu, 
      // in denen der Counter tatsächlich liegen kann.
      if (8 >= swapendianness_start && 8 <= counter_word_end) w8 = make_tsh_uint4(hashstring0[8], hashstring1[8], hashstring2[8], hashstring3[8]);
      if (9 >= swapendianness_start && 9 <= counter_word_end) w9 = make_tsh_uint4(hashstring0[9], hashstring1[9], hashstring2[9], hashstring3[9]);
      if (10 >= swapendianness_start && 10 <= counter_word_end) wa = make_tsh_uint4(hashstring0[10], hashstring1[10], hashstring2[10], hashstring3[10]);
      if (11 >= swapendianness_start && 11 <= counter_word_end) wb = make_tsh_uint4(hashstring0[11], hashstring1[11], hashstring2[11], hashstring3[11]);
      if (12 >= swapendianness_start && 12 <= counter_word_end) wc = make_tsh_uint4(hashstring0[12], hashstring1[12], hashstring2[12], hashstring3[12]);
      if (13 >= swapendianness_start && 13 <= counter_word_end) wd = make_tsh_uint4(hashstring0[13], hashstring1[13], hashstring2[13], hashstring3[13]);
      if (14 >= swapendianness_start && 14 <= counter_word_end) we = make_tsh_uint4(hashstring0[14], hashstring1[14], hashstring2[14], hashstring3[14]);
      if (15 >= swapendianness_start && 15 <= counter_word_end) wf = make_tsh_uint4(hashstring0[15], hashstring1[15], hashstring2[15], hashstring3[15]);

      // Pass everything directly via registers
      if (sha1_64_check_vec4_r8(
          w8, w9, wa, wb, wc, wd, we, wf,
          pre_w0, pre_w1, pre_w2, pre_w3, pre_w4, pre_w5, pre_w6, pre_w7,
          dig0, dig1, dig2, dig3, dig4,
          pre_a, pre_b, pre_c, pre_d, pre_e,
          targetdifficulty))
      {
          target_found = true;
      }

      // Increment counters separately
      increaseStringCounterBE((unsigned char*)hashstring0, identity_length_snd_block + clen0 - 1);
      increaseStringCounterBE((unsigned char*)hashstring1, identity_length_snd_block + clen1 - 1);
      increaseStringCounterBE((unsigned char*)hashstring2, identity_length_snd_block + clen2 - 1);
      increaseStringCounterBE((unsigned char*)hashstring3, identity_length_snd_block + clen3 - 1);
  }

  results[gid] = target_found;
}

// ============================================================================
// CUDA kernel: double SHA-1 block (slow phase)
// ============================================================================

__global__ void TeamSpeakHasher2_cuda(
    unsigned long long startcounter,
    unsigned int iterations,
    unsigned char targetdifficulty,
    const unsigned char* __restrict__ identity,
    unsigned int identity_length,
    unsigned char* __restrict__ results)
{
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int identity_length_snd_block = identity_length - 64;

  unsigned int hashstring[32];

  for (int i = 0; i < 64; i++) {
    ((unsigned char*)hashstring)[i] = identity[i];
  }

  // Hash the first block
  unsigned int digest1[5];
  digest1[0] = SHA1M_A;
  digest1[1] = SHA1M_B;
  digest1[2] = SHA1M_C;
  digest1[3] = SHA1M_D;
  digest1[4] = SHA1M_E;
  for (int j = 0; j < 16; j++) {
    hashstring[j] = swap_uint(((unsigned int*)hashstring)[j]);
  }
  sha1_64(hashstring, digest1);

  for (int i = 0; i < (int)identity_length_snd_block; i++) {
    ((unsigned char*)hashstring)[i] = identity[i + 64];
  }

  for (int i = (int)identity_length_snd_block; i < 128; i++) {
    ((unsigned char*)hashstring)[i] = 0;
  }

  for (int j = 0; j < (int)identity_length_snd_block / 4; j++) {
    hashstring[j] = swap_uint(hashstring[j]);
  }

  const int swapendianness_start = identity_length_snd_block / 4;

  // --- 4-lane vectorization ---
  unsigned long long chunk = iterations / 4;

  unsigned long long currentcounter0 = startcounter + (unsigned long long)gid * iterations;
  unsigned long long currentcounter1 = startcounter + (unsigned long long)gid * iterations + chunk;
  unsigned long long currentcounter2 = startcounter + (unsigned long long)gid * iterations + 2 * chunk;
  unsigned long long currentcounter3 = startcounter + (unsigned long long)gid * iterations + 3 * chunk;

  unsigned int hashstring0[32]; unsigned int hashstring1[32];
  unsigned int hashstring2[32]; unsigned int hashstring3[32];

  for (int i = 0; i < 32; i++) {
    hashstring0[i] = hashstring[i]; hashstring1[i] = hashstring[i];
    hashstring2[i] = hashstring[i]; hashstring3[i] = hashstring[i];
  }

  // Counter strings and padding for all 4 lanes
  unsigned long long clen0 = countertostring(((unsigned char*)hashstring0) + identity_length_snd_block, currentcounter0);
  ((unsigned char*)hashstring0)[identity_length_snd_block + clen0] = 0x80;
  *((unsigned long long*)(((unsigned char*)hashstring0) + 64 + 56)) = swap_ulong(8 * ((unsigned long long)identity_length + clen0));

  unsigned long long clen1 = countertostring(((unsigned char*)hashstring1) + identity_length_snd_block, currentcounter1);
  ((unsigned char*)hashstring1)[identity_length_snd_block + clen1] = 0x80;
  *((unsigned long long*)(((unsigned char*)hashstring1) + 64 + 56)) = swap_ulong(8 * ((unsigned long long)identity_length + clen1));

  unsigned long long clen2 = countertostring(((unsigned char*)hashstring2) + identity_length_snd_block, currentcounter2);
  ((unsigned char*)hashstring2)[identity_length_snd_block + clen2] = 0x80;
  *((unsigned long long*)(((unsigned char*)hashstring2) + 64 + 56)) = swap_ulong(8 * ((unsigned long long)identity_length + clen2));

  unsigned long long clen3 = countertostring(((unsigned char*)hashstring3) + identity_length_snd_block, currentcounter3);
  ((unsigned char*)hashstring3)[identity_length_snd_block + clen3] = 0x80;
  *((unsigned long long*)(((unsigned char*)hashstring3) + 64 + 56)) = swap_ulong(8 * ((unsigned long long)identity_length + clen3));

  // First word of second block to big-endian (only 0x00000000 or 0x80000000)
  hashstring0[16] = swap_uint(hashstring0[16]); hashstring1[16] = swap_uint(hashstring1[16]);
  hashstring2[16] = swap_uint(hashstring2[16]); hashstring3[16] = swap_uint(hashstring3[16]);

  // Length to big-endian (once)
  hashstring0[30] = swap_uint(hashstring0[30]); hashstring0[31] = swap_uint(hashstring0[31]);
  hashstring1[30] = swap_uint(hashstring1[30]); hashstring1[31] = swap_uint(hashstring1[31]);
  hashstring2[30] = swap_uint(hashstring2[30]); hashstring2[31] = swap_uint(hashstring2[31]);
  hashstring3[30] = swap_uint(hashstring3[30]); hashstring3[31] = swap_uint(hashstring3[31]);

  // Endianness swap for counter region in first block
  for (int j = swapendianness_start; j < 16; j++) {
    hashstring0[j] = swap_uint(hashstring0[j]); hashstring1[j] = swap_uint(hashstring1[j]);
    hashstring2[j] = swap_uint(hashstring2[j]); hashstring3[j] = swap_uint(hashstring3[j]);
  }

  // Compute last word in block 1 that changes due to counter
  int max_clen = (int)max(max(clen0, clen1), max(clen2, clen3));
  int counter_word_end = min((int)((identity_length_snd_block + max_clen - 1) / 4), 15);

  // ====================================================================
  // 1. STATT ARRAYS: Variablen direkt in Hardwareregistern anlegen
  // ====================================================================
  tsh_uint4 dig1_0 = make_tsh_uint4(digest1[0]);
  tsh_uint4 dig1_1 = make_tsh_uint4(digest1[1]);
  tsh_uint4 dig1_2 = make_tsh_uint4(digest1[2]);
  tsh_uint4 dig1_3 = make_tsh_uint4(digest1[3]);
  tsh_uint4 dig1_4 = make_tsh_uint4(digest1[4]);

  // Block 1 statische Woerter initialisieren (W0 - W15)
  tsh_uint4 b1_w0 = make_tsh_uint4(hashstring0[0], hashstring1[0], hashstring2[0], hashstring3[0]);
  tsh_uint4 b1_w1 = make_tsh_uint4(hashstring0[1], hashstring1[1], hashstring2[1], hashstring3[1]);
  tsh_uint4 b1_w2 = make_tsh_uint4(hashstring0[2], hashstring1[2], hashstring2[2], hashstring3[2]);
  tsh_uint4 b1_w3 = make_tsh_uint4(hashstring0[3], hashstring1[3], hashstring2[3], hashstring3[3]);
  tsh_uint4 b1_w4 = make_tsh_uint4(hashstring0[4], hashstring1[4], hashstring2[4], hashstring3[4]);
  tsh_uint4 b1_w5 = make_tsh_uint4(hashstring0[5], hashstring1[5], hashstring2[5], hashstring3[5]);
  tsh_uint4 b1_w6 = make_tsh_uint4(hashstring0[6], hashstring1[6], hashstring2[6], hashstring3[6]);
  tsh_uint4 b1_w7 = make_tsh_uint4(hashstring0[7], hashstring1[7], hashstring2[7], hashstring3[7]);
  tsh_uint4 b1_w8 = make_tsh_uint4(hashstring0[8], hashstring1[8], hashstring2[8], hashstring3[8]);
  tsh_uint4 b1_w9 = make_tsh_uint4(hashstring0[9], hashstring1[9], hashstring2[9], hashstring3[9]);
  tsh_uint4 b1_wa = make_tsh_uint4(hashstring0[10], hashstring1[10], hashstring2[10], hashstring3[10]);
  tsh_uint4 b1_wb = make_tsh_uint4(hashstring0[11], hashstring1[11], hashstring2[11], hashstring3[11]);
  tsh_uint4 b1_wc = make_tsh_uint4(hashstring0[12], hashstring1[12], hashstring2[12], hashstring3[12]);
  tsh_uint4 b1_wd = make_tsh_uint4(hashstring0[13], hashstring1[13], hashstring2[13], hashstring3[13]);
  tsh_uint4 b1_we = make_tsh_uint4(hashstring0[14], hashstring1[14], hashstring2[14], hashstring3[14]);
  tsh_uint4 b1_wf = make_tsh_uint4(hashstring0[15], hashstring1[15], hashstring2[15], hashstring3[15]);

  // Block 2 ist GANZ statisch, da der Counter hier im ersten Block liegt!
  // (Indices 16 bis 31 im originalen hashstring-Array)
  tsh_uint4 b2_w0 = make_tsh_uint4(hashstring0[16], hashstring1[16], hashstring2[16], hashstring3[16]);
  tsh_uint4 b2_w1 = make_tsh_uint4(hashstring0[17], hashstring1[17], hashstring2[17], hashstring3[17]);
  tsh_uint4 b2_w2 = make_tsh_uint4(hashstring0[18], hashstring1[18], hashstring2[18], hashstring3[18]);
  tsh_uint4 b2_w3 = make_tsh_uint4(hashstring0[19], hashstring1[19], hashstring2[19], hashstring3[19]);
  tsh_uint4 b2_w4 = make_tsh_uint4(hashstring0[20], hashstring1[20], hashstring2[20], hashstring3[20]);
  tsh_uint4 b2_w5 = make_tsh_uint4(hashstring0[21], hashstring1[21], hashstring2[21], hashstring3[21]);
  tsh_uint4 b2_w6 = make_tsh_uint4(hashstring0[22], hashstring1[22], hashstring2[22], hashstring3[22]);
  tsh_uint4 b2_w7 = make_tsh_uint4(hashstring0[23], hashstring1[23], hashstring2[23], hashstring3[23]);
  tsh_uint4 b2_w8 = make_tsh_uint4(hashstring0[24], hashstring1[24], hashstring2[24], hashstring3[24]);
  tsh_uint4 b2_w9 = make_tsh_uint4(hashstring0[25], hashstring1[25], hashstring2[25], hashstring3[25]);
  tsh_uint4 b2_wa = make_tsh_uint4(hashstring0[26], hashstring1[26], hashstring2[26], hashstring3[26]);
  tsh_uint4 b2_wb = make_tsh_uint4(hashstring0[27], hashstring1[27], hashstring2[27], hashstring3[27]);
  tsh_uint4 b2_wc = make_tsh_uint4(hashstring0[28], hashstring1[28], hashstring2[28], hashstring3[28]);
  tsh_uint4 b2_wd = make_tsh_uint4(hashstring0[29], hashstring1[29], hashstring2[29], hashstring3[29]);
  tsh_uint4 b2_we = make_tsh_uint4(hashstring0[30], hashstring1[30], hashstring2[30], hashstring3[30]);
  tsh_uint4 b2_wf = make_tsh_uint4(hashstring0[31], hashstring1[31], hashstring2[31], hashstring3[31]);

  bool target_found = false;

  // ====================================================================
  // 2. DIE HAUPTSCHLEIFE (Pure Register)
  // ====================================================================
  for (unsigned long long it = 0; it < chunk; it++)
  {

      // Frischen Mid-State (Digest) fuer diese Iteration laden
      tsh_uint4 mid_a = dig1_0;
      tsh_uint4 mid_b = dig1_1;
      tsh_uint4 mid_c = dig1_2;
      tsh_uint4 mid_d = dig1_3;
      tsh_uint4 mid_e = dig1_4;

      // Nur die variablen Counter-Woerter (W8-W15 in Block 1) aktualisieren
      if (8 >= swapendianness_start && 8 <= counter_word_end) b1_w8 = make_tsh_uint4(hashstring0[8], hashstring1[8], hashstring2[8], hashstring3[8]);
      if (9 >= swapendianness_start && 9 <= counter_word_end) b1_w9 = make_tsh_uint4(hashstring0[9], hashstring1[9], hashstring2[9], hashstring3[9]);
      if (10 >= swapendianness_start && 10 <= counter_word_end) b1_wa = make_tsh_uint4(hashstring0[10], hashstring1[10], hashstring2[10], hashstring3[10]);
      if (11 >= swapendianness_start && 11 <= counter_word_end) b1_wb = make_tsh_uint4(hashstring0[11], hashstring1[11], hashstring2[11], hashstring3[11]);
      if (12 >= swapendianness_start && 12 <= counter_word_end) b1_wc = make_tsh_uint4(hashstring0[12], hashstring1[12], hashstring2[12], hashstring3[12]);
      if (13 >= swapendianness_start && 13 <= counter_word_end) b1_wd = make_tsh_uint4(hashstring0[13], hashstring1[13], hashstring2[13], hashstring3[13]);
      if (14 >= swapendianness_start && 14 <= counter_word_end) b1_we = make_tsh_uint4(hashstring0[14], hashstring1[14], hashstring2[14], hashstring3[14]);
      if (15 >= swapendianness_start && 15 <= counter_word_end) b1_wf = make_tsh_uint4(hashstring0[15], hashstring1[15], hashstring2[15], hashstring3[15]);

      // BLOCK 1: Komplette Kompression. Veraendert mid_a bis mid_e!
      sha1_64_vec4(
          b1_w0, b1_w1, b1_w2, b1_w3, b1_w4, b1_w5, b1_w6, b1_w7,
          b1_w8, b1_w9, b1_wa, b1_wb, b1_wc, b1_wd, b1_we, b1_wf,
          mid_a, mid_b, mid_c, mid_d, mid_e);

      // BLOCK 2: Check mit dem neuen Mid-State ausruestet
      if (sha1_64_check_vec4(
          b2_w0, b2_w1, b2_w2, b2_w3, b2_w4, b2_w5, b2_w6, b2_w7,
          b2_w8, b2_w9, b2_wa, b2_wb, b2_wc, b2_wd, b2_we, b2_wf,
          mid_a, mid_b, mid_c, mid_d, mid_e, targetdifficulty))
      {
          target_found = true;
      }

      // Counter hochzählen
      increaseStringCounterBE((unsigned char*)hashstring0, identity_length_snd_block + clen0 - 1);
      increaseStringCounterBE((unsigned char*)hashstring1, identity_length_snd_block + clen1 - 1);
      increaseStringCounterBE((unsigned char*)hashstring2, identity_length_snd_block + clen2 - 1);
      increaseStringCounterBE((unsigned char*)hashstring3, identity_length_snd_block + clen3 - 1);
  }

  results[gid] = target_found;
}

#endif // CUDAKERNEL_CUH_
