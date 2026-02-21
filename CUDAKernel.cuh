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
// Rotation primitives
// ============================================================================

__device__ __forceinline__ unsigned int rotl32(unsigned int val, unsigned int n)
{
	return __funnelshift_l(val, val, n);
}

// ============================================================================
// SHA-1 constants
// ============================================================================

typedef enum sha1_constants
{
	SHA1M_A = 0x67452301,
	SHA1M_B = 0xefcdab89,
	SHA1M_C = 0x98badcfe,
	SHA1M_D = 0x10325476,
	SHA1M_E = 0xc3d2e1f0,

	SHA1C00 = 0x5a827999,
	SHA1C01 = 0x6ed9eba1,
	SHA1C02 = 0x8f1bbcdc,
	SHA1C03 = 0xca62c1d6u
} sha1_constants_t;

// ============================================================================
// LOP3.LUT PTX inline functions for SHA-1 round functions
// ============================================================================

// Ch(x,y,z) = z ^ (x & (y ^ z))  -- truth table 0xCA
__device__ __forceinline__ unsigned int lop3_0xCA(unsigned int a, unsigned int b, unsigned int c)
{
	unsigned int r;
	asm("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
	return r;
}

// Parity(x,y,z) = x ^ y ^ z  -- truth table 0x96
__device__ __forceinline__ unsigned int lop3_0x96(unsigned int a, unsigned int b, unsigned int c)
{
	unsigned int r;
	asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
	return r;
}

// Maj(x,y,z) = (x & y) | (z & (x ^ y))  -- truth table 0xE8
__device__ __forceinline__ unsigned int lop3_0xE8(unsigned int a, unsigned int b, unsigned int c)
{
	unsigned int r;
	asm("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
	return r;
}

// ============================================================================
// Overloaded SHA-1 round functions (scalar + vector dispatch)
// ============================================================================

// Scalar overloads
__device__ __forceinline__ unsigned int SHA1_F0o(unsigned int x, unsigned int y, unsigned int z)
{
	return lop3_0xCA(x, y, z);
}

__device__ __forceinline__ unsigned int SHA1_F1(unsigned int x, unsigned int y, unsigned int z)
{
	return lop3_0x96(x, y, z);
}

__device__ __forceinline__ unsigned int SHA1_F2o(unsigned int x, unsigned int y, unsigned int z)
{
	return lop3_0xE8(x, y, z);
}

// ============================================================================
// IMAD helper: integer add via FMAHeavy pipe (instead of INT/ALU pipe)
// ============================================================================

// On Ada Lovelace each SM sub-partition has 3 execution pipes:
//   FMAHeavy (IMAD, FP32 FMA)  |  FMALite (FP32 only)  |  INT (IADD3, LOP3, SHF, LEA)
// A pure-integer kernel leaves FMAHeavy and FMALite completely idle.
// By routing some additions through IMAD (a*1+b), we spread load across two pipes.
__device__ __forceinline__ unsigned int imad_add(unsigned int a, unsigned int b)
{
	unsigned int r;
	asm("mad.lo.u32 %0, %1, 1, %2;" : "=r"(r) : "r"(a), "r"(b));
	return r;
}

// ============================================================================
// SHA-1 step macros
// ============================================================================

// IMAD pipe-balancing strategy:
//   1. frot = f(b,c,d) + rotl(a,5)  via IMAD  → FMAHeavy (OFF critical path)
//   2. e   += x + K                 via IADD3  → INT      (ON critical path, 4 cyc)
//   3. e    = e + frot              via IMAD   → FMAHeavy (ON critical path, 4 cyc)
//   4. b    = rotl(b, 30)           via SHF    → INT      (independent)
//
// Critical path: e_prev → IADD3(4) → IMAD(4) = 8 cycles per round.
// Previous version: e_prev → IADD3(4) → LEA.HI(4) → IADD3(4) = 12 cycles.
//
// Key insight: by computing frot via IMAD on FMAHeavy, ptxas CANNOT fuse
// the rotation with the addition into LEA.HI (which is INT-pipe only).
// The rotation stays as a separate SHF, and frot is ready before the
// critical path needs it (since it's independent of e).
//
// Pipe balance per step: 3 INT (LOP3 + SHF + IADD3 + SHF) / 2 FMAHeavy (IMAD + IMAD)
#define SHA1_STEP(f, a, b, c, d, e, x)                     \
{                                                           \
  unsigned int frot = imad_add(f(b, c, d), rotl32(a, 5u)); \
  e += x + (unsigned int)(K);                               \
  e  = imad_add(e, frot);                                   \
  b  = rotl32(b, 30u);                                      \
}

// ============================================================================
// Byte swap functions
// ============================================================================

__device__ __forceinline__ unsigned int swap_uint(unsigned int val)
{
	return __byte_perm(val, 0, 0x0123);
}

__device__ __forceinline__ unsigned long long swap_ulong(unsigned long long val)
{
	unsigned int lo = (unsigned int)(val);
	unsigned int hi = (unsigned int)(val >> 32);
	unsigned int tlo = __byte_perm(lo, 0, 0x0123);
	unsigned int thi = __byte_perm(hi, 0, 0x0123);
	return ((unsigned long long)tlo << 32) | (unsigned long long)thi;
}

// ============================================================================
// Counter to string conversion
// ============================================================================

__device__ __forceinline__ unsigned long long countertostring(unsigned char* dst, unsigned long long c)
{
	if (c == 0) { *dst = '0'; return 1; }
	unsigned char* dst0 = dst;
	unsigned char* dst1 = dst;
	unsigned long long counterlength = 0;
	while (c)
	{
		unsigned char currentdigit = (unsigned char)(c % 10);
		*dst1 = '0' + currentdigit;
		dst1++;
		counterlength++;
		c = c / 10;
	}
	// invert string
	dst1--;
	while (dst0 < dst1)
	{
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

__device__ __forceinline__ void increaseStringCounterBE(unsigned char* hashstring_bytes, int counter_end_idx)
{
	bool add = 1;
	int current_idx = counter_end_idx;
	while (add)
	{
		// Map logical string index to physical big-endian memory index
		int phys_idx = (current_idx & ~3) | (3 - (current_idx & 3));
		unsigned char currentdigit = hashstring_bytes[phys_idx];
		if (currentdigit == '9')
		{
			hashstring_bytes[phys_idx] = '0';
			current_idx--;
		}
		else
		{
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

	// Wir wandern durch die 8 Z�hler-W�rter (von hinten nach vorne)
#pragma unroll
	for (int i = 15; i >= 8; i--)
	{
		// Wort nur anfassen, wenn der Z�hler �berhaupt so weit reicht
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

			// Die 4 Bytes innerhalb des 32-Bit Wortes von hinten nach vorne pr�fen
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
						// Overflow auf '0', carry bleibt true f�r das n�chste Byte
						w = (w & ~(0xFF << shift)) | ('0' << shift);
					}
					else
					{
						// Z�hler um 1 erh�hen, carry beenden
						w += (1 << shift);
						carry = false;
					}
				}
			}

			// Zur�ck in das richtige Register schreiben
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

__device__ __forceinline__ void sha1_64(unsigned int block[16], unsigned int digest[5])
{
	unsigned int a = digest[0];
	unsigned int b = digest[1];
	unsigned int c = digest[2];
	unsigned int d = digest[3];
	unsigned int e = digest[4];

	unsigned int w0_t = block[0];
	unsigned int w1_t = block[1];
	unsigned int w2_t = block[2];
	unsigned int w3_t = block[3];
	unsigned int w4_t = block[4];
	unsigned int w5_t = block[5];
	unsigned int w6_t = block[6];
	unsigned int w7_t = block[7];
	unsigned int w8_t = block[8];
	unsigned int w9_t = block[9];
	unsigned int wa_t = block[10];
	unsigned int wb_t = block[11];
	unsigned int wc_t = block[12];
	unsigned int wd_t = block[13];
	unsigned int we_t = block[14];
	unsigned int wf_t = block[15];

#undef K
#define K SHA1C00

	SHA1_STEP(SHA1_F0o, a, b, c, d, e, w0_t);
	SHA1_STEP(SHA1_F0o, e, a, b, c, d, w1_t);
	SHA1_STEP(SHA1_F0o, d, e, a, b, c, w2_t);
	SHA1_STEP(SHA1_F0o, c, d, e, a, b, w3_t);
	SHA1_STEP(SHA1_F0o, b, c, d, e, a, w4_t);
	SHA1_STEP(SHA1_F0o, a, b, c, d, e, w5_t);
	SHA1_STEP(SHA1_F0o, e, a, b, c, d, w6_t);
	SHA1_STEP(SHA1_F0o, d, e, a, b, c, w7_t);
	SHA1_STEP(SHA1_F0o, c, d, e, a, b, w8_t);
	SHA1_STEP(SHA1_F0o, b, c, d, e, a, w9_t);
	SHA1_STEP(SHA1_F0o, a, b, c, d, e, wa_t);
	SHA1_STEP(SHA1_F0o, e, a, b, c, d, wb_t);
	SHA1_STEP(SHA1_F0o, d, e, a, b, c, wc_t);
	SHA1_STEP(SHA1_F0o, c, d, e, a, b, wd_t);
	SHA1_STEP(SHA1_F0o, b, c, d, e, a, we_t);
	SHA1_STEP(SHA1_F0o, a, b, c, d, e, wf_t);

	w0_t = rotl32((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP(SHA1_F0o, e, a, b, c, d, w0_t);
	w1_t = rotl32((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP(SHA1_F0o, d, e, a, b, c, w1_t);
	w2_t = rotl32((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP(SHA1_F0o, c, d, e, a, b, w2_t);
	w3_t = rotl32((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP(SHA1_F0o, b, c, d, e, a, w3_t);

#undef K
#define K SHA1C01

	w4_t = rotl32((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, w4_t);
	w5_t = rotl32((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, w5_t);
	w6_t = rotl32((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, w6_t);
	w7_t = rotl32((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, w7_t);
	w8_t = rotl32((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, w8_t);
	w9_t = rotl32((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, w9_t);
	wa_t = rotl32((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, wa_t);
	wb_t = rotl32((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, wb_t);
	wc_t = rotl32((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, wc_t);
	wd_t = rotl32((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, wd_t);
	we_t = rotl32((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, we_t);
	wf_t = rotl32((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, wf_t);
	w0_t = rotl32((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, w0_t);
	w1_t = rotl32((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, w1_t);
	w2_t = rotl32((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, w2_t);
	w3_t = rotl32((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, w3_t);
	w4_t = rotl32((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, w4_t);
	w5_t = rotl32((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, w5_t);
	w6_t = rotl32((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, w6_t);
	w7_t = rotl32((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, w7_t);

#undef K
#define K SHA1C02

	w8_t = rotl32((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP(SHA1_F2o, a, b, c, d, e, w8_t);
	w9_t = rotl32((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP(SHA1_F2o, e, a, b, c, d, w9_t);
	wa_t = rotl32((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP(SHA1_F2o, d, e, a, b, c, wa_t);
	wb_t = rotl32((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP(SHA1_F2o, c, d, e, a, b, wb_t);
	wc_t = rotl32((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP(SHA1_F2o, b, c, d, e, a, wc_t);
	wd_t = rotl32((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP(SHA1_F2o, a, b, c, d, e, wd_t);
	we_t = rotl32((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP(SHA1_F2o, e, a, b, c, d, we_t);
	wf_t = rotl32((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP(SHA1_F2o, d, e, a, b, c, wf_t);
	w0_t = rotl32((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP(SHA1_F2o, c, d, e, a, b, w0_t);
	w1_t = rotl32((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP(SHA1_F2o, b, c, d, e, a, w1_t);
	w2_t = rotl32((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP(SHA1_F2o, a, b, c, d, e, w2_t);
	w3_t = rotl32((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP(SHA1_F2o, e, a, b, c, d, w3_t);
	w4_t = rotl32((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP(SHA1_F2o, d, e, a, b, c, w4_t);
	w5_t = rotl32((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP(SHA1_F2o, c, d, e, a, b, w5_t);
	w6_t = rotl32((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP(SHA1_F2o, b, c, d, e, a, w6_t);
	w7_t = rotl32((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP(SHA1_F2o, a, b, c, d, e, w7_t);
	w8_t = rotl32((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP(SHA1_F2o, e, a, b, c, d, w8_t);
	w9_t = rotl32((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP(SHA1_F2o, d, e, a, b, c, w9_t);
	wa_t = rotl32((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP(SHA1_F2o, c, d, e, a, b, wa_t);
	wb_t = rotl32((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP(SHA1_F2o, b, c, d, e, a, wb_t);

#undef K
#define K SHA1C03

	wc_t = rotl32((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, wc_t);
	wd_t = rotl32((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, wd_t);
	we_t = rotl32((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, we_t);
	wf_t = rotl32((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, wf_t);
	w0_t = rotl32((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, w0_t);
	w1_t = rotl32((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, w1_t);
	w2_t = rotl32((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, w2_t);
	w3_t = rotl32((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, w3_t);
	w4_t = rotl32((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, w4_t);
	w5_t = rotl32((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, w5_t);
	w6_t = rotl32((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, w6_t);
	w7_t = rotl32((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, w7_t);
	w8_t = rotl32((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, w8_t);
	w9_t = rotl32((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, w9_t);
	wa_t = rotl32((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, wa_t);
	wb_t = rotl32((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, wb_t);
	wc_t = rotl32((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, wc_t);
	wd_t = rotl32((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, wd_t);
	we_t = rotl32((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, we_t);
	wf_t = rotl32((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, wf_t);

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

	SHA1_STEP_VEC4(SHA1_F0o, a, b, c, d, e, w0_t);
	SHA1_STEP_VEC4(SHA1_F0o, e, a, b, c, d, w1_t);
	SHA1_STEP_VEC4(SHA1_F0o, d, e, a, b, c, w2_t);
	SHA1_STEP_VEC4(SHA1_F0o, c, d, e, a, b, w3_t);
	SHA1_STEP_VEC4(SHA1_F0o, b, c, d, e, a, w4_t);
	SHA1_STEP_VEC4(SHA1_F0o, a, b, c, d, e, w5_t);
	SHA1_STEP_VEC4(SHA1_F0o, e, a, b, c, d, w6_t);
	SHA1_STEP_VEC4(SHA1_F0o, d, e, a, b, c, w7_t);
	SHA1_STEP_VEC4(SHA1_F0o, c, d, e, a, b, w8_t);
	SHA1_STEP_VEC4(SHA1_F0o, b, c, d, e, a, w9_t);
	SHA1_STEP_VEC4(SHA1_F0o, a, b, c, d, e, wa_t);
	SHA1_STEP_VEC4(SHA1_F0o, e, a, b, c, d, wb_t);
	SHA1_STEP_VEC4(SHA1_F0o, d, e, a, b, c, wc_t);
	SHA1_STEP_VEC4(SHA1_F0o, c, d, e, a, b, wd_t);
	SHA1_STEP_VEC4(SHA1_F0o, b, c, d, e, a, we_t);
	SHA1_STEP_VEC4(SHA1_F0o, a, b, c, d, e, wf_t);

	w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP_VEC4(SHA1_F0o, e, a, b, c, d, w0_t);
	w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP_VEC4(SHA1_F0o, d, e, a, b, c, w1_t);
	w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP_VEC4(SHA1_F0o, c, d, e, a, b, w2_t);
	w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP_VEC4(SHA1_F0o, b, c, d, e, a, w3_t);

#undef K
#define K SHA1C01

	w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, w4_t);
	w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, w5_t);
	w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, w6_t);
	w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, w7_t);
	w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, w8_t);
	w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, w9_t);
	wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, wa_t);
	wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, wb_t);
	wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, wc_t);
	wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, wd_t);
	we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, we_t);
	wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, wf_t);
	w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, w0_t);
	w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, w1_t);
	w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, w2_t);
	w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, w3_t);
	w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, w4_t);
	w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, w5_t);
	w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, w6_t);
	w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, w7_t);

#undef K
#define K SHA1C02

	w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, a, b, c, d, e, w8_t);
	w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, e, a, b, c, d, w9_t);
	wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, d, e, a, b, c, wa_t);
	wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, c, d, e, a, b, wb_t);
	wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, b, c, d, e, a, wc_t);
	wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, a, b, c, d, e, wd_t);
	we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, e, a, b, c, d, we_t);
	wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, d, e, a, b, c, wf_t);
	w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, c, d, e, a, b, w0_t);
	w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, b, c, d, e, a, w1_t);
	w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, a, b, c, d, e, w2_t);
	w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, e, a, b, c, d, w3_t);
	w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, d, e, a, b, c, w4_t);
	w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, c, d, e, a, b, w5_t);
	w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, b, c, d, e, a, w6_t);
	w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, a, b, c, d, e, w7_t);
	w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, e, a, b, c, d, w8_t);
	w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, d, e, a, b, c, w9_t);
	wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, c, d, e, a, b, wa_t);
	wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, b, c, d, e, a, wb_t);

#undef K
#define K SHA1C03

	wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, wc_t);
	wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, wd_t);
	we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, we_t);
	wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, wf_t);
	w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, w0_t);
	w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, w1_t);
	w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, w2_t);
	w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, w3_t);
	w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, w4_t);
	w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, w5_t);
	w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, w6_t);
	w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, w7_t);
	w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, w8_t);
	w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, w9_t);
	wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, wa_t);
	wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, wb_t);
	wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, wc_t);
	wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, wd_t);
	we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, we_t);
	wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, wf_t);

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

	SHA1_STEP_VEC4(SHA1_F0o, a, b, c, d, e, w0_t);
	SHA1_STEP_VEC4(SHA1_F0o, e, a, b, c, d, w1_t);
	SHA1_STEP_VEC4(SHA1_F0o, d, e, a, b, c, w2_t);
	SHA1_STEP_VEC4(SHA1_F0o, c, d, e, a, b, w3_t);
	SHA1_STEP_VEC4(SHA1_F0o, b, c, d, e, a, w4_t);
	SHA1_STEP_VEC4(SHA1_F0o, a, b, c, d, e, w5_t);
	SHA1_STEP_VEC4(SHA1_F0o, e, a, b, c, d, w6_t);
	SHA1_STEP_VEC4(SHA1_F0o, d, e, a, b, c, w7_t);
	SHA1_STEP_VEC4(SHA1_F0o, c, d, e, a, b, w8_t);
	SHA1_STEP_VEC4(SHA1_F0o, b, c, d, e, a, w9_t);
	SHA1_STEP_VEC4(SHA1_F0o, a, b, c, d, e, wa_t);
	SHA1_STEP_VEC4(SHA1_F0o, e, a, b, c, d, wb_t);
	SHA1_STEP_VEC4(SHA1_F0o, d, e, a, b, c, wc_t);
	SHA1_STEP_VEC4(SHA1_F0o, c, d, e, a, b, wd_t);
	SHA1_STEP_VEC4(SHA1_F0o, b, c, d, e, a, we_t);
	SHA1_STEP_VEC4(SHA1_F0o, a, b, c, d, e, wf_t);

	w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP_VEC4(SHA1_F0o, e, a, b, c, d, w0_t);
	w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP_VEC4(SHA1_F0o, d, e, a, b, c, w1_t);
	w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP_VEC4(SHA1_F0o, c, d, e, a, b, w2_t);
	w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP_VEC4(SHA1_F0o, b, c, d, e, a, w3_t);

#undef K
#define K SHA1C01

	w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, w4_t);
	w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, w5_t);
	w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, w6_t);
	w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, w7_t);
	w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, w8_t);
	w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, w9_t);
	wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, wa_t);
	wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, wb_t);
	wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, wc_t);
	wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, wd_t);
	we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, we_t);
	wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, wf_t);
	w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, w0_t);
	w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, w1_t);
	w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, w2_t);
	w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, w3_t);
	w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, w4_t);
	w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, w5_t);
	w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, w6_t);
	w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, w7_t);

#undef K
#define K SHA1C02

	w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, a, b, c, d, e, w8_t);
	w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, e, a, b, c, d, w9_t);
	wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, d, e, a, b, c, wa_t);
	wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, c, d, e, a, b, wb_t);
	wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, b, c, d, e, a, wc_t);
	wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, a, b, c, d, e, wd_t);
	we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, e, a, b, c, d, we_t);
	wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, d, e, a, b, c, wf_t);
	w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, c, d, e, a, b, w0_t);
	w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, b, c, d, e, a, w1_t);
	w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, a, b, c, d, e, w2_t);
	w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, e, a, b, c, d, w3_t);
	w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, d, e, a, b, c, w4_t);
	w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, c, d, e, a, b, w5_t);
	w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, b, c, d, e, a, w6_t);
	w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, a, b, c, d, e, w7_t);
	w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, e, a, b, c, d, w8_t);
	w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, d, e, a, b, c, w9_t);
	wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, c, d, e, a, b, wa_t);
	wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, b, c, d, e, a, wb_t);

#undef K
#define K SHA1C03

	wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, wc_t);
	wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, wd_t);
	we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, we_t);
	wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, wf_t);
	w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, w0_t);
	w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, w1_t);
	w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, w2_t);
	w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, w3_t);
	w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, w4_t);
	w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, w5_t);
	w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, w6_t);
	w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, w7_t);
	w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, w8_t);
	w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, w9_t);
	wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, wa_t);
	wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, wb_t);
	wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, wc_t);
	wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, wd_t);
	we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, we_t);
	wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, wf_t);

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
// Scalar SHA-1 with CLZ early-exit (full 80 rounds, single hash)
// ============================================================================

__device__ __forceinline__ bool sha1_64_check_scalar(
	unsigned int w0_t, unsigned int w1_t, unsigned int w2_t, unsigned int w3_t,
	unsigned int w4_t, unsigned int w5_t, unsigned int w6_t, unsigned int w7_t,
	unsigned int w8_t, unsigned int w9_t, unsigned int wa_t, unsigned int wb_t,
	unsigned int wc_t, unsigned int wd_t, unsigned int we_t, unsigned int wf_t,
	unsigned int dig0, unsigned int dig1, unsigned int dig2,
	unsigned int dig3, unsigned int dig4,
	unsigned char targetdifficulty)
{
	unsigned int a = dig0;
	unsigned int b = dig1;
	unsigned int c = dig2;
	unsigned int d = dig3;
	unsigned int e = dig4;

#undef K
#define K SHA1C00

	SHA1_STEP(SHA1_F0o, a, b, c, d, e, w0_t);
	SHA1_STEP(SHA1_F0o, e, a, b, c, d, w1_t);
	SHA1_STEP(SHA1_F0o, d, e, a, b, c, w2_t);
	SHA1_STEP(SHA1_F0o, c, d, e, a, b, w3_t);
	SHA1_STEP(SHA1_F0o, b, c, d, e, a, w4_t);
	SHA1_STEP(SHA1_F0o, a, b, c, d, e, w5_t);
	SHA1_STEP(SHA1_F0o, e, a, b, c, d, w6_t);
	SHA1_STEP(SHA1_F0o, d, e, a, b, c, w7_t);
	SHA1_STEP(SHA1_F0o, c, d, e, a, b, w8_t);
	SHA1_STEP(SHA1_F0o, b, c, d, e, a, w9_t);
	SHA1_STEP(SHA1_F0o, a, b, c, d, e, wa_t);
	SHA1_STEP(SHA1_F0o, e, a, b, c, d, wb_t);
	SHA1_STEP(SHA1_F0o, d, e, a, b, c, wc_t);
	SHA1_STEP(SHA1_F0o, c, d, e, a, b, wd_t);
	SHA1_STEP(SHA1_F0o, b, c, d, e, a, we_t);
	SHA1_STEP(SHA1_F0o, a, b, c, d, e, wf_t);

	w0_t = rotl32((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP(SHA1_F0o, e, a, b, c, d, w0_t);
	w1_t = rotl32((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP(SHA1_F0o, d, e, a, b, c, w1_t);
	w2_t = rotl32((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP(SHA1_F0o, c, d, e, a, b, w2_t);
	w3_t = rotl32((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP(SHA1_F0o, b, c, d, e, a, w3_t);

#undef K
#define K SHA1C01

	w4_t = rotl32((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, w4_t);
	w5_t = rotl32((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, w5_t);
	w6_t = rotl32((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, w6_t);
	w7_t = rotl32((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, w7_t);
	w8_t = rotl32((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, w8_t);
	w9_t = rotl32((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, w9_t);
	wa_t = rotl32((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, wa_t);
	wb_t = rotl32((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, wb_t);
	wc_t = rotl32((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, wc_t);
	wd_t = rotl32((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, wd_t);
	we_t = rotl32((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, we_t);
	wf_t = rotl32((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, wf_t);
	w0_t = rotl32((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, w0_t);
	w1_t = rotl32((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, w1_t);
	w2_t = rotl32((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, w2_t);
	w3_t = rotl32((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, w3_t);
	w4_t = rotl32((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, w4_t);
	w5_t = rotl32((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, w5_t);
	w6_t = rotl32((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, w6_t);
	w7_t = rotl32((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, w7_t);

#undef K
#define K SHA1C02

	w8_t = rotl32((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP(SHA1_F2o, a, b, c, d, e, w8_t);
	w9_t = rotl32((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP(SHA1_F2o, e, a, b, c, d, w9_t);
	wa_t = rotl32((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP(SHA1_F2o, d, e, a, b, c, wa_t);
	wb_t = rotl32((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP(SHA1_F2o, c, d, e, a, b, wb_t);
	wc_t = rotl32((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP(SHA1_F2o, b, c, d, e, a, wc_t);
	wd_t = rotl32((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP(SHA1_F2o, a, b, c, d, e, wd_t);
	we_t = rotl32((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP(SHA1_F2o, e, a, b, c, d, we_t);
	wf_t = rotl32((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP(SHA1_F2o, d, e, a, b, c, wf_t);
	w0_t = rotl32((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP(SHA1_F2o, c, d, e, a, b, w0_t);
	w1_t = rotl32((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP(SHA1_F2o, b, c, d, e, a, w1_t);
	w2_t = rotl32((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP(SHA1_F2o, a, b, c, d, e, w2_t);
	w3_t = rotl32((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP(SHA1_F2o, e, a, b, c, d, w3_t);
	w4_t = rotl32((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP(SHA1_F2o, d, e, a, b, c, w4_t);
	w5_t = rotl32((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP(SHA1_F2o, c, d, e, a, b, w5_t);
	w6_t = rotl32((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP(SHA1_F2o, b, c, d, e, a, w6_t);
	w7_t = rotl32((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP(SHA1_F2o, a, b, c, d, e, w7_t);
	w8_t = rotl32((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP(SHA1_F2o, e, a, b, c, d, w8_t);
	w9_t = rotl32((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP(SHA1_F2o, d, e, a, b, c, w9_t);
	wa_t = rotl32((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP(SHA1_F2o, c, d, e, a, b, wa_t);
	wb_t = rotl32((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP(SHA1_F2o, b, c, d, e, a, wb_t);

#undef K
#define K SHA1C03

	wc_t = rotl32((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, wc_t);
	wd_t = rotl32((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, wd_t);
	we_t = rotl32((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, we_t);
	wf_t = rotl32((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, wf_t);
	w0_t = rotl32((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, w0_t);
	w1_t = rotl32((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, w1_t);
	w2_t = rotl32((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, w2_t);
	w3_t = rotl32((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, w3_t);
	w4_t = rotl32((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, w4_t);
	w5_t = rotl32((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, w5_t);
	w6_t = rotl32((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, w6_t);
	w7_t = rotl32((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, w7_t);
	w8_t = rotl32((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, w8_t);
	w9_t = rotl32((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, w9_t);
	wa_t = rotl32((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, wa_t);
	wb_t = rotl32((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, wb_t);
	wc_t = rotl32((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, wc_t);
	wd_t = rotl32((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, wd_t);
	we_t = rotl32((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, we_t);
	wf_t = rotl32((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, wf_t);

	// ---- CLZ-based early exit (scalar) ----
	unsigned int final_h0 = dig0 + a;
	unsigned int z = __clz(final_h0);

	unsigned int td_cap = min((unsigned int)targetdifficulty, 32u);
	if (z < td_cap) return false;

	if (z == 32)
	{
		unsigned int final_h1 = dig1 + b;
		z += __clz(final_h1);
	}

	return z >= targetdifficulty;
}

// ============================================================================
// Scalar SHA-1 with CLZ early-exit, starting from round 8 (precomputed)
// ============================================================================

__device__ __forceinline__ bool sha1_64_check_scalar_r8(
	unsigned int w8_t, unsigned int w9_t, unsigned int wa_t, unsigned int wb_t,
	unsigned int wc_t, unsigned int wd_t, unsigned int we_t, unsigned int wf_t,
	unsigned int pre_w0_t, unsigned int pre_w1_t, unsigned int pre_w2_t, unsigned int pre_w3_t,
	unsigned int pre_w4_t, unsigned int pre_w5_t, unsigned int pre_w6_t, unsigned int pre_w7_t,
	unsigned int dig0, unsigned int dig1, unsigned int dig2,
	unsigned int dig3, unsigned int dig4,
	unsigned int pre_a, unsigned int pre_b, unsigned int pre_c,
	unsigned int pre_d, unsigned int pre_e,
	unsigned char targetdifficulty)
{
	unsigned int a = pre_a;
	unsigned int b = pre_b;
	unsigned int c = pre_c;
	unsigned int d = pre_d;
	unsigned int e = pre_e;

	unsigned int w0_t = pre_w0_t;
	unsigned int w1_t = pre_w1_t;
	unsigned int w2_t = pre_w2_t;
	unsigned int w3_t = pre_w3_t;
	unsigned int w4_t = pre_w4_t;
	unsigned int w5_t = pre_w5_t;
	unsigned int w6_t = pre_w6_t;
	unsigned int w7_t = pre_w7_t;

#undef K
#define K SHA1C00

	// ROUNDS 0-7 SKIPPED (pre-computed)

	// From round 8
	SHA1_STEP(SHA1_F0o, c, d, e, a, b, w8_t);
	SHA1_STEP(SHA1_F0o, b, c, d, e, a, w9_t);
	SHA1_STEP(SHA1_F0o, a, b, c, d, e, wa_t);
	SHA1_STEP(SHA1_F0o, e, a, b, c, d, wb_t);
	SHA1_STEP(SHA1_F0o, d, e, a, b, c, wc_t);
	SHA1_STEP(SHA1_F0o, c, d, e, a, b, wd_t);
	SHA1_STEP(SHA1_F0o, b, c, d, e, a, we_t);
	SHA1_STEP(SHA1_F0o, a, b, c, d, e, wf_t);

	w0_t = rotl32((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP(SHA1_F0o, e, a, b, c, d, w0_t);
	w1_t = rotl32((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP(SHA1_F0o, d, e, a, b, c, w1_t);
	w2_t = rotl32((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP(SHA1_F0o, c, d, e, a, b, w2_t);
	w3_t = rotl32((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP(SHA1_F0o, b, c, d, e, a, w3_t);

#undef K
#define K SHA1C01

	w4_t = rotl32((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, w4_t);
	w5_t = rotl32((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, w5_t);
	w6_t = rotl32((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, w6_t);
	w7_t = rotl32((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, w7_t);
	w8_t = rotl32((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, w8_t);
	w9_t = rotl32((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, w9_t);
	wa_t = rotl32((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, wa_t);
	wb_t = rotl32((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, wb_t);
	wc_t = rotl32((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, wc_t);
	wd_t = rotl32((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, wd_t);
	we_t = rotl32((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, we_t);
	wf_t = rotl32((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, wf_t);
	w0_t = rotl32((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, w0_t);
	w1_t = rotl32((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, w1_t);
	w2_t = rotl32((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, w2_t);
	w3_t = rotl32((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, w3_t);
	w4_t = rotl32((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, w4_t);
	w5_t = rotl32((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, w5_t);
	w6_t = rotl32((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, w6_t);
	w7_t = rotl32((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, w7_t);

#undef K
#define K SHA1C02

	w8_t = rotl32((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP(SHA1_F2o, a, b, c, d, e, w8_t);
	w9_t = rotl32((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP(SHA1_F2o, e, a, b, c, d, w9_t);
	wa_t = rotl32((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP(SHA1_F2o, d, e, a, b, c, wa_t);
	wb_t = rotl32((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP(SHA1_F2o, c, d, e, a, b, wb_t);
	wc_t = rotl32((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP(SHA1_F2o, b, c, d, e, a, wc_t);
	wd_t = rotl32((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP(SHA1_F2o, a, b, c, d, e, wd_t);
	we_t = rotl32((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP(SHA1_F2o, e, a, b, c, d, we_t);
	wf_t = rotl32((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP(SHA1_F2o, d, e, a, b, c, wf_t);
	w0_t = rotl32((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP(SHA1_F2o, c, d, e, a, b, w0_t);
	w1_t = rotl32((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP(SHA1_F2o, b, c, d, e, a, w1_t);
	w2_t = rotl32((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP(SHA1_F2o, a, b, c, d, e, w2_t);
	w3_t = rotl32((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP(SHA1_F2o, e, a, b, c, d, w3_t);
	w4_t = rotl32((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP(SHA1_F2o, d, e, a, b, c, w4_t);
	w5_t = rotl32((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP(SHA1_F2o, c, d, e, a, b, w5_t);
	w6_t = rotl32((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP(SHA1_F2o, b, c, d, e, a, w6_t);
	w7_t = rotl32((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP(SHA1_F2o, a, b, c, d, e, w7_t);
	w8_t = rotl32((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP(SHA1_F2o, e, a, b, c, d, w8_t);
	w9_t = rotl32((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP(SHA1_F2o, d, e, a, b, c, w9_t);
	wa_t = rotl32((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP(SHA1_F2o, c, d, e, a, b, wa_t);
	wb_t = rotl32((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP(SHA1_F2o, b, c, d, e, a, wb_t);

#undef K
#define K SHA1C03

	wc_t = rotl32((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, wc_t);
	wd_t = rotl32((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, wd_t);
	we_t = rotl32((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, we_t);
	wf_t = rotl32((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, wf_t);
	w0_t = rotl32((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, w0_t);
	w1_t = rotl32((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, w1_t);
	w2_t = rotl32((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, w2_t);
	w3_t = rotl32((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, w3_t);
	w4_t = rotl32((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, w4_t);
	w5_t = rotl32((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, w5_t);
	w6_t = rotl32((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, w6_t);
	w7_t = rotl32((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, w7_t);
	w8_t = rotl32((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, w8_t);
	w9_t = rotl32((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, w9_t);
	wa_t = rotl32((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, wa_t);
	wb_t = rotl32((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP(SHA1_F1, a, b, c, d, e, wb_t);
	wc_t = rotl32((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP(SHA1_F1, e, a, b, c, d, wc_t);
	wd_t = rotl32((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP(SHA1_F1, d, e, a, b, c, wd_t);
	we_t = rotl32((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP(SHA1_F1, c, d, e, a, b, we_t);
	wf_t = rotl32((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP(SHA1_F1, b, c, d, e, a, wf_t);

	// ---- CLZ-based early exit (scalar) ----
	unsigned int final_h0 = dig0 + a;
	unsigned int z = __clz(final_h0);

	unsigned int td_cap = min((unsigned int)targetdifficulty, 32u);
	if (z < td_cap) return false;

	if (z == 32)
	{
		unsigned int final_h1 = dig1 + b;
		z += __clz(final_h1);
	}

	return z >= targetdifficulty;
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
	unsigned char targetdifficulty)
{

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
	SHA1_STEP_VEC4(SHA1_F0o, c, d, e, a, b, w8_t);
	SHA1_STEP_VEC4(SHA1_F0o, b, c, d, e, a, w9_t);
	SHA1_STEP_VEC4(SHA1_F0o, a, b, c, d, e, wa_t);
	SHA1_STEP_VEC4(SHA1_F0o, e, a, b, c, d, wb_t);
	SHA1_STEP_VEC4(SHA1_F0o, d, e, a, b, c, wc_t);
	SHA1_STEP_VEC4(SHA1_F0o, c, d, e, a, b, wd_t);
	SHA1_STEP_VEC4(SHA1_F0o, b, c, d, e, a, we_t);
	SHA1_STEP_VEC4(SHA1_F0o, a, b, c, d, e, wf_t);

	w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP_VEC4(SHA1_F0o, e, a, b, c, d, w0_t);
	w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP_VEC4(SHA1_F0o, d, e, a, b, c, w1_t);
	w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP_VEC4(SHA1_F0o, c, d, e, a, b, w2_t);
	w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP_VEC4(SHA1_F0o, b, c, d, e, a, w3_t);

#undef K
#define K SHA1C01

	w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, w4_t);
	w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, w5_t);
	w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, w6_t);
	w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, w7_t);
	w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, w8_t);
	w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, w9_t);
	wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, wa_t);
	wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, wb_t);
	wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, wc_t);
	wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, wd_t);
	we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, we_t);
	wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, wf_t);
	w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, w0_t);
	w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, w1_t);
	w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, w2_t);
	w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, w3_t);
	w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, w4_t);
	w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, w5_t);
	w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, w6_t);
	w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, w7_t);

#undef K
#define K SHA1C02

	w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, a, b, c, d, e, w8_t);
	w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, e, a, b, c, d, w9_t);
	wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, d, e, a, b, c, wa_t);
	wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, c, d, e, a, b, wb_t);
	wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, b, c, d, e, a, wc_t);
	wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, a, b, c, d, e, wd_t);
	we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, e, a, b, c, d, we_t);
	wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, d, e, a, b, c, wf_t);
	w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, c, d, e, a, b, w0_t);
	w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, b, c, d, e, a, w1_t);
	w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, a, b, c, d, e, w2_t);
	w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, e, a, b, c, d, w3_t);
	w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, d, e, a, b, c, w4_t);
	w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, c, d, e, a, b, w5_t);
	w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, b, c, d, e, a, w6_t);
	w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, a, b, c, d, e, w7_t);
	w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, e, a, b, c, d, w8_t);
	w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, d, e, a, b, c, w9_t);
	wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, c, d, e, a, b, wa_t);
	wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP_VEC4(SHA1_F2o, b, c, d, e, a, wb_t);

#undef K
#define K SHA1C03

	wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, wc_t);
	wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, wd_t);
	we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, we_t);
	wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, wf_t);
	w0_t = rotl32_vec4((wd_t ^ w8_t ^ w2_t ^ w0_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, w0_t);
	w1_t = rotl32_vec4((we_t ^ w9_t ^ w3_t ^ w1_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, w1_t);
	w2_t = rotl32_vec4((wf_t ^ wa_t ^ w4_t ^ w2_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, w2_t);
	w3_t = rotl32_vec4((w0_t ^ wb_t ^ w5_t ^ w3_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, w3_t);
	w4_t = rotl32_vec4((w1_t ^ wc_t ^ w6_t ^ w4_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, w4_t);
	w5_t = rotl32_vec4((w2_t ^ wd_t ^ w7_t ^ w5_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, w5_t);
	w6_t = rotl32_vec4((w3_t ^ we_t ^ w8_t ^ w6_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, w6_t);
	w7_t = rotl32_vec4((w4_t ^ wf_t ^ w9_t ^ w7_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, w7_t);
	w8_t = rotl32_vec4((w5_t ^ w0_t ^ wa_t ^ w8_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, w8_t);
	w9_t = rotl32_vec4((w6_t ^ w1_t ^ wb_t ^ w9_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, w9_t);
	wa_t = rotl32_vec4((w7_t ^ w2_t ^ wc_t ^ wa_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, wa_t);
	wb_t = rotl32_vec4((w8_t ^ w3_t ^ wd_t ^ wb_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, a, b, c, d, e, wb_t);
	wc_t = rotl32_vec4((w9_t ^ w4_t ^ we_t ^ wc_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, e, a, b, c, d, wc_t);
	wd_t = rotl32_vec4((wa_t ^ w5_t ^ wf_t ^ wd_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, d, e, a, b, c, wd_t);
	we_t = rotl32_vec4((wb_t ^ w6_t ^ w0_t ^ we_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, c, d, e, a, b, we_t);
	wf_t = rotl32_vec4((wc_t ^ w7_t ^ w1_t ^ wf_t), 1u);
	SHA1_STEP_VEC4(SHA1_F1, b, c, d, e, a, wf_t);

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

	for (int i = 0; i < 64; i++)
	{
		((unsigned char*)hashstring)[i] = identity[i];
	}

	// Hash the first block
	unsigned int digest1[5];
	digest1[0] = SHA1M_A;
	digest1[1] = SHA1M_B;
	digest1[2] = SHA1M_C;
	digest1[3] = SHA1M_D;
	digest1[4] = SHA1M_E;
	for (int j = 0; j < 16; j++)
	{
		hashstring[j] = swap_uint(((unsigned int*)hashstring)[j]);
	}
	sha1_64(hashstring, digest1);

	for (int i = 0; i < (int)identity_length_snd_block; i++)
	{
		((unsigned char*)hashstring)[i] = identity[i + 64];
	}

	for (int i = (int)identity_length_snd_block; i < 64; i++)
	{
		((unsigned char*)hashstring)[i] = 0;
	}

	for (int j = 0; j < (int)identity_length_snd_block / 4; j++)
	{
		hashstring[j] = swap_uint(hashstring[j]);
	}

	const int swapendianness_start = identity_length_snd_block / 4;

	// --- Scalar path (1 hash per thread) ---
	unsigned long long currentcounter = startcounter + (unsigned long long)gid * iterations;

	// Counter string and padding
	unsigned long long clen = countertostring(
		((unsigned char*)hashstring) + identity_length_snd_block, currentcounter);
	((unsigned char*)hashstring)[identity_length_snd_block + clen] = 0x80;
	*((unsigned long long*)(((unsigned char*)hashstring) + 56)) =
		swap_ulong(8 * ((unsigned long long)identity_length + clen));

	// Length to big-endian
	hashstring[14] = swap_uint(hashstring[14]);
	hashstring[15] = swap_uint(hashstring[15]);

	// Endianness swap for counter region
	for (int j = swapendianness_start; j < 14; j++)
	{
		hashstring[j] = swap_uint(hashstring[j]);
	}

	// Scalar digest and precomputed state
	unsigned int dig0 = digest1[0];
	unsigned int dig1 = digest1[1];
	unsigned int dig2 = digest1[2];
	unsigned int dig3 = digest1[3];
	unsigned int dig4 = digest1[4];

	unsigned int pre_a = dig0, pre_b = dig1, pre_c = dig2, pre_d = dig3, pre_e = dig4;

	// Static words W0-W7 (don't change with counter)
	unsigned int pre_w0 = hashstring[0], pre_w1 = hashstring[1];
	unsigned int pre_w2 = hashstring[2], pre_w3 = hashstring[3];
	unsigned int pre_w4 = hashstring[4], pre_w5 = hashstring[5];
	unsigned int pre_w6 = hashstring[6], pre_w7 = hashstring[7];

	// --- PARTIAL ROUND PRECOMPUTATION (rounds 0-7) ---
#undef K
#define K SHA1C00

	SHA1_STEP(SHA1_F0o, pre_a, pre_b, pre_c, pre_d, pre_e, pre_w0);
	SHA1_STEP(SHA1_F0o, pre_e, pre_a, pre_b, pre_c, pre_d, pre_w1);
	SHA1_STEP(SHA1_F0o, pre_d, pre_e, pre_a, pre_b, pre_c, pre_w2);
	SHA1_STEP(SHA1_F0o, pre_c, pre_d, pre_e, pre_a, pre_b, pre_w3);
	SHA1_STEP(SHA1_F0o, pre_b, pre_c, pre_d, pre_e, pre_a, pre_w4);
	SHA1_STEP(SHA1_F0o, pre_a, pre_b, pre_c, pre_d, pre_e, pre_w5);
	SHA1_STEP(SHA1_F0o, pre_e, pre_a, pre_b, pre_c, pre_d, pre_w6);
	SHA1_STEP(SHA1_F0o, pre_d, pre_e, pre_a, pre_b, pre_c, pre_w7);

	bool target_found = false;

	// ====================================================================
	// HOT LOOP (scalar, 1 hash per iteration)
	// ====================================================================
	for (unsigned int it = 0; it < iterations; it++)
	{
		if (sha1_64_check_scalar_r8(
			hashstring[8], hashstring[9], hashstring[10], hashstring[11],
			hashstring[12], hashstring[13], hashstring[14], hashstring[15],
			pre_w0, pre_w1, pre_w2, pre_w3, pre_w4, pre_w5, pre_w6, pre_w7,
			dig0, dig1, dig2, dig3, dig4,
			pre_a, pre_b, pre_c, pre_d, pre_e,
			targetdifficulty))
		{
			target_found = true;
		}

		increaseStringCounterBE((unsigned char*)hashstring,
			identity_length_snd_block + clen - 1);
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

	for (int i = 0; i < 64; i++)
	{
		((unsigned char*)hashstring)[i] = identity[i];
	}

	// Hash the first block
	unsigned int digest1[5];
	digest1[0] = SHA1M_A;
	digest1[1] = SHA1M_B;
	digest1[2] = SHA1M_C;
	digest1[3] = SHA1M_D;
	digest1[4] = SHA1M_E;
	for (int j = 0; j < 16; j++)
	{
		hashstring[j] = swap_uint(((unsigned int*)hashstring)[j]);
	}
	sha1_64(hashstring, digest1);

	for (int i = 0; i < (int)identity_length_snd_block; i++)
	{
		((unsigned char*)hashstring)[i] = identity[i + 64];
	}

	for (int i = (int)identity_length_snd_block; i < 128; i++)
	{
		((unsigned char*)hashstring)[i] = 0;
	}

	for (int j = 0; j < (int)identity_length_snd_block / 4; j++)
	{
		hashstring[j] = swap_uint(hashstring[j]);
	}

	const int swapendianness_start = identity_length_snd_block / 4;

	// --- Scalar path (1 hash per thread) ---
	unsigned long long currentcounter = startcounter + (unsigned long long)gid * iterations;

	// Counter string and padding
	unsigned long long clen = countertostring(
		((unsigned char*)hashstring) + identity_length_snd_block, currentcounter);
	((unsigned char*)hashstring)[identity_length_snd_block + clen] = 0x80;
	*((unsigned long long*)(((unsigned char*)hashstring) + 64 + 56)) =
		swap_ulong(8 * ((unsigned long long)identity_length + clen));

	// First word of second block to big-endian (only 0x00000000 or 0x80000000)
	hashstring[16] = swap_uint(hashstring[16]);

	// Length to big-endian
	hashstring[30] = swap_uint(hashstring[30]);
	hashstring[31] = swap_uint(hashstring[31]);

	// Endianness swap for counter region in first block
	for (int j = swapendianness_start; j < 16; j++)
	{
		hashstring[j] = swap_uint(hashstring[j]);
	}

	// Block 2 words are completely static (counter is in block 1)
	unsigned int b2_w0 = hashstring[16], b2_w1 = hashstring[17];
	unsigned int b2_w2 = hashstring[18], b2_w3 = hashstring[19];
	unsigned int b2_w4 = hashstring[20], b2_w5 = hashstring[21];
	unsigned int b2_w6 = hashstring[22], b2_w7 = hashstring[23];
	unsigned int b2_w8 = hashstring[24], b2_w9 = hashstring[25];
	unsigned int b2_wa = hashstring[26], b2_wb = hashstring[27];
	unsigned int b2_wc = hashstring[28], b2_wd = hashstring[29];
	unsigned int b2_we = hashstring[30], b2_wf = hashstring[31];

	bool target_found = false;

	// ====================================================================
	// HOT LOOP (scalar, 1 hash per iteration, 2 SHA-1 blocks)
	// ====================================================================
	for (unsigned int it = 0; it < iterations; it++)
	{
		// BLOCK 1: Full scalar SHA-1 compression
		unsigned int blk1_dig[5] = { digest1[0], digest1[1], digest1[2], digest1[3], digest1[4] };
		sha1_64(hashstring, blk1_dig);

		// BLOCK 2: Check with the new mid-state
		if (sha1_64_check_scalar(
			b2_w0, b2_w1, b2_w2, b2_w3, b2_w4, b2_w5, b2_w6, b2_w7,
			b2_w8, b2_w9, b2_wa, b2_wb, b2_wc, b2_wd, b2_we, b2_wf,
			blk1_dig[0], blk1_dig[1], blk1_dig[2], blk1_dig[3], blk1_dig[4],
			targetdifficulty))
		{
			target_found = true;
		}

		increaseStringCounterBE((unsigned char*)hashstring,
			identity_length_snd_block + clen - 1);
	}

	results[gid] = target_found;
}

#endif // CUDAKERNEL_CUH_
