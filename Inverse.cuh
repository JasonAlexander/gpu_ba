/*
 * Copyright (c) 2011-2013 NVIDIA Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 *
 *   Redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimer.
 *
 *   Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 *   Neither the name of NVIDIA Corporation nor the names of its contributors
 *   may be used to endorse or promote products derived from this software 
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */


#pragma once

#include <iomanip>

//#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>
//#include <curand_kernel.h>
//#include <driver_types.h>
//#include <device_launch_parameters.h>

#define USE_PIVOTING   0


__device__ __forceinline__ float fmnaOp (float a, float b, float c)
{
    return -(a * b) + c;
}

__device__ __forceinline__ float mulOp (float a, float b)
{
    return a * b;
}

__device__ __forceinline__ float rcpOp (float a)
{
    return 1.0f / a;
}

__device__ __forceinline__ float absOp (float a)
{
    return fabsf(a);
}

__device__ __forceinline__ float negOp (float a)
{
    return -(a);
}



__device__ __forceinline__ double fmnaOp (double a, double b, double c)
{
    return -(a * b) + c;
}

__device__ __forceinline__ double mulOp (double a, double b)
{
    return a * b;
}

__device__ __forceinline__ double rcpOp (double a)
{
    return 1.0 / a;
}

__device__ __forceinline__ double absOp (double a)
{
    return fabs(a);
}

__device__ __forceinline__ double negOp (double a)
{
    return -(a);
}

template<typename T>
__global__ void matinv_3x3 (float *A, float *B, int batch, float l, bool mult)
{
    const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
    const int thrdNum = blkNum * blockDim.x + threadIdx.x;
//    int perm0, perm1, perm2;
//    int icol0, icol1, icol2;
    T AA00, AA01, AA02; 
    T AA10, AA11, AA12;
    T AA20, AA21, AA22;
    T tmp;

#if USE_PIVOTING
    float t;
    float p;
    int i, pvt;
#endif

    if (thrdNum < batch) {

        AA00 = (T)A[0 * batch + thrdNum];
			if(mult) AA00 = (AA00+ (AA00*l)); else AA00 += l;
        AA10 = (T)A[1 * batch + thrdNum];
        AA20 = (T)A[2 * batch + thrdNum];
        AA01 = (T)A[1 * batch + thrdNum];
        AA11 = (T)A[3 * batch + thrdNum];
			if(mult) AA11 = (AA11+ (AA11*l)); else AA11 += l;
        AA21 = (T)A[4 * batch + thrdNum];
        AA02 = (T)A[2 * batch + thrdNum];
        AA12 = (T)A[4 * batch + thrdNum];
        AA22 = (T)A[5 * batch + thrdNum];
			if(mult) AA22 = (AA22+ (AA22*l)); else AA22 += l;

//        perm0 = 0;
//        perm1 = 1;
//        perm2 = 2;
        
        /****************** iteration 0 ***********/

#if USE_PIVOTING
        /* search pivot row */
        p = absOp (AA00);
        pvt = 0;
        t = absOp (AA10);
        if (t > p) { p = t;  pvt = 1; }
        t = absOp (AA20);
        if (t > p) { p = t;  pvt = 2; }
        
        /* swap pivot row with row 0 */
        if (pvt == 1) {
            tmp = AA00;  AA00 = AA10;  AA10 = tmp;
            tmp = AA01;  AA01 = AA11;  AA11 = tmp;
            tmp = AA02;  AA02 = AA12;  AA12 = tmp;
            /* update permutation vector based on row swap */
            i = perm0;  perm0 = perm1;  perm1 = i;
        }
        if (pvt == 2) {
            tmp = AA00;  AA00 = AA20;  AA20 = tmp;
            tmp = AA01;  AA01 = AA21;  AA21 = tmp;
            tmp = AA02;  AA02 = AA22;  AA22 = tmp;
            /* update permutation vector based on row swap */
            i = perm0;  perm0 = perm2;  perm2 = i;
        }
#endif // USE_PIVOTING


        /* scale current row */
        tmp = rcpOp (AA00);
//        icol0 = perm0;
        AA00 = tmp;
        AA01 = mulOp (tmp, AA01);
        AA02 = mulOp (tmp, AA02);

        /* eliminate above and below current row */
        tmp = AA10;
        AA10 = mulOp (negOp(tmp), AA00);
        AA11 = fmnaOp (tmp, AA01, AA11);
        AA12 = fmnaOp (tmp, AA02, AA12);

        tmp = AA20;
        AA20 = mulOp (negOp(tmp), AA00);
        AA21 = fmnaOp (tmp, AA01, AA21);
        AA22 = fmnaOp (tmp, AA02, AA22);
        
        /****************** iteration 1 ***********/

#if USE_PIVOTING
        /* search pivot row */
        p = absOp (AA11);
        pvt = 1;
        t = absOp (AA21);
        if (t > p) { p = t;  pvt = 2; }

        /* swap pivot row with row 1 */
        if (pvt == 2) {
            tmp = AA10;   AA10 = AA20;   AA20 = tmp;
            tmp = AA11;   AA11 = AA21;   AA21 = tmp;
            tmp = AA12;   AA12 = AA22;   AA22 = tmp;
            /* update permutation vector based on row swap */
            i = perm1;   perm1 = perm2;   perm2 = i;
        }
#endif // USE_PIVOTING

        /* scale current row */
        tmp = rcpOp (AA11);
//        icol1 = perm1;
        AA10 = mulOp (tmp, AA10);
        AA11 = tmp;
        AA12 = mulOp (tmp, AA12);

        /* eliminate above and below current row */
        tmp = AA01;
        AA00 = fmnaOp (tmp, AA10, AA00);
        AA01 = mulOp (negOp(tmp), AA11);
        AA02 = fmnaOp (tmp, AA12, AA02);
        
        tmp = AA21;
        AA20 = fmnaOp (tmp, AA10, AA20);
        AA21 = mulOp (negOp(tmp), AA11);
        AA22 = fmnaOp (tmp, AA12, AA22);
        
        /****************** iteration 2 ****************/

        /* scale current row */
        tmp = rcpOp (AA22);
//        icol2 = perm2;
        AA20 = mulOp (tmp, AA20);
        AA21 = mulOp (tmp, AA21);
        AA22 = tmp;

        /* eliminate above and below current row */
        tmp = AA02;
        AA00 = fmnaOp (tmp, AA20, AA00);
        AA01 = fmnaOp (tmp, AA21, AA01); 
        AA02 = mulOp (negOp(tmp), AA22);

        tmp = AA12;
        AA10 = fmnaOp (tmp, AA20, AA10);
        AA11 = fmnaOp (tmp, AA21, AA11);
        AA12 = mulOp (negOp(tmp), AA22);

        /* sort columns into the correct order */
		/*
        B[(3*icol0 + 0) * batch + thrdNum] = AA00;
        B[(3*icol0 + 1) * batch + thrdNum] = AA10;
        B[(3*icol0 + 2) * batch + thrdNum] = AA20;
        B[(3*icol1 + 0) * batch + thrdNum] = AA01;
        B[(3*icol1 + 1) * batch + thrdNum] = AA11;
        B[(3*icol1 + 2) * batch + thrdNum] = AA21;
        B[(3*icol2 + 0) * batch + thrdNum] = AA02;
        B[(3*icol2 + 1) * batch + thrdNum] = AA12;
        B[(3*icol2 + 2) * batch + thrdNum] = AA22;
		*/

		B[0 * batch + thrdNum] = AA00;
        B[1 * batch + thrdNum] = AA10;
        B[2 * batch + thrdNum] = AA20;
        B[3 * batch + thrdNum] = AA11;
        B[4 * batch + thrdNum] = AA21;
        B[5 * batch + thrdNum] = AA22;
    }
}

template<typename T>
__global__ void matinv_6x6 (float *A, float *B, int batch, float l, bool mult)
{
    const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
    const int thrdNum = blkNum * blockDim.x + threadIdx.x;
//    int perm0, perm1, perm2, perm3, perm4, perm5;
//    int icol0, icol1, icol2, icol3, icol4, icol5;
    T AA00, AA01, AA02, AA03, AA04, AA05;
    T AA10, AA11, AA12, AA13, AA14, AA15;
    T AA20, AA21, AA22, AA23, AA24, AA25;
    T AA30, AA31, AA32, AA33, AA34, AA35; 
    T AA40, AA41, AA42, AA43, AA44, AA45;
    T AA50, AA51, AA52, AA53, AA54, AA55;
    T tmp;
#if USE_PIVOTING
    float t;
    float p;
    int i, pvt;
#endif


    if (thrdNum < batch) {

        AA00 = (T)A[0 * batch + thrdNum];
			if(mult) AA00 = (AA00+ (AA00*l)); else AA00 += l;
        AA10 = (T)A[1 * batch + thrdNum];
        AA20 = (T)A[2 * batch + thrdNum];
        AA30 = (T)A[3 * batch + thrdNum];
        AA40 = (T)A[4 * batch + thrdNum];
        AA50 = (T)A[5 * batch + thrdNum];

        AA01 = (T)A[1 * batch + thrdNum];
        AA11 = (T)A[6 * batch + thrdNum];
			if(mult) AA11 = (AA11+ (AA11*l)); else AA11 += l;
        AA21 = (T)A[7 * batch + thrdNum];
        AA31 = (T)A[8 * batch + thrdNum];
        AA41 = (T)A[9 * batch + thrdNum];
        AA51 = (T)A[10 * batch + thrdNum];

        AA02 = (T)A[2 * batch + thrdNum];
        AA12 = (T)A[7 * batch + thrdNum];
        AA22 = (T)A[11 * batch + thrdNum];
			if(mult) AA22 = (AA22+ (AA22*l)); else AA22 += l;
        AA32 = (T)A[12 * batch + thrdNum];
        AA42 = (T)A[13 * batch + thrdNum];
        AA52 = (T)A[14 * batch + thrdNum];

        AA03 = (T)A[3 * batch + thrdNum];
        AA13 = (T)A[8 * batch + thrdNum];
        AA23 = (T)A[12 * batch + thrdNum];
        AA33 = (T)A[15 * batch + thrdNum];
			if(mult) AA33 = (AA33+ (AA00*l)); else AA33 += l;
        AA43 = (T)A[16 * batch + thrdNum];
        AA53 = (T)A[17 * batch + thrdNum];

        AA04 = (T)A[4 * batch + thrdNum];
        AA14 = (T)A[9 * batch + thrdNum];
        AA24 = (T)A[13 * batch + thrdNum];
        AA34 = (T)A[16 * batch + thrdNum];
        AA44 = (T)A[18 * batch + thrdNum];
			if(mult) AA44 = (AA44+ (AA00*l)); else AA44 += l;
        AA54 = (T)A[19 * batch + thrdNum];

        AA05 = (T)A[5 * batch + thrdNum];
        AA15 = (T)A[10 * batch + thrdNum];
        AA25 = (T)A[14 * batch + thrdNum];
        AA35 = (T)A[17 * batch + thrdNum];
        AA45 = (T)A[19 * batch + thrdNum];
        AA55 = (T)A[20 * batch + thrdNum];
			if(mult) AA55 = (AA55+ (AA00*l)); else AA55 += l;

//        perm0 = 0;
//        perm1 = 1;
//        perm2 = 2;
//        perm3 = 3;
//        perm4 = 4;
//        perm5 = 5;
        
        /****************** iteration 0 ***********/


        /* scale current row */
        tmp = rcpOp (AA00);
//        icol0 = perm0;
        AA00 = tmp;
        AA01 = mulOp (tmp, AA01);
        AA02 = mulOp (tmp, AA02);
        AA03 = mulOp (tmp, AA03);
        AA04 = mulOp (tmp, AA04);
        AA05 = mulOp (tmp, AA05);

        /* eliminate above and below current row */
        tmp = AA10;
        AA10 = mulOp (negOp(tmp), AA00);
        AA11 = fmnaOp (tmp, AA01, AA11);
        AA12 = fmnaOp (tmp, AA02, AA12);
        AA13 = fmnaOp (tmp, AA03, AA13);
        AA14 = fmnaOp (tmp, AA04, AA14);
        AA15 = fmnaOp (tmp, AA05, AA15);

        tmp = AA20;
        AA20 = mulOp (negOp(tmp), AA00);
        AA21 = fmnaOp (tmp, AA01, AA21);
        AA22 = fmnaOp (tmp, AA02, AA22);
        AA23 = fmnaOp (tmp, AA03, AA23);
        AA24 = fmnaOp (tmp, AA04, AA24);
        AA25 = fmnaOp (tmp, AA05, AA25);

        tmp = AA30;
        AA30 = mulOp (negOp(tmp), AA00);
        AA31 = fmnaOp (tmp, AA01, AA31);
        AA32 = fmnaOp (tmp, AA02, AA32);
        AA33 = fmnaOp (tmp, AA03, AA33);
        AA34 = fmnaOp (tmp, AA04, AA34);
        AA35 = fmnaOp (tmp, AA05, AA35);

        tmp = AA40;
        AA40 = mulOp (negOp(tmp), AA00);
        AA41 = fmnaOp (tmp, AA01, AA41);
        AA42 = fmnaOp (tmp, AA02, AA42);
        AA43 = fmnaOp (tmp, AA03, AA43);
        AA44 = fmnaOp (tmp, AA04, AA44);
        AA45 = fmnaOp (tmp, AA05, AA45);
        
        tmp = AA50;
        AA50 = mulOp (negOp(tmp), AA00);
        AA51 = fmnaOp (tmp, AA01, AA51);
        AA52 = fmnaOp (tmp, AA02, AA52);
        AA53 = fmnaOp (tmp, AA03, AA53);
        AA54 = fmnaOp (tmp, AA04, AA54);
        AA55 = fmnaOp (tmp, AA05, AA55);

        /****************** iteration 1 ***********/

#if USE_PIVOTING
        /* search pivot row */
        p = absOp (AA11);
        pvt = 1;
        t = absOp (AA21);
        if (t > p) { p = t;  pvt = 2; }
        t = absOp (AA31);
        if (t > p) { p = t;  pvt = 3; }
        t = absOp (AA41);
        if (t > p) { p = t;  pvt = 4; }
        t = absOp (AA51);
        if (t > p) { p = t;  pvt = 5; }

        /* swap pivot row with row 1 */
        if (pvt == 2) {
            tmp = AA10;   AA10 = AA20;   AA20 = tmp;
            tmp = AA11;   AA11 = AA21;   AA21 = tmp;
            tmp = AA12;   AA12 = AA22;   AA22 = tmp;
            tmp = AA13;   AA13 = AA23;   AA23 = tmp;
            tmp = AA14;   AA14 = AA24;   AA24 = tmp;
            tmp = AA15;   AA15 = AA25;   AA25 = tmp;
            /* update permutation vector based on row swap */
            i = perm1;   perm1 = perm2;   perm2 = i;
        }
        if (pvt == 3) {
            tmp = AA10;   AA10 = AA30;   AA30 = tmp;
            tmp = AA11;   AA11 = AA31;   AA31 = tmp;
            tmp = AA12;   AA12 = AA32;   AA32 = tmp;
            tmp = AA13;   AA13 = AA33;   AA33 = tmp;
            tmp = AA14;   AA14 = AA34;   AA34 = tmp;
            tmp = AA15;   AA15 = AA35;   AA35 = tmp;
            /* update permutation vector based on row swap */
            i = perm1;   perm1 = perm3;   perm3 = i;
        }
        if (pvt == 4) {
            tmp = AA10;   AA10 = AA40;   AA40 = tmp;
            tmp = AA11;   AA11 = AA41;   AA41 = tmp;
            tmp = AA12;   AA12 = AA42;   AA42 = tmp;
            tmp = AA13;   AA13 = AA43;   AA43 = tmp;
            tmp = AA14;   AA14 = AA44;   AA44 = tmp;
            tmp = AA15;   AA15 = AA45;   AA45 = tmp;
           /* update permutation vector based on row swap */
            i = perm1;   perm1 = perm4;   perm4 = i;
        }
        if (pvt == 5) {
            tmp = AA10;   AA10 = AA50;   AA50 = tmp;
            tmp = AA11;   AA11 = AA51;   AA51 = tmp;
            tmp = AA12;   AA12 = AA52;   AA52 = tmp;
            tmp = AA13;   AA13 = AA53;   AA53 = tmp;
            tmp = AA14;   AA14 = AA54;   AA54 = tmp;
            tmp = AA15;   AA15 = AA55;   AA55 = tmp;
           /* update permutation vector based on row swap */
            i = perm1;   perm1 = perm5;   perm5 = i;
        }
#endif // USE_PIVOTING

        /* scale current row */
        tmp = rcpOp (AA11);
//        icol1 = perm1;
        AA10 = mulOp (tmp, AA10);
        AA11 = tmp;
        AA12 = mulOp (tmp, AA12);
        AA13 = mulOp (tmp, AA13);
        AA14 = mulOp (tmp, AA14);
        AA15 = mulOp (tmp, AA15);

        /* eliminate above and below current row */
        tmp = AA01;
        AA00 = fmnaOp (tmp, AA10, AA00);
        AA01 = mulOp (negOp(tmp), AA11);
        AA02 = fmnaOp (tmp, AA12, AA02);
        AA03 = fmnaOp (tmp, AA13, AA03);
        AA04 = fmnaOp (tmp, AA14, AA04);
        AA05 = fmnaOp (tmp, AA15, AA05);
        
        tmp = AA21;
        AA20 = fmnaOp (tmp, AA10, AA20);
        AA21 = mulOp (negOp(tmp), AA11);
        AA22 = fmnaOp (tmp, AA12, AA22);
        AA23 = fmnaOp (tmp, AA13, AA23);
        AA24 = fmnaOp (tmp, AA14, AA24);
        AA25 = fmnaOp (tmp, AA15, AA25);
        
        tmp = AA31;
        AA30 = fmnaOp (tmp, AA10, AA30);
        AA31 = mulOp (negOp(tmp), AA11);
        AA32 = fmnaOp (tmp, AA12, AA32);
        AA33 = fmnaOp (tmp, AA13, AA33);
        AA34 = fmnaOp (tmp, AA14, AA34);
        AA35 = fmnaOp (tmp, AA15, AA35);

        tmp = AA41;
        AA40 = fmnaOp (tmp, AA10, AA40);
        AA41 = mulOp (negOp(tmp), AA11);
        AA42 = fmnaOp (tmp, AA12, AA42);
        AA43 = fmnaOp (tmp, AA13, AA43);
        AA44 = fmnaOp (tmp, AA14, AA44);
        AA45 = fmnaOp (tmp, AA15, AA45);

        tmp = AA51;
        AA50 = fmnaOp (tmp, AA10, AA50);
        AA51 = mulOp (negOp(tmp), AA11);
        AA52 = fmnaOp (tmp, AA12, AA52);
        AA53 = fmnaOp (tmp, AA13, AA53);
        AA54 = fmnaOp (tmp, AA14, AA54);
        AA55 = fmnaOp (tmp, AA15, AA55);
        
        /****************** iteration 2 ****************/

#if USE_PIVOTING
        /* search pivot row */
        p = absOp (AA22);
        pvt = 2;
        t = absOp (AA32);
        if (t > p) { p = t;  pvt = 3; }
        t = absOp (AA42);
        if (t > p) { p = t;  pvt = 4; }
        t = absOp (AA52);
        if (t > p) { p = t;  pvt = 5; }

        /* swap pivot row with row 2 */
        if (pvt == 3) {
            tmp = AA20;   AA20 = AA30;   AA30 = tmp;
            tmp = AA21;   AA21 = AA31;   AA31 = tmp;
            tmp = AA22;   AA22 = AA32;   AA32 = tmp;
            tmp = AA23;   AA23 = AA33;   AA33 = tmp;
            tmp = AA24;   AA24 = AA34;   AA34 = tmp;
            tmp = AA25;   AA25 = AA35;   AA35 = tmp;
            /* update permutation vector based on row swap */
            i = perm2;   perm2 = perm3;   perm3 = i;
        }
        if (pvt == 4) {
            tmp = AA20;   AA20 = AA40;   AA40 = tmp;
            tmp = AA21;   AA21 = AA41;   AA41 = tmp;
            tmp = AA22;   AA22 = AA42;   AA42 = tmp;
            tmp = AA23;   AA23 = AA43;   AA43 = tmp;
            tmp = AA24;   AA24 = AA44;   AA44 = tmp;
            tmp = AA25;   AA25 = AA45;   AA45 = tmp;
            /* update permutation vector based on row swap */
            i = perm2;   perm2 = perm4;   perm4 = i;
        }
        if (pvt == 5) {
            tmp = AA20;   AA20 = AA50;   AA50 = tmp;
            tmp = AA21;   AA21 = AA51;   AA51 = tmp;
            tmp = AA22;   AA22 = AA52;   AA52 = tmp;
            tmp = AA23;   AA23 = AA53;   AA53 = tmp;
            tmp = AA24;   AA24 = AA54;   AA54 = tmp;
            tmp = AA25;   AA25 = AA55;   AA55 = tmp;
            /* update permutation vector based on row swap */
            i = perm2;   perm2 = perm5;   perm5 = i;
        }
#endif // USE_PIVOTING

        /* scale current row */
        tmp = rcpOp (AA22);
//        icol2 = perm2;
        AA20 = mulOp (tmp, AA20);
        AA21 = mulOp (tmp, AA21);
        AA22 = tmp;
        AA23 = mulOp (tmp, AA23);
        AA24 = mulOp (tmp, AA24);
        AA25 = mulOp (tmp, AA25);

        /* eliminate above and below current row */
        tmp = AA02;
        AA00 = fmnaOp (tmp, AA20, AA00);
        AA01 = fmnaOp (tmp, AA21, AA01); 
        AA02 = mulOp (negOp(tmp), AA22);
        AA03 = fmnaOp (tmp, AA23, AA03);
        AA04 = fmnaOp (tmp, AA24, AA04);
        AA05 = fmnaOp (tmp, AA25, AA05);

        tmp = AA12;
        AA10 = fmnaOp (tmp, AA20, AA10);
        AA11 = fmnaOp (tmp, AA21, AA11);
        AA12 = mulOp (negOp(tmp), AA22);
        AA13 = fmnaOp (tmp, AA23, AA13);
        AA14 = fmnaOp (tmp, AA24, AA14);
        AA15 = fmnaOp (tmp, AA25, AA15);

        tmp = AA32;
        AA30 = fmnaOp (tmp, AA20, AA30);
        AA31 = fmnaOp (tmp, AA21, AA31);
        AA32 = mulOp (negOp(tmp), AA22);
        AA33 = fmnaOp (tmp, AA23, AA33);
        AA34 = fmnaOp (tmp, AA24, AA34);
        AA35 = fmnaOp (tmp, AA25, AA35);

        tmp = AA42;
        AA40 = fmnaOp (tmp, AA20, AA40);
        AA41 = fmnaOp (tmp, AA21, AA41);
        AA42 = mulOp (negOp(tmp), AA22);
        AA43 = fmnaOp (tmp, AA23, AA43);
        AA44 = fmnaOp (tmp, AA24, AA44);
        AA45 = fmnaOp (tmp, AA25, AA45);

        tmp = AA52;
        AA50 = fmnaOp (tmp, AA20, AA50);
        AA51 = fmnaOp (tmp, AA21, AA51);
        AA52 = mulOp (negOp(tmp), AA22);
        AA53 = fmnaOp (tmp, AA23, AA53);
        AA54 = fmnaOp (tmp, AA24, AA54);
        AA55 = fmnaOp (tmp, AA25, AA55);

        /****************** iteration 3 ****************/

#if USE_PIVOTING
        /* search pivot row */
        p = absOp (AA33);
        pvt = 3;
        t = absOp (AA43);
        if (t > p) { p = t;  pvt = 4; }
        t = absOp (AA53);
        if (t > p) { p = t;  pvt = 5; }

        /* swap pivot row with row 3 */
        if (pvt == 4) {
            tmp = AA30;   AA30 = AA40;   AA40 = tmp;
            tmp = AA31;   AA31 = AA41;   AA41 = tmp;
            tmp = AA32;   AA32 = AA42;   AA42 = tmp;
            tmp = AA33;   AA33 = AA43;   AA43 = tmp;
            tmp = AA34;   AA34 = AA44;   AA44 = tmp;
            tmp = AA35;   AA35 = AA45;   AA45 = tmp;
            /* update permutation vector based on row swap */
            i = perm3;   perm3 = perm4;   perm4 = i;
        }
        if (pvt == 5) {
            tmp = AA30;   AA30 = AA50;   AA50 = tmp;
            tmp = AA31;   AA31 = AA51;   AA51 = tmp;
            tmp = AA32;   AA32 = AA52;   AA52 = tmp;
            tmp = AA33;   AA33 = AA53;   AA53 = tmp;
            tmp = AA34;   AA34 = AA54;   AA54 = tmp;
            tmp = AA35;   AA35 = AA55;   AA55 = tmp;
            /* update permutation vector based on row swap */
            i = perm3;   perm3 = perm5;   perm5 = i;
        }
#endif // USE_PIVOTING

        /* scale current row */
        tmp = rcpOp (AA33);
//        icol3 = perm3;
        AA30 = mulOp (tmp, AA30);
        AA31 = mulOp (tmp, AA31);
        AA32 = mulOp (tmp, AA32);
        AA33 = tmp;
        AA34 = mulOp (tmp, AA34);
        AA35 = mulOp (tmp, AA35);

        /* eliminate above and below current row */
        tmp = AA03;
        AA00 = fmnaOp (tmp, AA30, AA00);
        AA01 = fmnaOp (tmp, AA31, AA01);
        AA02 = fmnaOp (tmp, AA32, AA02);
        AA03 = mulOp (negOp(tmp), AA33);
        AA04 = fmnaOp (tmp, AA34, AA04);
        AA05 = fmnaOp (tmp, AA35, AA05);

        tmp = AA13;
        AA10 = fmnaOp (tmp, AA30, AA10);
        AA11 = fmnaOp (tmp, AA31, AA11);
        AA12 = fmnaOp (tmp, AA32, AA12);
        AA13 = mulOp (negOp(tmp), AA33);
        AA14 = fmnaOp (tmp, AA34, AA14);
        AA15 = fmnaOp (tmp, AA35, AA15);

        tmp = AA23;
        AA20 = fmnaOp (tmp, AA30, AA20);
        AA21 = fmnaOp (tmp, AA31, AA21);
        AA22 = fmnaOp (tmp, AA32, AA22);
        AA23 = mulOp (negOp(tmp), AA33);
        AA24 = fmnaOp (tmp, AA34, AA24);
        AA25 = fmnaOp (tmp, AA35, AA25);

        tmp = AA43;
        AA40 = fmnaOp (tmp, AA30, AA40);
        AA41 = fmnaOp (tmp, AA31, AA41);
        AA42 = fmnaOp (tmp, AA32, AA42);
        AA43 = mulOp (negOp(tmp), AA33);
        AA44 = fmnaOp (tmp, AA34, AA44);
        AA45 = fmnaOp (tmp, AA35, AA45);

        tmp = AA53;
        AA50 = fmnaOp (tmp, AA30, AA50);
        AA51 = fmnaOp (tmp, AA31, AA51);
        AA52 = fmnaOp (tmp, AA32, AA52);
        AA53 = mulOp (negOp(tmp), AA33);
        AA54 = fmnaOp (tmp, AA34, AA54);
        AA55 = fmnaOp (tmp, AA35, AA55);

        /****************** iteration 4 ****************/

#if USE_PIVOTING
        /* search pivot row */
        p = absOp (AA44);
        pvt = 4;
        t = absOp (AA54);
        if (t > p) { p = t;  pvt = 5; }

        /* swap pivot row with row 4 */
        if (pvt == 5) {
            tmp = AA40;   AA40 = AA50;   AA50 = tmp;
            tmp = AA41;   AA41 = AA51;   AA51 = tmp;
            tmp = AA42;   AA42 = AA52;   AA52 = tmp;
            tmp = AA43;   AA43 = AA53;   AA53 = tmp;
            tmp = AA44;   AA44 = AA54;   AA54 = tmp;
            tmp = AA45;   AA45 = AA55;   AA55 = tmp;
            /* update permutation vector based on row swap */
            i = perm4;   perm4 = perm5;   perm5 = i;
        }
#endif // USE_PIVOTING

        /* scale current row */
        tmp = rcpOp (AA44);
//        icol4 = perm4;
        AA40 = mulOp (tmp, AA40);
        AA41 = mulOp (tmp, AA41);
        AA42 = mulOp (tmp, AA42);
        AA43 = mulOp (tmp, AA43);
        AA44 = tmp;
        AA45 = mulOp (tmp, AA45);

        /* eliminate above and below current row */
        tmp = AA04;
        AA00 = fmnaOp (tmp, AA40, AA00);
        AA01 = fmnaOp (tmp, AA41, AA01);
        AA02 = fmnaOp (tmp, AA42, AA02);
        AA03 = fmnaOp (tmp, AA43, AA03);
        AA04 = mulOp (negOp(tmp), AA44);
        AA05 = fmnaOp (tmp, AA45, AA05);

        tmp = AA14;
        AA10 = fmnaOp (tmp, AA40, AA10);
        AA11 = fmnaOp (tmp, AA41, AA11);
        AA12 = fmnaOp (tmp, AA42, AA12);
        AA13 = fmnaOp (tmp, AA43, AA13);
        AA14 = mulOp (negOp(tmp), AA44);
        AA15 = fmnaOp (tmp, AA45, AA15);

        tmp = AA24;
        AA20 = fmnaOp (tmp, AA40, AA20);
        AA21 = fmnaOp (tmp, AA41, AA21);
        AA22 = fmnaOp (tmp, AA42, AA22);
        AA23 = fmnaOp (tmp, AA43, AA23);
        AA24 = mulOp (negOp(tmp), AA44);
        AA25 = fmnaOp (tmp, AA45, AA25);

        tmp = AA34;
        AA30 = fmnaOp (tmp, AA40, AA30);
        AA31 = fmnaOp (tmp, AA41, AA31);
        AA32 = fmnaOp (tmp, AA42, AA32);
        AA33 = fmnaOp (tmp, AA43, AA33);
        AA34 = mulOp (negOp(tmp), AA44);
        AA35 = fmnaOp (tmp, AA45, AA35);

        tmp = AA54;
        AA50 = fmnaOp (tmp, AA40, AA50);
        AA51 = fmnaOp (tmp, AA41, AA51);
        AA52 = fmnaOp (tmp, AA42, AA52);
        AA53 = fmnaOp (tmp, AA43, AA53);
        AA54 = mulOp (negOp(tmp), AA44);
        AA55 = fmnaOp (tmp, AA45, AA55);

        /****************** iteration 5 ****************/

        /* scale current row */
        tmp = rcpOp (AA55);
//        icol5 = perm5;
        AA50 = mulOp (tmp, AA50);
        AA51 = mulOp (tmp, AA51);
        AA52 = mulOp (tmp, AA52);
        AA53 = mulOp (tmp, AA53);
        AA54 = mulOp (tmp, AA54);
        AA55 = tmp;

        /* eliminate above and below current row */
        tmp = AA05;
        AA00 = fmnaOp (tmp, AA50, AA00);
        AA01 = fmnaOp (tmp, AA51, AA01);
        AA02 = fmnaOp (tmp, AA52, AA02);
        AA03 = fmnaOp (tmp, AA53, AA03);
        AA04 = fmnaOp (tmp, AA54, AA04);
        AA05 = mulOp (negOp(tmp), AA55);

        tmp = AA15;
        AA10 = fmnaOp (tmp, AA50, AA10);
        AA11 = fmnaOp (tmp, AA51, AA11);
        AA12 = fmnaOp (tmp, AA52, AA12);
        AA13 = fmnaOp (tmp, AA53, AA13);
        AA14 = fmnaOp (tmp, AA54, AA14);
        AA15 = mulOp (negOp(tmp), AA55);

        tmp = AA25;
        AA20 = fmnaOp (tmp, AA50, AA20);
        AA21 = fmnaOp (tmp, AA51, AA21);
        AA22 = fmnaOp (tmp, AA52, AA22);
        AA23 = fmnaOp (tmp, AA53, AA23);
        AA24 = fmnaOp (tmp, AA54, AA24);
        AA25 = mulOp (negOp(tmp), AA55);

        tmp = AA35;
        AA30 = fmnaOp (tmp, AA50, AA30);
        AA31 = fmnaOp (tmp, AA51, AA31);
        AA32 = fmnaOp (tmp, AA52, AA32);
        AA33 = fmnaOp (tmp, AA53, AA33);
        AA34 = fmnaOp (tmp, AA54, AA34);
        AA35 = mulOp (negOp(tmp), AA55);

        tmp = AA45;
        AA40 = fmnaOp (tmp, AA50, AA40);
        AA41 = fmnaOp (tmp, AA51, AA41);
        AA42 = fmnaOp (tmp, AA52, AA42);
        AA43 = fmnaOp (tmp, AA53, AA43);
        AA44 = fmnaOp (tmp, AA54, AA44);
        AA45 = mulOp (negOp(tmp), AA55);

        /* sort columns into the correct order */
        B[0 * batch + thrdNum] = AA00;
        B[1 * batch + thrdNum] = AA10;
        B[2 * batch + thrdNum] = AA20;
        B[3 * batch + thrdNum] = AA30;
        B[4 * batch + thrdNum] = AA40;
        B[5 * batch + thrdNum] = AA50;
        //B[(6*icol1 + 0) * batch + thrdNum] = AA01;
        B[6 * batch + thrdNum] = AA11;
        B[7 * batch + thrdNum] = AA21;
        B[8 * batch + thrdNum] = AA31;
        B[9 * batch + thrdNum] = AA41;
        B[10 * batch + thrdNum] = AA51;
        //B[(6*icol2 + 0) * batch + thrdNum] = AA02;
        //B[(6*icol2 + 1) * batch + thrdNum] = AA12;
        B[11 * batch + thrdNum] = AA22;
        B[12 * batch + thrdNum] = AA32;
        B[13 * batch + thrdNum] = AA42;
        B[14 * batch + thrdNum] = AA52;
        //B[(6*icol3 + 0) * batch + thrdNum] = AA03;
        //B[(6*icol3 + 1) * batch + thrdNum] = AA13;
        //B[(6*icol3 + 2) * batch + thrdNum] = AA23;
        B[15 * batch + thrdNum] = AA33;
        B[16 * batch + thrdNum] = AA43;
        B[17 * batch + thrdNum] = AA53;
        //B[4 * batch + thrdNum] = AA04;
        //B[9 * batch + thrdNum] = AA14;
        //B[13 * batch + thrdNum] = AA24;
        //B[16 * batch + thrdNum] = AA34;
        B[18 * batch + thrdNum] = AA44;
        B[19 * batch + thrdNum] = AA54;
        //B[5 * batch + thrdNum] = AA05;
        //B[10 * batch + thrdNum] = AA15;
        //B[14 * batch + thrdNum] = AA25;
        //B[17 * batch + thrdNum] = AA35;
        //B[19 * batch + thrdNum] = AA45;
        B[20 * batch + thrdNum] = AA55;
    }
}

template<typename T>
__global__ void matinv_9x9 (float *A, float *B, int batch, float l, bool mult)
{
    const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
    const int thrdNum = blkNum * blockDim.x + threadIdx.x;
//    int perm0, perm1, perm2, perm3, perm4, perm5, perm6, perm7, perm8;
//    int icol0, icol1, icol2, icol3, icol4, icol5, icol6, icol7, icol8;
    T AA00, AA01, AA02, AA03, AA04, AA05, AA06, AA07, AA08;
    T AA10, AA11, AA12, AA13, AA14, AA15, AA16, AA17, AA18;
    T AA20, AA21, AA22, AA23, AA24, AA25, AA26, AA27, AA28;
    T AA30, AA31, AA32, AA33, AA34, AA35, AA36, AA37, AA38; 
    T AA40, AA41, AA42, AA43, AA44, AA45, AA46, AA47, AA48;
    T AA50, AA51, AA52, AA53, AA54, AA55, AA56, AA57, AA58;
    T AA60, AA61, AA62, AA63, AA64, AA65, AA66, AA67, AA68;
    T AA70, AA71, AA72, AA73, AA74, AA75, AA76, AA77, AA78;
    T AA80, AA81, AA82, AA83, AA84, AA85, AA86, AA87, AA88;
    T tmp;
#if USE_PIVOTING
    float t;
    float p;
    int i, pvt;
#endif


    if (thrdNum < batch) {

        AA00 = (T)A[0 * batch + thrdNum];
			if(mult) AA00 = (AA00+ (AA00*l)); else AA00 += l;
        AA10 = (T)A[1 * batch + thrdNum];
        AA20 = (T)A[2 * batch + thrdNum];
        AA30 = (T)A[3 * batch + thrdNum];
        AA40 = (T)A[4 * batch + thrdNum];
        AA50 = (T)A[5 * batch + thrdNum];
        AA60 = (T)A[6 * batch + thrdNum];
        AA70 = (T)A[7 * batch + thrdNum];
        AA80 = (T)A[8 * batch + thrdNum];
       
		AA01 = (T)A[1 * batch + thrdNum];
        AA11 = (T)A[9 * batch + thrdNum];
			if(mult) AA11 = (AA11+ (AA11*l)); else AA11 += l;
        AA21 = (T)A[10 * batch + thrdNum];
        AA31 = (T)A[11 * batch + thrdNum];
        AA41 = (T)A[12 * batch + thrdNum];
        AA51 = (T)A[13 * batch + thrdNum];
        AA61 = (T)A[14 * batch + thrdNum];
        AA71 = (T)A[15 * batch + thrdNum];
        AA81 = (T)A[16 * batch + thrdNum];
        
		AA02 = (T)A[2 * batch + thrdNum];
        AA12 = (T)A[10 * batch + thrdNum];
        AA22 = (T)A[17 * batch + thrdNum];
			if(mult) AA22 = (AA22+ (AA22*l)); else AA22 += l;
        AA32 = (T)A[18 * batch + thrdNum];
        AA42 = (T)A[19 * batch + thrdNum];
        AA52 = (T)A[20 * batch + thrdNum];
        AA62 = (T)A[21 * batch + thrdNum];
        AA72 = (T)A[22 * batch + thrdNum];
        AA82 = (T)A[23 * batch + thrdNum];
        
		AA03 = (T)A[3 * batch + thrdNum];
        AA13 = (T)A[11 * batch + thrdNum];
        AA23 = (T)A[18 * batch + thrdNum];
        AA33 = (T)A[24 * batch + thrdNum];
			if(mult) AA33 = (AA33+ (AA33*l)); else AA33 += l;
        AA43 = (T)A[25 * batch + thrdNum];
        AA53 = (T)A[26 * batch + thrdNum];
        AA63 = (T)A[27 * batch + thrdNum];
        AA73 = (T)A[28 * batch + thrdNum];
        AA83 = (T)A[29 * batch + thrdNum];
        
		AA04 = (T)A[4 * batch + thrdNum];
        AA14 = (T)A[12 * batch + thrdNum];
        AA24 = (T)A[19 * batch + thrdNum];
        AA34 = (T)A[25 * batch + thrdNum];
        AA44 = (T)A[30 * batch + thrdNum];
			if(mult) AA44 = (AA44+ (AA44*l)); else AA44 += l;
        AA54 = (T)A[31 * batch + thrdNum];
        AA64 = (T)A[32 * batch + thrdNum];
        AA74 = (T)A[33 * batch + thrdNum];
        AA84 = (T)A[34 * batch + thrdNum];
        
		AA05 = (T)A[5 * batch + thrdNum];
        AA15 = (T)A[13 * batch + thrdNum];
        AA25 = (T)A[20 * batch + thrdNum];
        AA35 = (T)A[26 * batch + thrdNum];
        AA45 = (T)A[31 * batch + thrdNum];
        AA55 = (T)A[35 * batch + thrdNum];
			if(mult) AA55 = (AA55+ (AA55*l)); else AA55 += l;
        AA65 = (T)A[36 * batch + thrdNum];
        AA75 = (T)A[37 * batch + thrdNum];
        AA85 = (T)A[38 * batch + thrdNum];
        
		AA06 = (T)A[6 * batch + thrdNum];
        AA16 = (T)A[14 * batch + thrdNum];
        AA26 = (T)A[21 * batch + thrdNum];
        AA36 = (T)A[27 * batch + thrdNum];
        AA46 = (T)A[32 * batch + thrdNum];
        AA56 = (T)A[36 * batch + thrdNum];
        AA66 = (T)A[39 * batch + thrdNum];
			if(mult) AA66 = (AA66+ (AA66*l)); else AA66 += l;
        AA76 = (T)A[40 * batch + thrdNum];
        AA86 = (T)A[41 * batch + thrdNum];
        
		AA07 = (T)A[7 * batch + thrdNum];
        AA17 = (T)A[15 * batch + thrdNum];
        AA27 = (T)A[22 * batch + thrdNum];
        AA37 = (T)A[28 * batch + thrdNum];
        AA47 = (T)A[33 * batch + thrdNum];
        AA57 = (T)A[37 * batch + thrdNum];
        AA67 = (T)A[40 * batch + thrdNum];
        AA77 = (T)A[42 * batch + thrdNum];
			if(mult) AA77 = (AA77+ (AA77*l)); else AA77 += l;
        AA87 = (T)A[43 * batch + thrdNum];
        
		AA08 = (T)A[8 * batch + thrdNum];
        AA18 = (T)A[16 * batch + thrdNum];
        AA28 = (T)A[23 * batch + thrdNum];
        AA38 = (T)A[29 * batch + thrdNum];
        AA48 = (T)A[34 * batch + thrdNum];
        AA58 = (T)A[38 * batch + thrdNum];
        AA68 = (T)A[41 * batch + thrdNum];
        AA78 = (T)A[43 * batch + thrdNum];
        AA88 = (T)A[44 * batch + thrdNum];
			if(mult) AA88 = (AA88+ (AA88*l)); else AA88 += l;

//        perm0 = 0;
//        perm1 = 1;
//        perm2 = 2;
//        perm3 = 3;
//        perm4 = 4;
//        perm5 = 5;
//        perm6 = 6;
//        perm7 = 7;
//        perm8 = 8;
        
        /****************** iteration 0 ***********/

#if USE_PIVOTING
        /* search pivot row */
        p = absOp (AA00);
        pvt = 0;
        t = absOp (AA10);
        if (t > p) { p = t;  pvt = 1; }
        t = absOp (AA20);
        if (t > p) { p = t;  pvt = 2; }
        t = absOp (AA30);
        if (t > p) { p = t;  pvt = 3; }
        t = absOp (AA40);
        if (t > p) { p = t;  pvt = 4; }
        t = absOp (AA50);
        if (t > p) { p = t;  pvt = 5; }
        t = absOp (AA60);
        if (t > p) { p = t;  pvt = 6; }
        t = absOp (AA70);
        if (t > p) { p = t;  pvt = 7; }
        t = absOp (AA80);
        if (t > p) { p = t;  pvt = 8; }
        
        /* swap pivot row with row 0 */
        if (pvt == 1) {
            tmp = AA00;  AA00 = AA10;  AA10 = tmp;
            tmp = AA01;  AA01 = AA11;  AA11 = tmp;
            tmp = AA02;  AA02 = AA12;  AA12 = tmp;
            tmp = AA03;  AA03 = AA13;  AA13 = tmp;
            tmp = AA04;  AA04 = AA14;  AA14 = tmp;
            tmp = AA05;  AA05 = AA15;  AA15 = tmp;
            tmp = AA06;  AA06 = AA16;  AA16 = tmp;
            tmp = AA07;  AA07 = AA17;  AA17 = tmp;
            tmp = AA08;  AA08 = AA18;  AA18 = tmp;
            /* update permutation vector based on row swap */
            i = perm0;  perm0 = perm1;  perm1 = i;
        }
        if (pvt == 2) {
            tmp = AA00;  AA00 = AA20;  AA20 = tmp;
            tmp = AA01;  AA01 = AA21;  AA21 = tmp;
            tmp = AA02;  AA02 = AA22;  AA22 = tmp;
            tmp = AA03;  AA03 = AA23;  AA23 = tmp;
            tmp = AA04;  AA04 = AA24;  AA24 = tmp;
            tmp = AA05;  AA05 = AA25;  AA25 = tmp;
            tmp = AA06;  AA06 = AA26;  AA26 = tmp;
            tmp = AA07;  AA07 = AA27;  AA27 = tmp;
            tmp = AA08;  AA08 = AA28;  AA28 = tmp;
            /* update permutation vector based on row swap */
            i = perm0;  perm0 = perm2;  perm2 = i;
        }
        if (pvt == 3) {
            tmp = AA00;  AA00 = AA30;  AA30 = tmp;
            tmp = AA01;  AA01 = AA31;  AA31 = tmp;            
            tmp = AA02;  AA02 = AA32;  AA32 = tmp;
            tmp = AA03;  AA03 = AA33;  AA33 = tmp;
            tmp = AA04;  AA04 = AA34;  AA34 = tmp;
            tmp = AA05;  AA05 = AA35;  AA35 = tmp;
            tmp = AA06;  AA06 = AA36;  AA36 = tmp;
            tmp = AA07;  AA07 = AA37;  AA37 = tmp;
            tmp = AA08;  AA08 = AA38;  AA38 = tmp;
            /* update permutation vector based on row swap */
            i = perm0;  perm0 = perm3;  perm3 = i;
        }
        if (pvt == 4) {
            tmp = AA00;  AA00 = AA40;  AA40 = tmp;
            tmp = AA01;  AA01 = AA41;  AA41 = tmp;            
            tmp = AA02;  AA02 = AA42;  AA42 = tmp;
            tmp = AA03;  AA03 = AA43;  AA43 = tmp;
            tmp = AA04;  AA04 = AA44;  AA44 = tmp;
            tmp = AA05;  AA05 = AA45;  AA45 = tmp;
            tmp = AA06;  AA06 = AA46;  AA46 = tmp;
            tmp = AA07;  AA07 = AA47;  AA47 = tmp;
            tmp = AA08;  AA08 = AA48;  AA48 = tmp;
            /* update permutation vector based on row swap */
            i = perm0;  perm0 = perm4;  perm4 = i;
        }
        if (pvt == 5) {
            tmp = AA00;  AA00 = AA50;  AA50 = tmp;
            tmp = AA01;  AA01 = AA51;  AA51 = tmp;            
            tmp = AA02;  AA02 = AA52;  AA52 = tmp;
            tmp = AA03;  AA03 = AA53;  AA53 = tmp;
            tmp = AA04;  AA04 = AA54;  AA54 = tmp;
            tmp = AA05;  AA05 = AA55;  AA55 = tmp;
            tmp = AA06;  AA06 = AA56;  AA56 = tmp;
            tmp = AA07;  AA07 = AA57;  AA57 = tmp;
            tmp = AA08;  AA08 = AA58;  AA58 = tmp;
            /* update permutation vector based on row swap */
            i = perm0;  perm0 = perm5;  perm5 = i;
        }
        if (pvt == 6) {
            tmp = AA00;  AA00 = AA60;  AA60 = tmp;
            tmp = AA01;  AA01 = AA61;  AA61 = tmp;            
            tmp = AA02;  AA02 = AA62;  AA62 = tmp;
            tmp = AA03;  AA03 = AA63;  AA63 = tmp;
            tmp = AA04;  AA04 = AA64;  AA64 = tmp;
            tmp = AA05;  AA05 = AA65;  AA65 = tmp;
            tmp = AA06;  AA06 = AA66;  AA66 = tmp;
            tmp = AA07;  AA07 = AA67;  AA67 = tmp;
            tmp = AA08;  AA08 = AA68;  AA68 = tmp;
            /* update permutation vector based on row swap */
            i = perm0;  perm0 = perm6;  perm6 = i;
        }
        if (pvt == 7) {
            tmp = AA00;  AA00 = AA70;  AA70 = tmp;
            tmp = AA01;  AA01 = AA71;  AA71 = tmp;            
            tmp = AA02;  AA02 = AA72;  AA72 = tmp;
            tmp = AA03;  AA03 = AA73;  AA73 = tmp;
            tmp = AA04;  AA04 = AA74;  AA74 = tmp;
            tmp = AA05;  AA05 = AA75;  AA75 = tmp;
            tmp = AA06;  AA06 = AA76;  AA76 = tmp;
            tmp = AA07;  AA07 = AA77;  AA77 = tmp;
            tmp = AA08;  AA08 = AA78;  AA78 = tmp;
            /* update permutation vector based on row swap */
            i = perm0;  perm0 = perm7;  perm7 = i;
        }
        if (pvt == 8) {
            tmp = AA00;  AA00 = AA80;  AA80 = tmp;
            tmp = AA01;  AA01 = AA81;  AA81 = tmp;            
            tmp = AA02;  AA02 = AA82;  AA82 = tmp;
            tmp = AA03;  AA03 = AA83;  AA83 = tmp;
            tmp = AA04;  AA04 = AA84;  AA84 = tmp;
            tmp = AA05;  AA05 = AA85;  AA85 = tmp;
            tmp = AA06;  AA06 = AA86;  AA86 = tmp;
            tmp = AA07;  AA07 = AA87;  AA87 = tmp;
            tmp = AA08;  AA08 = AA88;  AA88 = tmp;
            /* update permutation vector based on row swap */
            i = perm0;  perm0 = perm8;  perm8 = i;
        }
#endif // USE_PIVOTING

        /* scale current row */
        tmp = rcpOp (AA00);
//        icol0 = perm0;
        AA00 = tmp;
        AA01 = mulOp (tmp, AA01);
        AA02 = mulOp (tmp, AA02);
        AA03 = mulOp (tmp, AA03);
        AA04 = mulOp (tmp, AA04);
        AA05 = mulOp (tmp, AA05);
        AA06 = mulOp (tmp, AA06);
        AA07 = mulOp (tmp, AA07);
        AA08 = mulOp (tmp, AA08);

        /* eliminate above and below current row */
        tmp = AA10;
        AA10 = mulOp (negOp(tmp), AA00);
        AA11 = fmnaOp (tmp, AA01, AA11);
        AA12 = fmnaOp (tmp, AA02, AA12);
        AA13 = fmnaOp (tmp, AA03, AA13);
        AA14 = fmnaOp (tmp, AA04, AA14);
        AA15 = fmnaOp (tmp, AA05, AA15);
        AA16 = fmnaOp (tmp, AA06, AA16);
        AA17 = fmnaOp (tmp, AA07, AA17);
        AA18 = fmnaOp (tmp, AA08, AA18);

        tmp = AA20;
        AA20 = mulOp (negOp(tmp), AA00);
        AA21 = fmnaOp (tmp, AA01, AA21);
        AA22 = fmnaOp (tmp, AA02, AA22);
        AA23 = fmnaOp (tmp, AA03, AA23);
        AA24 = fmnaOp (tmp, AA04, AA24);
        AA25 = fmnaOp (tmp, AA05, AA25);
        AA26 = fmnaOp (tmp, AA06, AA26);
        AA27 = fmnaOp (tmp, AA07, AA27);
        AA28 = fmnaOp (tmp, AA08, AA28);

        tmp = AA30;
        AA30 = mulOp (negOp(tmp), AA00);
        AA31 = fmnaOp (tmp, AA01, AA31);
        AA32 = fmnaOp (tmp, AA02, AA32);
        AA33 = fmnaOp (tmp, AA03, AA33);
        AA34 = fmnaOp (tmp, AA04, AA34);
        AA35 = fmnaOp (tmp, AA05, AA35);
        AA36 = fmnaOp (tmp, AA06, AA36);
        AA37 = fmnaOp (tmp, AA07, AA37);
        AA38 = fmnaOp (tmp, AA08, AA38);

        tmp = AA40;
        AA40 = mulOp (negOp(tmp), AA00);
        AA41 = fmnaOp (tmp, AA01, AA41);
        AA42 = fmnaOp (tmp, AA02, AA42);
        AA43 = fmnaOp (tmp, AA03, AA43);
        AA44 = fmnaOp (tmp, AA04, AA44);
        AA45 = fmnaOp (tmp, AA05, AA45);
        AA46 = fmnaOp (tmp, AA06, AA46);
        AA47 = fmnaOp (tmp, AA07, AA47);
        AA48 = fmnaOp (tmp, AA08, AA48);
        
        tmp = AA50;
        AA50 = mulOp (negOp(tmp), AA00);
        AA51 = fmnaOp (tmp, AA01, AA51);
        AA52 = fmnaOp (tmp, AA02, AA52);
        AA53 = fmnaOp (tmp, AA03, AA53);
        AA54 = fmnaOp (tmp, AA04, AA54);
        AA55 = fmnaOp (tmp, AA05, AA55);
        AA56 = fmnaOp (tmp, AA06, AA56);
        AA57 = fmnaOp (tmp, AA07, AA57);
        AA58 = fmnaOp (tmp, AA08, AA58);

        tmp = AA60;
        AA60 = mulOp (negOp(tmp), AA00);
        AA61 = fmnaOp (tmp, AA01, AA61);
        AA62 = fmnaOp (tmp, AA02, AA62);
        AA63 = fmnaOp (tmp, AA03, AA63);
        AA64 = fmnaOp (tmp, AA04, AA64);
        AA65 = fmnaOp (tmp, AA05, AA65);
        AA66 = fmnaOp (tmp, AA06, AA66);
        AA67 = fmnaOp (tmp, AA07, AA67);
        AA68 = fmnaOp (tmp, AA08, AA68);

        tmp = AA70;
        AA70 = mulOp (negOp(tmp), AA00);
        AA71 = fmnaOp (tmp, AA01, AA71);
        AA72 = fmnaOp (tmp, AA02, AA72);
        AA73 = fmnaOp (tmp, AA03, AA73);
        AA74 = fmnaOp (tmp, AA04, AA74);
        AA75 = fmnaOp (tmp, AA05, AA75);
        AA76 = fmnaOp (tmp, AA06, AA76);
        AA77 = fmnaOp (tmp, AA07, AA77);
        AA78 = fmnaOp (tmp, AA08, AA78);

        tmp = AA80;
        AA80 = mulOp (negOp(tmp), AA00);
        AA81 = fmnaOp (tmp, AA01, AA81);
        AA82 = fmnaOp (tmp, AA02, AA82);
        AA83 = fmnaOp (tmp, AA03, AA83);
        AA84 = fmnaOp (tmp, AA04, AA84);
        AA85 = fmnaOp (tmp, AA05, AA85);
        AA86 = fmnaOp (tmp, AA06, AA86);
        AA87 = fmnaOp (tmp, AA07, AA87);
        AA88 = fmnaOp (tmp, AA08, AA88);

        /****************** iteration 1 ***********/

#if USE_PIVOTING
        /* search pivot row */
        p = absOp (AA11);
        pvt = 1;
        t = absOp (AA21);
        if (t > p) { p = t;  pvt = 2; }
        t = absOp (AA31);
        if (t > p) { p = t;  pvt = 3; }
        t = absOp (AA41);
        if (t > p) { p = t;  pvt = 4; }
        t = absOp (AA51);
        if (t > p) { p = t;  pvt = 5; }
        t = absOp (AA61);
        if (t > p) { p = t;  pvt = 6; }
        t = absOp (AA71);
        if (t > p) { p = t;  pvt = 7; }
        t = absOp (AA81);
        if (t > p) { p = t;  pvt = 8; }

        /* swap pivot row with row 1 */
        if (pvt == 2) {
            tmp = AA10;   AA10 = AA20;   AA20 = tmp;
            tmp = AA11;   AA11 = AA21;   AA21 = tmp;
            tmp = AA12;   AA12 = AA22;   AA22 = tmp;
            tmp = AA13;   AA13 = AA23;   AA23 = tmp;
            tmp = AA14;   AA14 = AA24;   AA24 = tmp;
            tmp = AA15;   AA15 = AA25;   AA25 = tmp;
            tmp = AA16;   AA16 = AA26;   AA26 = tmp;
            tmp = AA17;   AA17 = AA27;   AA27 = tmp;
            tmp = AA18;   AA18 = AA28;   AA28 = tmp;
            /* update permutation vector based on row swap */
            i = perm1;   perm1 = perm2;   perm2 = i;
        }
        if (pvt == 3) {
            tmp = AA10;   AA10 = AA30;   AA30 = tmp;
            tmp = AA11;   AA11 = AA31;   AA31 = tmp;
            tmp = AA12;   AA12 = AA32;   AA32 = tmp;
            tmp = AA13;   AA13 = AA33;   AA33 = tmp;
            tmp = AA14;   AA14 = AA34;   AA34 = tmp;
            tmp = AA15;   AA15 = AA35;   AA35 = tmp;
            tmp = AA16;   AA16 = AA36;   AA36 = tmp;
            tmp = AA17;   AA17 = AA37;   AA37 = tmp;
            tmp = AA18;   AA18 = AA38;   AA38 = tmp;
            /* update permutation vector based on row swap */
            i = perm1;   perm1 = perm3;   perm3 = i;
        }
        if (pvt == 4) {
            tmp = AA10;   AA10 = AA40;   AA40 = tmp;
            tmp = AA11;   AA11 = AA41;   AA41 = tmp;
            tmp = AA12;   AA12 = AA42;   AA42 = tmp;
            tmp = AA13;   AA13 = AA43;   AA43 = tmp;
            tmp = AA14;   AA14 = AA44;   AA44 = tmp;
            tmp = AA15;   AA15 = AA45;   AA45 = tmp;
            tmp = AA16;   AA16 = AA46;   AA46 = tmp;
            tmp = AA17;   AA17 = AA47;   AA47 = tmp;
            tmp = AA18;   AA18 = AA48;   AA48 = tmp;
           /* update permutation vector based on row swap */
            i = perm1;   perm1 = perm4;   perm4 = i;
        }
        if (pvt == 5) {
            tmp = AA10;   AA10 = AA50;   AA50 = tmp;
            tmp = AA11;   AA11 = AA51;   AA51 = tmp;
            tmp = AA12;   AA12 = AA52;   AA52 = tmp;
            tmp = AA13;   AA13 = AA53;   AA53 = tmp;
            tmp = AA14;   AA14 = AA54;   AA54 = tmp;
            tmp = AA15;   AA15 = AA55;   AA55 = tmp;
            tmp = AA16;   AA16 = AA56;   AA56 = tmp;
            tmp = AA17;   AA17 = AA57;   AA57 = tmp;
            tmp = AA18;   AA18 = AA58;   AA58 = tmp;
           /* update permutation vector based on row swap */
            i = perm1;   perm1 = perm5;   perm5 = i;
        }
        if (pvt == 6) {
            tmp = AA10;   AA10 = AA60;   AA60 = tmp;
            tmp = AA11;   AA11 = AA61;   AA61 = tmp;
            tmp = AA12;   AA12 = AA62;   AA62 = tmp;
            tmp = AA13;   AA13 = AA63;   AA63 = tmp;
            tmp = AA14;   AA14 = AA64;   AA64 = tmp;
            tmp = AA15;   AA15 = AA65;   AA65 = tmp;
            tmp = AA16;   AA16 = AA66;   AA66 = tmp;
            tmp = AA17;   AA17 = AA67;   AA67 = tmp;
            tmp = AA18;   AA18 = AA68;   AA68 = tmp;
           /* update permutation vector based on row swap */
            i = perm1;   perm1 = perm6;   perm6 = i;
        }
        if (pvt == 7) {
            tmp = AA10;   AA10 = AA70;   AA70 = tmp;
            tmp = AA11;   AA11 = AA71;   AA71 = tmp;
            tmp = AA12;   AA12 = AA72;   AA72 = tmp;
            tmp = AA13;   AA13 = AA73;   AA73 = tmp;
            tmp = AA14;   AA14 = AA74;   AA74 = tmp;
            tmp = AA15;   AA15 = AA75;   AA75 = tmp;
            tmp = AA16;   AA16 = AA76;   AA76 = tmp;
            tmp = AA17;   AA17 = AA77;   AA77 = tmp;
            tmp = AA18;   AA18 = AA78;   AA78 = tmp;
           /* update permutation vector based on row swap */
            i = perm1;   perm1 = perm7;   perm7 = i;
        }
        if (pvt == 8) {
            tmp = AA10;   AA10 = AA80;   AA80 = tmp;
            tmp = AA11;   AA11 = AA81;   AA81 = tmp;
            tmp = AA12;   AA12 = AA82;   AA82 = tmp;
            tmp = AA13;   AA13 = AA83;   AA83 = tmp;
            tmp = AA14;   AA14 = AA84;   AA84 = tmp;
            tmp = AA15;   AA15 = AA85;   AA85 = tmp;
            tmp = AA16;   AA16 = AA86;   AA86 = tmp;
            tmp = AA17;   AA17 = AA87;   AA87 = tmp;
            tmp = AA18;   AA18 = AA88;   AA88 = tmp;
           /* update permutation vector based on row swap */
            i = perm1;   perm1 = perm8;   perm8 = i;
        }
#endif // USE_PIVOTING

        /* scale current row */
        tmp = rcpOp (AA11);
//        icol1 = perm1;
        AA10 = mulOp (tmp, AA10);
        AA11 = tmp;
        AA12 = mulOp (tmp, AA12);
        AA13 = mulOp (tmp, AA13);
        AA14 = mulOp (tmp, AA14);
        AA15 = mulOp (tmp, AA15);
        AA16 = mulOp (tmp, AA16);
        AA17 = mulOp (tmp, AA17);
        AA18 = mulOp (tmp, AA18);

        /* eliminate above and below current row */
        tmp = AA01;
        AA00 = fmnaOp (tmp, AA10, AA00);
        AA01 = mulOp (negOp(tmp), AA11);
        AA02 = fmnaOp (tmp, AA12, AA02);
        AA03 = fmnaOp (tmp, AA13, AA03);
        AA04 = fmnaOp (tmp, AA14, AA04);
        AA05 = fmnaOp (tmp, AA15, AA05);
        AA06 = fmnaOp (tmp, AA16, AA06);
        AA07 = fmnaOp (tmp, AA17, AA07);
        AA08 = fmnaOp (tmp, AA18, AA08);
        
        tmp = AA21;
        AA20 = fmnaOp (tmp, AA10, AA20);
        AA21 = mulOp (negOp(tmp), AA11);
        AA22 = fmnaOp (tmp, AA12, AA22);
        AA23 = fmnaOp (tmp, AA13, AA23);
        AA24 = fmnaOp (tmp, AA14, AA24);
        AA25 = fmnaOp (tmp, AA15, AA25);
        AA26 = fmnaOp (tmp, AA16, AA26);
        AA27 = fmnaOp (tmp, AA17, AA27);
        AA28 = fmnaOp (tmp, AA18, AA28);
        
        tmp = AA31;
        AA30 = fmnaOp (tmp, AA10, AA30);
        AA31 = mulOp (negOp(tmp), AA11);
        AA32 = fmnaOp (tmp, AA12, AA32);
        AA33 = fmnaOp (tmp, AA13, AA33);
        AA34 = fmnaOp (tmp, AA14, AA34);
        AA35 = fmnaOp (tmp, AA15, AA35);
        AA36 = fmnaOp (tmp, AA16, AA36);
        AA37 = fmnaOp (tmp, AA17, AA37);
        AA38 = fmnaOp (tmp, AA18, AA38);

        tmp = AA41;
        AA40 = fmnaOp (tmp, AA10, AA40);
        AA41 = mulOp (negOp(tmp), AA11);
        AA42 = fmnaOp (tmp, AA12, AA42);
        AA43 = fmnaOp (tmp, AA13, AA43);
        AA44 = fmnaOp (tmp, AA14, AA44);
        AA45 = fmnaOp (tmp, AA15, AA45);
        AA46 = fmnaOp (tmp, AA16, AA46);
        AA47 = fmnaOp (tmp, AA17, AA47);
        AA48 = fmnaOp (tmp, AA18, AA48);

        tmp = AA51;
        AA50 = fmnaOp (tmp, AA10, AA50);
        AA51 = mulOp (negOp(tmp), AA11);
        AA52 = fmnaOp (tmp, AA12, AA52);
        AA53 = fmnaOp (tmp, AA13, AA53);
        AA54 = fmnaOp (tmp, AA14, AA54);
        AA55 = fmnaOp (tmp, AA15, AA55);
        AA56 = fmnaOp (tmp, AA16, AA56);
        AA57 = fmnaOp (tmp, AA17, AA57);
        AA58 = fmnaOp (tmp, AA18, AA58);

        tmp = AA61;
        AA60 = fmnaOp (tmp, AA10, AA60);
        AA61 = mulOp (negOp(tmp), AA11);
        AA62 = fmnaOp (tmp, AA12, AA62);
        AA63 = fmnaOp (tmp, AA13, AA63);
        AA64 = fmnaOp (tmp, AA14, AA64);
        AA65 = fmnaOp (tmp, AA15, AA65);
        AA66 = fmnaOp (tmp, AA16, AA66);
        AA67 = fmnaOp (tmp, AA17, AA67);
        AA68 = fmnaOp (tmp, AA18, AA68);

        tmp = AA71;
        AA70 = fmnaOp (tmp, AA10, AA70);
        AA71 = mulOp (negOp(tmp), AA11);
        AA72 = fmnaOp (tmp, AA12, AA72);
        AA73 = fmnaOp (tmp, AA13, AA73);
        AA74 = fmnaOp (tmp, AA14, AA74);
        AA75 = fmnaOp (tmp, AA15, AA75);
        AA76 = fmnaOp (tmp, AA16, AA76);
        AA77 = fmnaOp (tmp, AA17, AA77);
        AA78 = fmnaOp (tmp, AA18, AA78);
        
        tmp = AA81;
        AA80 = fmnaOp (tmp, AA10, AA80);
        AA81 = mulOp (negOp(tmp), AA11);
        AA82 = fmnaOp (tmp, AA12, AA82);
        AA83 = fmnaOp (tmp, AA13, AA83);
        AA84 = fmnaOp (tmp, AA14, AA84);
        AA85 = fmnaOp (tmp, AA15, AA85);
        AA86 = fmnaOp (tmp, AA16, AA86);
        AA87 = fmnaOp (tmp, AA17, AA87);
        AA88 = fmnaOp (tmp, AA18, AA88);
        
        /****************** iteration 2 ****************/

#if USE_PIVOTING
        /* search pivot row */
        p = absOp (AA22);
        pvt = 2;
        t = absOp (AA32);
        if (t > p) { p = t;  pvt = 3; }
        t = absOp (AA42);
        if (t > p) { p = t;  pvt = 4; }
        t = absOp (AA52);
        if (t > p) { p = t;  pvt = 5; }
        t = absOp (AA62);
        if (t > p) { p = t;  pvt = 6; }
        t = absOp (AA72);
        if (t > p) { p = t;  pvt = 7; }
        t = absOp (AA82);
        if (t > p) { p = t;  pvt = 8; }

        /* swap pivot row with row 2 */
        if (pvt == 3) {
            tmp = AA20;   AA20 = AA30;   AA30 = tmp;
            tmp = AA21;   AA21 = AA31;   AA31 = tmp;
            tmp = AA22;   AA22 = AA32;   AA32 = tmp;
            tmp = AA23;   AA23 = AA33;   AA33 = tmp;
            tmp = AA24;   AA24 = AA34;   AA34 = tmp;
            tmp = AA25;   AA25 = AA35;   AA35 = tmp;
            tmp = AA26;   AA26 = AA36;   AA36 = tmp;
            tmp = AA27;   AA27 = AA37;   AA37 = tmp;
            tmp = AA28;   AA28 = AA38;   AA38 = tmp;
            /* update permutation vector based on row swap */
            i = perm2;   perm2 = perm3;   perm3 = i;
        }
        if (pvt == 4) {
            tmp = AA20;   AA20 = AA40;   AA40 = tmp;
            tmp = AA21;   AA21 = AA41;   AA41 = tmp;
            tmp = AA22;   AA22 = AA42;   AA42 = tmp;
            tmp = AA23;   AA23 = AA43;   AA43 = tmp;
            tmp = AA24;   AA24 = AA44;   AA44 = tmp;
            tmp = AA25;   AA25 = AA45;   AA45 = tmp;
            tmp = AA26;   AA26 = AA46;   AA46 = tmp;
            tmp = AA27;   AA27 = AA47;   AA47 = tmp;
            tmp = AA28;   AA28 = AA48;   AA48 = tmp;
            /* update permutation vector based on row swap */
            i = perm2;   perm2 = perm4;   perm4 = i;
        }
        if (pvt == 5) {
            tmp = AA20;   AA20 = AA50;   AA50 = tmp;
            tmp = AA21;   AA21 = AA51;   AA51 = tmp;
            tmp = AA22;   AA22 = AA52;   AA52 = tmp;
            tmp = AA23;   AA23 = AA53;   AA53 = tmp;
            tmp = AA24;   AA24 = AA54;   AA54 = tmp;
            tmp = AA25;   AA25 = AA55;   AA55 = tmp;
            tmp = AA26;   AA26 = AA56;   AA56 = tmp;
            tmp = AA27;   AA27 = AA57;   AA57 = tmp;
            tmp = AA28;   AA28 = AA58;   AA58 = tmp;
            /* update permutation vector based on row swap */
            i = perm2;   perm2 = perm5;   perm5 = i;
        }
        if (pvt == 6) {
            tmp = AA20;   AA20 = AA60;   AA60 = tmp;
            tmp = AA21;   AA21 = AA61;   AA61 = tmp;
            tmp = AA22;   AA22 = AA62;   AA62 = tmp;
            tmp = AA23;   AA23 = AA63;   AA63 = tmp;
            tmp = AA24;   AA24 = AA64;   AA64 = tmp;
            tmp = AA25;   AA25 = AA65;   AA65 = tmp;
            tmp = AA26;   AA26 = AA66;   AA66 = tmp;
            tmp = AA27;   AA27 = AA67;   AA67 = tmp;
            tmp = AA28;   AA28 = AA68;   AA68 = tmp;
            /* update permutation vector based on row swap */
            i = perm2;   perm2 = perm6;   perm6 = i;
        }
        if (pvt == 7) {
            tmp = AA20;   AA20 = AA70;   AA70 = tmp;
            tmp = AA21;   AA21 = AA71;   AA71 = tmp;
            tmp = AA22;   AA22 = AA72;   AA72 = tmp;
            tmp = AA23;   AA23 = AA73;   AA73 = tmp;
            tmp = AA24;   AA24 = AA74;   AA74 = tmp;
            tmp = AA25;   AA25 = AA75;   AA75 = tmp;
            tmp = AA26;   AA26 = AA76;   AA76 = tmp;
            tmp = AA27;   AA27 = AA77;   AA77 = tmp;
            tmp = AA28;   AA28 = AA78;   AA78 = tmp;
            /* update permutation vector based on row swap */
            i = perm2;   perm2 = perm7;   perm7 = i;
        }
        if (pvt == 8) {
            tmp = AA20;   AA20 = AA80;   AA80 = tmp;
            tmp = AA21;   AA21 = AA81;   AA81 = tmp;
            tmp = AA22;   AA22 = AA82;   AA82 = tmp;
            tmp = AA23;   AA23 = AA83;   AA83 = tmp;
            tmp = AA24;   AA24 = AA84;   AA84 = tmp;
            tmp = AA25;   AA25 = AA85;   AA85 = tmp;
            tmp = AA26;   AA26 = AA86;   AA86 = tmp;
            tmp = AA27;   AA27 = AA87;   AA87 = tmp;
            tmp = AA28;   AA28 = AA88;   AA88 = tmp;
            /* update permutation vector based on row swap */
            i = perm2;   perm2 = perm8;   perm8 = i;
        }
#endif // USE_PIVOTING

        /* scale current row */
        tmp = rcpOp (AA22);
//        icol2 = perm2;
        AA20 = mulOp (tmp, AA20);
        AA21 = mulOp (tmp, AA21);
        AA22 = tmp;
        AA23 = mulOp (tmp, AA23);
        AA24 = mulOp (tmp, AA24);
        AA25 = mulOp (tmp, AA25);
        AA26 = mulOp (tmp, AA26);
        AA27 = mulOp (tmp, AA27);
        AA28 = mulOp (tmp, AA28);

        /* eliminate above and below current row */
        tmp = AA02;
        AA00 = fmnaOp (tmp, AA20, AA00);
        AA01 = fmnaOp (tmp, AA21, AA01); 
        AA02 = mulOp (negOp(tmp), AA22);
        AA03 = fmnaOp (tmp, AA23, AA03);
        AA04 = fmnaOp (tmp, AA24, AA04);
        AA05 = fmnaOp (tmp, AA25, AA05);
        AA06 = fmnaOp (tmp, AA26, AA06);
        AA07 = fmnaOp (tmp, AA27, AA07);
        AA08 = fmnaOp (tmp, AA28, AA08);

        tmp = AA12;
        AA10 = fmnaOp (tmp, AA20, AA10);
        AA11 = fmnaOp (tmp, AA21, AA11);
        AA12 = mulOp (negOp(tmp), AA22);
        AA13 = fmnaOp (tmp, AA23, AA13);
        AA14 = fmnaOp (tmp, AA24, AA14);
        AA15 = fmnaOp (tmp, AA25, AA15);
        AA16 = fmnaOp (tmp, AA26, AA16);
        AA17 = fmnaOp (tmp, AA27, AA17);
        AA18 = fmnaOp (tmp, AA28, AA18);

        tmp = AA32;
        AA30 = fmnaOp (tmp, AA20, AA30);
        AA31 = fmnaOp (tmp, AA21, AA31);
        AA32 = mulOp (negOp(tmp), AA22);
        AA33 = fmnaOp (tmp, AA23, AA33);
        AA34 = fmnaOp (tmp, AA24, AA34);
        AA35 = fmnaOp (tmp, AA25, AA35);
        AA36 = fmnaOp (tmp, AA26, AA36);
        AA37 = fmnaOp (tmp, AA27, AA37);
        AA38 = fmnaOp (tmp, AA28, AA38);

        tmp = AA42;
        AA40 = fmnaOp (tmp, AA20, AA40);
        AA41 = fmnaOp (tmp, AA21, AA41);
        AA42 = mulOp (negOp(tmp), AA22);
        AA43 = fmnaOp (tmp, AA23, AA43);
        AA44 = fmnaOp (tmp, AA24, AA44);
        AA45 = fmnaOp (tmp, AA25, AA45);
        AA46 = fmnaOp (tmp, AA26, AA46);
        AA47 = fmnaOp (tmp, AA27, AA47);
        AA48 = fmnaOp (tmp, AA28, AA48);

        tmp = AA52;
        AA50 = fmnaOp (tmp, AA20, AA50);
        AA51 = fmnaOp (tmp, AA21, AA51);
        AA52 = mulOp (negOp(tmp), AA22);
        AA53 = fmnaOp (tmp, AA23, AA53);
        AA54 = fmnaOp (tmp, AA24, AA54);
        AA55 = fmnaOp (tmp, AA25, AA55);
        AA56 = fmnaOp (tmp, AA26, AA56);
        AA57 = fmnaOp (tmp, AA27, AA57);
        AA58 = fmnaOp (tmp, AA28, AA58);

        tmp = AA62;
        AA60 = fmnaOp (tmp, AA20, AA60);
        AA61 = fmnaOp (tmp, AA21, AA61);
        AA62 = mulOp (negOp(tmp), AA22);
        AA63 = fmnaOp (tmp, AA23, AA63);
        AA64 = fmnaOp (tmp, AA24, AA64);
        AA65 = fmnaOp (tmp, AA25, AA65);
        AA66 = fmnaOp (tmp, AA26, AA66);
        AA67 = fmnaOp (tmp, AA27, AA67);
        AA68 = fmnaOp (tmp, AA28, AA68);

        tmp = AA72;
        AA70 = fmnaOp (tmp, AA20, AA70);
        AA71 = fmnaOp (tmp, AA21, AA71);
        AA72 = mulOp (negOp(tmp), AA22);
        AA73 = fmnaOp (tmp, AA23, AA73);
        AA74 = fmnaOp (tmp, AA24, AA74);
        AA75 = fmnaOp (tmp, AA25, AA75);
        AA76 = fmnaOp (tmp, AA26, AA76);
        AA77 = fmnaOp (tmp, AA27, AA77);
        AA78 = fmnaOp (tmp, AA28, AA78);

        tmp = AA82;
        AA80 = fmnaOp (tmp, AA20, AA80);
        AA81 = fmnaOp (tmp, AA21, AA81);
        AA82 = mulOp (negOp(tmp), AA22);
        AA83 = fmnaOp (tmp, AA23, AA83);
        AA84 = fmnaOp (tmp, AA24, AA84);
        AA85 = fmnaOp (tmp, AA25, AA85);
        AA86 = fmnaOp (tmp, AA26, AA86);
        AA87 = fmnaOp (tmp, AA27, AA87);
        AA88 = fmnaOp (tmp, AA28, AA88);

        /****************** iteration 3 ****************/

#if USE_PIVOTING
        /* search pivot row */
        p = absOp (AA33);
        pvt = 3;
        t = absOp (AA43);
        if (t > p) { p = t;  pvt = 4; }
        t = absOp (AA53);
        if (t > p) { p = t;  pvt = 5; }
        t = absOp (AA63);
        if (t > p) { p = t;  pvt = 6; }
        t = absOp (AA73);
        if (t > p) { p = t;  pvt = 7; }
        t = absOp (AA83);
        if (t > p) { p = t;  pvt = 8; }

        /* swap pivot row with row 3 */
        if (pvt == 4) {
            tmp = AA30;   AA30 = AA40;   AA40 = tmp;
            tmp = AA31;   AA31 = AA41;   AA41 = tmp;
            tmp = AA32;   AA32 = AA42;   AA42 = tmp;
            tmp = AA33;   AA33 = AA43;   AA43 = tmp;
            tmp = AA34;   AA34 = AA44;   AA44 = tmp;
            tmp = AA35;   AA35 = AA45;   AA45 = tmp;
            tmp = AA36;   AA36 = AA46;   AA46 = tmp;
            tmp = AA37;   AA37 = AA47;   AA47 = tmp;
            tmp = AA38;   AA38 = AA48;   AA48 = tmp;
            /* update permutation vector based on row swap */
            i = perm3;   perm3 = perm4;   perm4 = i;
        }
        if (pvt == 5) {
            tmp = AA30;   AA30 = AA50;   AA50 = tmp;
            tmp = AA31;   AA31 = AA51;   AA51 = tmp;
            tmp = AA32;   AA32 = AA52;   AA52 = tmp;
            tmp = AA33;   AA33 = AA53;   AA53 = tmp;
            tmp = AA34;   AA34 = AA54;   AA54 = tmp;
            tmp = AA35;   AA35 = AA55;   AA55 = tmp;
            tmp = AA36;   AA36 = AA56;   AA56 = tmp;
            tmp = AA37;   AA37 = AA57;   AA57 = tmp;
            tmp = AA38;   AA38 = AA58;   AA58 = tmp;
            /* update permutation vector based on row swap */
            i = perm3;   perm3 = perm5;   perm5 = i;
        }
        if (pvt == 6) {
            tmp = AA30;   AA30 = AA60;   AA60 = tmp;
            tmp = AA31;   AA31 = AA61;   AA61 = tmp;
            tmp = AA32;   AA32 = AA62;   AA62 = tmp;
            tmp = AA33;   AA33 = AA63;   AA63 = tmp;
            tmp = AA34;   AA34 = AA64;   AA64 = tmp;
            tmp = AA35;   AA35 = AA65;   AA65 = tmp;
            tmp = AA36;   AA36 = AA66;   AA66 = tmp;
            tmp = AA37;   AA37 = AA67;   AA67 = tmp;
            tmp = AA38;   AA38 = AA68;   AA68 = tmp;
            /* update permutation vector based on row swap */
            i = perm3;   perm3 = perm6;   perm6 = i;
        }
        if (pvt == 7) {
            tmp = AA30;   AA30 = AA70;   AA70 = tmp;
            tmp = AA31;   AA31 = AA71;   AA71 = tmp;
            tmp = AA32;   AA32 = AA72;   AA72 = tmp;
            tmp = AA33;   AA33 = AA73;   AA73 = tmp;
            tmp = AA34;   AA34 = AA74;   AA74 = tmp;
            tmp = AA35;   AA35 = AA75;   AA75 = tmp;
            tmp = AA36;   AA36 = AA76;   AA76 = tmp;
            tmp = AA37;   AA37 = AA77;   AA77 = tmp;
            tmp = AA38;   AA38 = AA78;   AA78 = tmp;
            /* update permutation vector based on row swap */
            i = perm3;   perm3 = perm7;   perm7 = i;
        }
        if (pvt == 8) {
            tmp = AA30;   AA30 = AA80;   AA80 = tmp;
            tmp = AA31;   AA31 = AA81;   AA81 = tmp;
            tmp = AA32;   AA32 = AA82;   AA82 = tmp;
            tmp = AA33;   AA33 = AA83;   AA83 = tmp;
            tmp = AA34;   AA34 = AA84;   AA84 = tmp;
            tmp = AA35;   AA35 = AA85;   AA85 = tmp;
            tmp = AA36;   AA36 = AA86;   AA86 = tmp;
            tmp = AA37;   AA37 = AA87;   AA87 = tmp;
            tmp = AA38;   AA38 = AA88;   AA88 = tmp;
            /* update permutation vector based on row swap */
            i = perm3;   perm3 = perm8;   perm8 = i;
        }
#endif // USE_PIVOTING

        /* scale current row */
        tmp = rcpOp (AA33);
//        icol3 = perm3;
        AA30 = mulOp (tmp, AA30);
        AA31 = mulOp (tmp, AA31);
        AA32 = mulOp (tmp, AA32);
        AA33 = tmp;
        AA34 = mulOp (tmp, AA34);
        AA35 = mulOp (tmp, AA35);
        AA36 = mulOp (tmp, AA36);
        AA37 = mulOp (tmp, AA37);
        AA38 = mulOp (tmp, AA38);

        /* eliminate above and below current row */
        tmp = AA03;
        AA00 = fmnaOp (tmp, AA30, AA00);
        AA01 = fmnaOp (tmp, AA31, AA01);
        AA02 = fmnaOp (tmp, AA32, AA02);
        AA03 = mulOp (negOp(tmp), AA33);
        AA04 = fmnaOp (tmp, AA34, AA04);
        AA05 = fmnaOp (tmp, AA35, AA05);
        AA06 = fmnaOp (tmp, AA36, AA06);
        AA07 = fmnaOp (tmp, AA37, AA07);
        AA08 = fmnaOp (tmp, AA38, AA08);

        tmp = AA13;
        AA10 = fmnaOp (tmp, AA30, AA10);
        AA11 = fmnaOp (tmp, AA31, AA11);
        AA12 = fmnaOp (tmp, AA32, AA12);
        AA13 = mulOp (negOp(tmp), AA33);
        AA14 = fmnaOp (tmp, AA34, AA14);
        AA15 = fmnaOp (tmp, AA35, AA15);
        AA16 = fmnaOp (tmp, AA36, AA16);
        AA17 = fmnaOp (tmp, AA37, AA17);
        AA18 = fmnaOp (tmp, AA38, AA18);

        tmp = AA23;
        AA20 = fmnaOp (tmp, AA30, AA20);
        AA21 = fmnaOp (tmp, AA31, AA21);
        AA22 = fmnaOp (tmp, AA32, AA22);
        AA23 = mulOp (negOp(tmp), AA33);
        AA24 = fmnaOp (tmp, AA34, AA24);
        AA25 = fmnaOp (tmp, AA35, AA25);
        AA26 = fmnaOp (tmp, AA36, AA26);
        AA27 = fmnaOp (tmp, AA37, AA27);
        AA28 = fmnaOp (tmp, AA38, AA28);

        tmp = AA43;
        AA40 = fmnaOp (tmp, AA30, AA40);
        AA41 = fmnaOp (tmp, AA31, AA41);
        AA42 = fmnaOp (tmp, AA32, AA42);
        AA43 = mulOp (negOp(tmp), AA33);
        AA44 = fmnaOp (tmp, AA34, AA44);
        AA45 = fmnaOp (tmp, AA35, AA45);
        AA46 = fmnaOp (tmp, AA36, AA46);
        AA47 = fmnaOp (tmp, AA37, AA47);
        AA48 = fmnaOp (tmp, AA38, AA48);

        tmp = AA53;
        AA50 = fmnaOp (tmp, AA30, AA50);
        AA51 = fmnaOp (tmp, AA31, AA51);
        AA52 = fmnaOp (tmp, AA32, AA52);
        AA53 = mulOp (negOp(tmp), AA33);
        AA54 = fmnaOp (tmp, AA34, AA54);
        AA55 = fmnaOp (tmp, AA35, AA55);
        AA56 = fmnaOp (tmp, AA36, AA56);
        AA57 = fmnaOp (tmp, AA37, AA57);
        AA58 = fmnaOp (tmp, AA38, AA58);

        tmp = AA63;
        AA60 = fmnaOp (tmp, AA30, AA60);
        AA61 = fmnaOp (tmp, AA31, AA61);
        AA62 = fmnaOp (tmp, AA32, AA62);
        AA63 = mulOp (negOp(tmp), AA33);
        AA64 = fmnaOp (tmp, AA34, AA64);
        AA65 = fmnaOp (tmp, AA35, AA65);
        AA66 = fmnaOp (tmp, AA36, AA66);
        AA67 = fmnaOp (tmp, AA37, AA67);
        AA68 = fmnaOp (tmp, AA38, AA68);

        tmp = AA73;
        AA70 = fmnaOp (tmp, AA30, AA70);
        AA71 = fmnaOp (tmp, AA31, AA71);
        AA72 = fmnaOp (tmp, AA32, AA72);
        AA73 = mulOp (negOp(tmp), AA33);
        AA74 = fmnaOp (tmp, AA34, AA74);
        AA75 = fmnaOp (tmp, AA35, AA75);
        AA76 = fmnaOp (tmp, AA36, AA76);
        AA77 = fmnaOp (tmp, AA37, AA77);
        AA78 = fmnaOp (tmp, AA38, AA78);

        tmp = AA83;
        AA80 = fmnaOp (tmp, AA30, AA80);
        AA81 = fmnaOp (tmp, AA31, AA81);
        AA82 = fmnaOp (tmp, AA32, AA82);
        AA83 = mulOp (negOp(tmp), AA33);
        AA84 = fmnaOp (tmp, AA34, AA84);
        AA85 = fmnaOp (tmp, AA35, AA85);
        AA86 = fmnaOp (tmp, AA36, AA86);
        AA87 = fmnaOp (tmp, AA37, AA87);
        AA88 = fmnaOp (tmp, AA38, AA88);

        /****************** iteration 4 ****************/

#if USE_PIVOTING
        /* search pivot row */
        p = absOp (AA44);
        pvt = 4;
        t = absOp (AA54);
        if (t > p) { p = t;  pvt = 5; }
        t = absOp (AA64);
        if (t > p) { p = t;  pvt = 6; }
        t = absOp (AA74);
        if (t > p) { p = t;  pvt = 7; }
        t = absOp (AA84);
        if (t > p) { p = t;  pvt = 8; }

        /* swap pivot row with row 4 */
        if (pvt == 5) {
            tmp = AA40;   AA40 = AA50;   AA50 = tmp;
            tmp = AA41;   AA41 = AA51;   AA51 = tmp;
            tmp = AA42;   AA42 = AA52;   AA52 = tmp;
            tmp = AA43;   AA43 = AA53;   AA53 = tmp;
            tmp = AA44;   AA44 = AA54;   AA54 = tmp;
            tmp = AA45;   AA45 = AA55;   AA55 = tmp;
            tmp = AA46;   AA46 = AA56;   AA56 = tmp;
            tmp = AA47;   AA47 = AA57;   AA57 = tmp;
            tmp = AA48;   AA48 = AA58;   AA58 = tmp;
            /* update permutation vector based on row swap */
            i = perm4;   perm4 = perm5;   perm5 = i;
        }
        if (pvt == 6) {
            tmp = AA40;   AA40 = AA60;   AA60 = tmp;
            tmp = AA41;   AA41 = AA61;   AA61 = tmp;
            tmp = AA42;   AA42 = AA62;   AA62 = tmp;
            tmp = AA43;   AA43 = AA63;   AA63 = tmp;
            tmp = AA44;   AA44 = AA64;   AA64 = tmp;
            tmp = AA45;   AA45 = AA65;   AA65 = tmp;
            tmp = AA46;   AA46 = AA66;   AA66 = tmp;
            tmp = AA47;   AA47 = AA67;   AA67 = tmp;
            tmp = AA48;   AA48 = AA68;   AA68 = tmp;
            /* update permutation vector based on row swap */
            i = perm4;   perm4 = perm6;   perm6 = i;
        }
        if (pvt == 7) {
            tmp = AA40;   AA40 = AA70;   AA70 = tmp;
            tmp = AA41;   AA41 = AA71;   AA71 = tmp;
            tmp = AA42;   AA42 = AA72;   AA72 = tmp;
            tmp = AA43;   AA43 = AA73;   AA73 = tmp;
            tmp = AA44;   AA44 = AA74;   AA74 = tmp;
            tmp = AA45;   AA45 = AA75;   AA75 = tmp;
            tmp = AA46;   AA46 = AA76;   AA76 = tmp;
            tmp = AA47;   AA47 = AA77;   AA77 = tmp;
            tmp = AA48;   AA48 = AA78;   AA78 = tmp;
            /* update permutation vector based on row swap */
            i = perm4;   perm4 = perm7;   perm7 = i;
        }
        if (pvt == 8) {
            tmp = AA40;   AA40 = AA80;   AA80 = tmp;
            tmp = AA41;   AA41 = AA81;   AA81 = tmp;
            tmp = AA42;   AA42 = AA82;   AA82 = tmp;
            tmp = AA43;   AA43 = AA83;   AA83 = tmp;
            tmp = AA44;   AA44 = AA84;   AA84 = tmp;
            tmp = AA45;   AA45 = AA85;   AA85 = tmp;
            tmp = AA46;   AA46 = AA86;   AA86 = tmp;
            tmp = AA47;   AA47 = AA87;   AA87 = tmp;
            tmp = AA48;   AA48 = AA88;   AA88 = tmp;
            /* update permutation vector based on row swap */
            i = perm4;   perm4 = perm8;   perm8 = i;
        }
#endif // USE_PIVOTING

        /* scale current row */
        tmp = rcpOp (AA44);
//        icol4 = perm4;
        AA40 = mulOp (tmp, AA40);
        AA41 = mulOp (tmp, AA41);
        AA42 = mulOp (tmp, AA42);
        AA43 = mulOp (tmp, AA43);
        AA44 = tmp;
        AA45 = mulOp (tmp, AA45);
        AA46 = mulOp (tmp, AA46);
        AA47 = mulOp (tmp, AA47);
        AA48 = mulOp (tmp, AA48);

        /* eliminate above and below current row */
        tmp = AA04;
        AA00 = fmnaOp (tmp, AA40, AA00);
        AA01 = fmnaOp (tmp, AA41, AA01);
        AA02 = fmnaOp (tmp, AA42, AA02);
        AA03 = fmnaOp (tmp, AA43, AA03);
        AA04 = mulOp (negOp(tmp), AA44);
        AA05 = fmnaOp (tmp, AA45, AA05);
        AA06 = fmnaOp (tmp, AA46, AA06);
        AA07 = fmnaOp (tmp, AA47, AA07);
        AA08 = fmnaOp (tmp, AA48, AA08);

        tmp = AA14;
        AA10 = fmnaOp (tmp, AA40, AA10);
        AA11 = fmnaOp (tmp, AA41, AA11);
        AA12 = fmnaOp (tmp, AA42, AA12);
        AA13 = fmnaOp (tmp, AA43, AA13);
        AA14 = mulOp (negOp(tmp), AA44);
        AA15 = fmnaOp (tmp, AA45, AA15);
        AA16 = fmnaOp (tmp, AA46, AA16);
        AA17 = fmnaOp (tmp, AA47, AA17);
        AA18 = fmnaOp (tmp, AA48, AA18);

        tmp = AA24;
        AA20 = fmnaOp (tmp, AA40, AA20);
        AA21 = fmnaOp (tmp, AA41, AA21);
        AA22 = fmnaOp (tmp, AA42, AA22);
        AA23 = fmnaOp (tmp, AA43, AA23);
        AA24 = mulOp (negOp(tmp), AA44);
        AA25 = fmnaOp (tmp, AA45, AA25);
        AA26 = fmnaOp (tmp, AA46, AA26);
        AA27 = fmnaOp (tmp, AA47, AA27);
        AA28 = fmnaOp (tmp, AA48, AA28);

        tmp = AA34;
        AA30 = fmnaOp (tmp, AA40, AA30);
        AA31 = fmnaOp (tmp, AA41, AA31);
        AA32 = fmnaOp (tmp, AA42, AA32);
        AA33 = fmnaOp (tmp, AA43, AA33);
        AA34 = mulOp (negOp(tmp), AA44);
        AA35 = fmnaOp (tmp, AA45, AA35);
        AA36 = fmnaOp (tmp, AA46, AA36);
        AA37 = fmnaOp (tmp, AA47, AA37);
        AA38 = fmnaOp (tmp, AA48, AA38);

        tmp = AA54;
        AA50 = fmnaOp (tmp, AA40, AA50);
        AA51 = fmnaOp (tmp, AA41, AA51);
        AA52 = fmnaOp (tmp, AA42, AA52);
        AA53 = fmnaOp (tmp, AA43, AA53);
        AA54 = mulOp (negOp(tmp), AA44);
        AA55 = fmnaOp (tmp, AA45, AA55);
        AA56 = fmnaOp (tmp, AA46, AA56);
        AA57 = fmnaOp (tmp, AA47, AA57);
        AA58 = fmnaOp (tmp, AA48, AA58);

        tmp = AA64;
        AA60 = fmnaOp (tmp, AA40, AA60);
        AA61 = fmnaOp (tmp, AA41, AA61);
        AA62 = fmnaOp (tmp, AA42, AA62);
        AA63 = fmnaOp (tmp, AA43, AA63);
        AA64 = mulOp (negOp(tmp), AA44);
        AA65 = fmnaOp (tmp, AA45, AA65);
        AA66 = fmnaOp (tmp, AA46, AA66);
        AA67 = fmnaOp (tmp, AA47, AA67);
        AA68 = fmnaOp (tmp, AA48, AA68);

        tmp = AA74;
        AA70 = fmnaOp (tmp, AA40, AA70);
        AA71 = fmnaOp (tmp, AA41, AA71);
        AA72 = fmnaOp (tmp, AA42, AA72);
        AA73 = fmnaOp (tmp, AA43, AA73);
        AA74 = mulOp (negOp(tmp), AA44);
        AA75 = fmnaOp (tmp, AA45, AA75);
        AA76 = fmnaOp (tmp, AA46, AA76);
        AA77 = fmnaOp (tmp, AA47, AA77);
        AA78 = fmnaOp (tmp, AA48, AA78);

        tmp = AA84;
        AA80 = fmnaOp (tmp, AA40, AA80);
        AA81 = fmnaOp (tmp, AA41, AA81);
        AA82 = fmnaOp (tmp, AA42, AA82);
        AA83 = fmnaOp (tmp, AA43, AA83);
        AA84 = mulOp (negOp(tmp), AA44);
        AA85 = fmnaOp (tmp, AA45, AA85);
        AA86 = fmnaOp (tmp, AA46, AA86);
        AA87 = fmnaOp (tmp, AA47, AA87);
        AA88 = fmnaOp (tmp, AA48, AA88);

        /****************** iteration 5 ****************/

#if USE_PIVOTING
        /* search pivot row */
        p = absOp (AA55);
        pvt = 5;
        t = absOp (AA65);
        if (t > p) { p = t;  pvt = 6; }
        t = absOp (AA75);
        if (t > p) { p = t;  pvt = 7; }
        t = absOp (AA85);
        if (t > p) { p = t;  pvt = 8; }

        /* swap pivot row with row 5 */
        if (pvt == 6) {
            tmp = AA50;   AA50 = AA60;   AA60 = tmp;
            tmp = AA51;   AA51 = AA61;   AA61 = tmp;
            tmp = AA52;   AA52 = AA62;   AA62 = tmp;
            tmp = AA53;   AA53 = AA63;   AA63 = tmp;
            tmp = AA54;   AA54 = AA64;   AA64 = tmp;
            tmp = AA55;   AA55 = AA65;   AA65 = tmp;
            tmp = AA56;   AA56 = AA66;   AA66 = tmp;
            tmp = AA57;   AA57 = AA67;   AA67 = tmp;
            tmp = AA58;   AA58 = AA68;   AA68 = tmp;
            /* update permutation vector based on row swap */
            i = perm5;   perm5 = perm6;   perm6 = i;
        }
        if (pvt == 7) {
            tmp = AA50;   AA50 = AA70;   AA70 = tmp;
            tmp = AA51;   AA51 = AA71;   AA71 = tmp;
            tmp = AA52;   AA52 = AA72;   AA72 = tmp;
            tmp = AA53;   AA53 = AA73;   AA73 = tmp;
            tmp = AA54;   AA54 = AA74;   AA74 = tmp;
            tmp = AA55;   AA55 = AA75;   AA75 = tmp;
            tmp = AA56;   AA56 = AA76;   AA76 = tmp;
            tmp = AA57;   AA57 = AA77;   AA77 = tmp;
            tmp = AA58;   AA58 = AA78;   AA78 = tmp;
            /* update permutation vector based on row swap */
            i = perm5;   perm5 = perm7;   perm7 = i;
        }
        if (pvt == 8) {
            tmp = AA50;   AA50 = AA80;   AA80 = tmp;
            tmp = AA51;   AA51 = AA81;   AA81 = tmp;
            tmp = AA52;   AA52 = AA82;   AA82 = tmp;
            tmp = AA53;   AA53 = AA83;   AA83 = tmp;
            tmp = AA54;   AA54 = AA84;   AA84 = tmp;
            tmp = AA55;   AA55 = AA85;   AA85 = tmp;
            tmp = AA56;   AA56 = AA86;   AA86 = tmp;
            tmp = AA57;   AA57 = AA87;   AA87 = tmp;
            tmp = AA58;   AA58 = AA88;   AA88 = tmp;
            /* update permutation vector based on row swap */
            i = perm5;   perm5 = perm8;   perm8 = i;
        }
#endif // USE_PIVOTING

        /* scale current row */
        tmp = rcpOp (AA55);
//        icol5 = perm5;
        AA50 = mulOp (tmp, AA50);
        AA51 = mulOp (tmp, AA51);
        AA52 = mulOp (tmp, AA52);
        AA53 = mulOp (tmp, AA53);
        AA54 = mulOp (tmp, AA54);
        AA55 = tmp;
        AA56 = mulOp (tmp, AA56);
        AA57 = mulOp (tmp, AA57);
        AA58 = mulOp (tmp, AA58);

        /* eliminate above and below current row */
        tmp = AA05;
        AA00 = fmnaOp (tmp, AA50, AA00);
        AA01 = fmnaOp (tmp, AA51, AA01);
        AA02 = fmnaOp (tmp, AA52, AA02);
        AA03 = fmnaOp (tmp, AA53, AA03);
        AA04 = fmnaOp (tmp, AA54, AA04);
        AA05 = mulOp (negOp(tmp), AA55);
        AA06 = fmnaOp (tmp, AA56, AA06);
        AA07 = fmnaOp (tmp, AA57, AA07);
        AA08 = fmnaOp (tmp, AA58, AA08);

        tmp = AA15;
        AA10 = fmnaOp (tmp, AA50, AA10);
        AA11 = fmnaOp (tmp, AA51, AA11);
        AA12 = fmnaOp (tmp, AA52, AA12);
        AA13 = fmnaOp (tmp, AA53, AA13);
        AA14 = fmnaOp (tmp, AA54, AA14);
        AA15 = mulOp (negOp(tmp), AA55);
        AA16 = fmnaOp (tmp, AA56, AA16);
        AA17 = fmnaOp (tmp, AA57, AA17);
        AA18 = fmnaOp (tmp, AA58, AA18);

        tmp = AA25;
        AA20 = fmnaOp (tmp, AA50, AA20);
        AA21 = fmnaOp (tmp, AA51, AA21);
        AA22 = fmnaOp (tmp, AA52, AA22);
        AA23 = fmnaOp (tmp, AA53, AA23);
        AA24 = fmnaOp (tmp, AA54, AA24);
        AA25 = mulOp (negOp(tmp), AA55);
        AA26 = fmnaOp (tmp, AA56, AA26);
        AA27 = fmnaOp (tmp, AA57, AA27);
        AA28 = fmnaOp (tmp, AA58, AA28);

        tmp = AA35;
        AA30 = fmnaOp (tmp, AA50, AA30);
        AA31 = fmnaOp (tmp, AA51, AA31);
        AA32 = fmnaOp (tmp, AA52, AA32);
        AA33 = fmnaOp (tmp, AA53, AA33);
        AA34 = fmnaOp (tmp, AA54, AA34);
        AA35 = mulOp (negOp(tmp), AA55);
        AA36 = fmnaOp (tmp, AA56, AA36);
        AA37 = fmnaOp (tmp, AA57, AA37);
        AA38 = fmnaOp (tmp, AA58, AA38);

        tmp = AA45;
        AA40 = fmnaOp (tmp, AA50, AA40);
        AA41 = fmnaOp (tmp, AA51, AA41);
        AA42 = fmnaOp (tmp, AA52, AA42);
        AA43 = fmnaOp (tmp, AA53, AA43);
        AA44 = fmnaOp (tmp, AA54, AA44);
        AA45 = mulOp (negOp(tmp), AA55);
        AA46 = fmnaOp (tmp, AA56, AA46);
        AA47 = fmnaOp (tmp, AA57, AA47);
        AA48 = fmnaOp (tmp, AA58, AA48);

        tmp = AA65;
        AA60 = fmnaOp (tmp, AA50, AA60);
        AA61 = fmnaOp (tmp, AA51, AA61);
        AA62 = fmnaOp (tmp, AA52, AA62);
        AA63 = fmnaOp (tmp, AA53, AA63);
        AA64 = fmnaOp (tmp, AA54, AA64);
        AA65 = mulOp (negOp(tmp), AA55);
        AA66 = fmnaOp (tmp, AA56, AA66);
        AA67 = fmnaOp (tmp, AA57, AA67);
        AA68 = fmnaOp (tmp, AA58, AA68);

        tmp = AA75;
        AA70 = fmnaOp (tmp, AA50, AA70);
        AA71 = fmnaOp (tmp, AA51, AA71);
        AA72 = fmnaOp (tmp, AA52, AA72);
        AA73 = fmnaOp (tmp, AA53, AA73);
        AA74 = fmnaOp (tmp, AA54, AA74);
        AA75 = mulOp (negOp(tmp), AA55);
        AA76 = fmnaOp (tmp, AA56, AA76);
        AA77 = fmnaOp (tmp, AA57, AA77);
        AA78 = fmnaOp (tmp, AA58, AA78);

        tmp = AA85;
        AA80 = fmnaOp (tmp, AA50, AA80);
        AA81 = fmnaOp (tmp, AA51, AA81);
        AA82 = fmnaOp (tmp, AA52, AA82);
        AA83 = fmnaOp (tmp, AA53, AA83);
        AA84 = fmnaOp (tmp, AA54, AA84);
        AA85 = mulOp (negOp(tmp), AA55);
        AA86 = fmnaOp (tmp, AA56, AA86);
        AA87 = fmnaOp (tmp, AA57, AA87);
        AA88 = fmnaOp (tmp, AA58, AA88);

        /****************** iteration 6 ****************/

#if USE_PIVOTING
        /* search pivot row */
        p = absOp (AA66);
        pvt = 6;
        t = absOp (AA76);
        if (t > p) { p = t;  pvt = 7; }
        t = absOp (AA86);
        if (t > p) { p = t;  pvt = 8; }

        /* swap pivot row with row 6 */
        if (pvt == 7) {
            tmp = AA60;   AA60 = AA70;   AA70 = tmp;
            tmp = AA61;   AA61 = AA71;   AA71 = tmp;
            tmp = AA62;   AA62 = AA72;   AA72 = tmp;
            tmp = AA63;   AA63 = AA73;   AA73 = tmp;
            tmp = AA64;   AA64 = AA74;   AA74 = tmp;
            tmp = AA65;   AA65 = AA75;   AA75 = tmp;
            tmp = AA66;   AA66 = AA76;   AA76 = tmp;
            tmp = AA67;   AA67 = AA77;   AA77 = tmp;
            tmp = AA68;   AA68 = AA78;   AA78 = tmp;
            /* update permutation vector based on row swap */
            i = perm6;   perm6 = perm7;   perm7 = i;
        }
        if (pvt == 8) {
            tmp = AA60;   AA60 = AA80;   AA80 = tmp;
            tmp = AA61;   AA61 = AA81;   AA81 = tmp;
            tmp = AA62;   AA62 = AA82;   AA82 = tmp;
            tmp = AA63;   AA63 = AA83;   AA83 = tmp;
            tmp = AA64;   AA64 = AA84;   AA84 = tmp;
            tmp = AA65;   AA65 = AA85;   AA85 = tmp;
            tmp = AA66;   AA66 = AA86;   AA86 = tmp;
            tmp = AA67;   AA67 = AA87;   AA87 = tmp;
            tmp = AA68;   AA68 = AA88;   AA88 = tmp;
            /* update permutation vector based on row swap */
            i = perm6;   perm6 = perm8;   perm8 = i;
        }
#endif // USE_PIVOTING

        /* scale current row */
        tmp = rcpOp (AA66);
//        icol6 = perm6;
        AA60 = mulOp (tmp, AA60);
        AA61 = mulOp (tmp, AA61);
        AA62 = mulOp (tmp, AA62);
        AA63 = mulOp (tmp, AA63);
        AA64 = mulOp (tmp, AA64);
        AA65 = mulOp (tmp, AA65);
        AA66 = tmp;
        AA67 = mulOp (tmp, AA67);
        AA68 = mulOp (tmp, AA68);

        /* eliminate above and below current row */
        tmp = AA06;
        AA00 = fmnaOp (tmp, AA60, AA00);
        AA01 = fmnaOp (tmp, AA61, AA01);
        AA02 = fmnaOp (tmp, AA62, AA02);
        AA03 = fmnaOp (tmp, AA63, AA03);
        AA04 = fmnaOp (tmp, AA64, AA04);
        AA05 = fmnaOp (tmp, AA65, AA05);
        AA06 = mulOp (negOp(tmp), AA66);
        AA07 = fmnaOp (tmp, AA67, AA07);
        AA08 = fmnaOp (tmp, AA68, AA08);

        tmp = AA16;
        AA10 = fmnaOp (tmp, AA60, AA10);
        AA11 = fmnaOp (tmp, AA61, AA11);
        AA12 = fmnaOp (tmp, AA62, AA12);
        AA13 = fmnaOp (tmp, AA63, AA13);
        AA14 = fmnaOp (tmp, AA64, AA14);
        AA15 = fmnaOp (tmp, AA65, AA15);
        AA16 = mulOp (negOp(tmp), AA66);
        AA17 = fmnaOp (tmp, AA67, AA17);
        AA18 = fmnaOp (tmp, AA68, AA18);

        tmp = AA26;
        AA20 = fmnaOp (tmp, AA60, AA20);
        AA21 = fmnaOp (tmp, AA61, AA21);
        AA22 = fmnaOp (tmp, AA62, AA22);
        AA23 = fmnaOp (tmp, AA63, AA23);
        AA24 = fmnaOp (tmp, AA64, AA24);
        AA25 = fmnaOp (tmp, AA65, AA25);
        AA26 = mulOp (negOp(tmp), AA66);
        AA27 = fmnaOp (tmp, AA67, AA27);
        AA28 = fmnaOp (tmp, AA68, AA28);

        tmp = AA36;
        AA30 = fmnaOp (tmp, AA60, AA30);
        AA31 = fmnaOp (tmp, AA61, AA31);
        AA32 = fmnaOp (tmp, AA62, AA32);
        AA33 = fmnaOp (tmp, AA63, AA33);
        AA34 = fmnaOp (tmp, AA64, AA34);
        AA35 = fmnaOp (tmp, AA65, AA35);
        AA36 = mulOp (negOp(tmp), AA66);
        AA37 = fmnaOp (tmp, AA67, AA37);
        AA38 = fmnaOp (tmp, AA68, AA38);

        tmp = AA46;
        AA40 = fmnaOp (tmp, AA60, AA40);
        AA41 = fmnaOp (tmp, AA61, AA41);
        AA42 = fmnaOp (tmp, AA62, AA42);
        AA43 = fmnaOp (tmp, AA63, AA43);
        AA44 = fmnaOp (tmp, AA64, AA44);
        AA45 = fmnaOp (tmp, AA65, AA45);
        AA46 = mulOp (negOp(tmp), AA66);
        AA47 = fmnaOp (tmp, AA67, AA47);
        AA48 = fmnaOp (tmp, AA68, AA48);

        tmp = AA56;
        AA50 = fmnaOp (tmp, AA60, AA50);
        AA51 = fmnaOp (tmp, AA61, AA51);
        AA52 = fmnaOp (tmp, AA62, AA52);
        AA53 = fmnaOp (tmp, AA63, AA53);
        AA54 = fmnaOp (tmp, AA64, AA54);
        AA55 = fmnaOp (tmp, AA65, AA55);
        AA56 = mulOp (negOp(tmp), AA66);
        AA57 = fmnaOp (tmp, AA67, AA57);
        AA58 = fmnaOp (tmp, AA68, AA58);

        tmp = AA76;
        AA70 = fmnaOp (tmp, AA60, AA70);
        AA71 = fmnaOp (tmp, AA61, AA71);
        AA72 = fmnaOp (tmp, AA62, AA72);
        AA73 = fmnaOp (tmp, AA63, AA73);
        AA74 = fmnaOp (tmp, AA64, AA74);
        AA75 = fmnaOp (tmp, AA65, AA75);
        AA76 = mulOp (negOp(tmp), AA66);
        AA77 = fmnaOp (tmp, AA67, AA77);
        AA78 = fmnaOp (tmp, AA68, AA78);

        tmp = AA86;
        AA80 = fmnaOp (tmp, AA60, AA80);
        AA81 = fmnaOp (tmp, AA61, AA81);
        AA82 = fmnaOp (tmp, AA62, AA82);
        AA83 = fmnaOp (tmp, AA63, AA83);
        AA84 = fmnaOp (tmp, AA64, AA84);
        AA85 = fmnaOp (tmp, AA65, AA85);
        AA86 = mulOp (negOp(tmp), AA66);
        AA87 = fmnaOp (tmp, AA67, AA87);
        AA88 = fmnaOp (tmp, AA68, AA88);

        /****************** iteration 7 ****************/

#if USE_PIVOTING
        /* search pivot row */
        p = absOp (AA77);
        pvt = 7;
        t = absOp (AA87);
        if (t > p) { p = t;  pvt = 8; }

        /* swap pivot row with row 7 */
        if (pvt == 8) {
            tmp = AA70;   AA70 = AA80;   AA80 = tmp;
            tmp = AA71;   AA71 = AA81;   AA81 = tmp;
            tmp = AA72;   AA72 = AA82;   AA82 = tmp;
            tmp = AA73;   AA73 = AA83;   AA83 = tmp;
            tmp = AA74;   AA74 = AA84;   AA84 = tmp;
            tmp = AA75;   AA75 = AA85;   AA85 = tmp;
            tmp = AA76;   AA76 = AA86;   AA86 = tmp;
            tmp = AA77;   AA77 = AA87;   AA87 = tmp;
            tmp = AA78;   AA78 = AA88;   AA88 = tmp;
            /* update permutation vector based on row swap */
            i = perm7;   perm7 = perm8;   perm8 = i;
        }
#endif // USE_PIVOTING

        /* scale current row */
        tmp = rcpOp (AA77);
//        icol7 = perm7;
        AA70 = mulOp (tmp, AA70);
        AA71 = mulOp (tmp, AA71);
        AA72 = mulOp (tmp, AA72);
        AA73 = mulOp (tmp, AA73);
        AA74 = mulOp (tmp, AA74);
        AA75 = mulOp (tmp, AA75);
        AA76 = mulOp (tmp, AA76);
        AA77 = tmp;
        AA78 = mulOp (tmp, AA78);

        /* eliminate above and below current row */
        tmp = AA07;
        AA00 = fmnaOp (tmp, AA70, AA00);
        AA01 = fmnaOp (tmp, AA71, AA01);
        AA02 = fmnaOp (tmp, AA72, AA02);
        AA03 = fmnaOp (tmp, AA73, AA03);
        AA04 = fmnaOp (tmp, AA74, AA04);
        AA05 = fmnaOp (tmp, AA75, AA05);
        AA06 = fmnaOp (tmp, AA76, AA06);
        AA07 = mulOp (negOp(tmp), AA77);
        AA08 = fmnaOp (tmp, AA78, AA08);

        tmp = AA17;
        AA10 = fmnaOp (tmp, AA70, AA10);
        AA11 = fmnaOp (tmp, AA71, AA11);
        AA12 = fmnaOp (tmp, AA72, AA12);
        AA13 = fmnaOp (tmp, AA73, AA13);
        AA14 = fmnaOp (tmp, AA74, AA14);
        AA15 = fmnaOp (tmp, AA75, AA15);
        AA16 = fmnaOp (tmp, AA76, AA16);
        AA17 = mulOp (negOp(tmp), AA77);
        AA18 = fmnaOp (tmp, AA78, AA18);

        tmp = AA27;
        AA20 = fmnaOp (tmp, AA70, AA20);
        AA21 = fmnaOp (tmp, AA71, AA21);
        AA22 = fmnaOp (tmp, AA72, AA22);
        AA23 = fmnaOp (tmp, AA73, AA23);
        AA24 = fmnaOp (tmp, AA74, AA24);
        AA25 = fmnaOp (tmp, AA75, AA25);
        AA26 = fmnaOp (tmp, AA76, AA26);
        AA27 = mulOp (negOp(tmp), AA77);
        AA28 = fmnaOp (tmp, AA78, AA28);

        tmp = AA37;
        AA30 = fmnaOp (tmp, AA70, AA30);
        AA31 = fmnaOp (tmp, AA71, AA31);
        AA32 = fmnaOp (tmp, AA72, AA32);
        AA33 = fmnaOp (tmp, AA73, AA33);
        AA34 = fmnaOp (tmp, AA74, AA34);
        AA35 = fmnaOp (tmp, AA75, AA35);
        AA36 = fmnaOp (tmp, AA76, AA36);
        AA37 = mulOp (negOp(tmp), AA77);
        AA38 = fmnaOp (tmp, AA78, AA38);

        tmp = AA47;
        AA40 = fmnaOp (tmp, AA70, AA40);
        AA41 = fmnaOp (tmp, AA71, AA41);
        AA42 = fmnaOp (tmp, AA72, AA42);
        AA43 = fmnaOp (tmp, AA73, AA43);
        AA44 = fmnaOp (tmp, AA74, AA44);
        AA45 = fmnaOp (tmp, AA75, AA45);
        AA46 = fmnaOp (tmp, AA76, AA46);
        AA47 = mulOp (negOp(tmp), AA77);
        AA48 = fmnaOp (tmp, AA78, AA48);

        tmp = AA57;
        AA50 = fmnaOp (tmp, AA70, AA50);
        AA51 = fmnaOp (tmp, AA71, AA51);
        AA52 = fmnaOp (tmp, AA72, AA52);
        AA53 = fmnaOp (tmp, AA73, AA53);
        AA54 = fmnaOp (tmp, AA74, AA54);
        AA55 = fmnaOp (tmp, AA75, AA55);
        AA56 = fmnaOp (tmp, AA76, AA56);
        AA57 = mulOp (negOp(tmp), AA77);
        AA58 = fmnaOp (tmp, AA78, AA58);

        tmp = AA67;
        AA60 = fmnaOp (tmp, AA70, AA60);
        AA61 = fmnaOp (tmp, AA71, AA61);
        AA62 = fmnaOp (tmp, AA72, AA62);
        AA63 = fmnaOp (tmp, AA73, AA63);
        AA64 = fmnaOp (tmp, AA74, AA64);
        AA65 = fmnaOp (tmp, AA75, AA65);
        AA66 = fmnaOp (tmp, AA76, AA66);
        AA67 = mulOp (negOp(tmp), AA77);
        AA68 = fmnaOp (tmp, AA78, AA68);

        tmp = AA87;
        AA80 = fmnaOp (tmp, AA70, AA80);
        AA81 = fmnaOp (tmp, AA71, AA81);
        AA82 = fmnaOp (tmp, AA72, AA82);
        AA83 = fmnaOp (tmp, AA73, AA83);
        AA84 = fmnaOp (tmp, AA74, AA84);
        AA85 = fmnaOp (tmp, AA75, AA85);
        AA86 = fmnaOp (tmp, AA76, AA86);
        AA87 = mulOp (negOp(tmp), AA77);
        AA88 = fmnaOp (tmp, AA78, AA88);

        /****************** iteration 8 ****************/

        /* scale current row */
        tmp = rcpOp (AA88);
//        icol8 = perm8;
        AA80 = mulOp (tmp, AA80);
        AA81 = mulOp (tmp, AA81);
        AA82 = mulOp (tmp, AA82);
        AA83 = mulOp (tmp, AA83);
        AA84 = mulOp (tmp, AA84);
        AA85 = mulOp (tmp, AA85);
        AA86 = mulOp (tmp, AA86);
        AA87 = mulOp (tmp, AA87);
        AA88 = tmp;

        /* eliminate above and below current row */
        tmp = AA08;
        AA00 = fmnaOp (tmp, AA80, AA00);
        AA01 = fmnaOp (tmp, AA81, AA01);
        AA02 = fmnaOp (tmp, AA82, AA02);
        AA03 = fmnaOp (tmp, AA83, AA03);
        AA04 = fmnaOp (tmp, AA84, AA04);
        AA05 = fmnaOp (tmp, AA85, AA05);
        AA06 = fmnaOp (tmp, AA86, AA06);
        AA07 = fmnaOp (tmp, AA87, AA07);
        AA08 = mulOp (negOp(tmp), AA88);

        tmp = AA18;
        AA10 = fmnaOp (tmp, AA80, AA10);
        AA11 = fmnaOp (tmp, AA81, AA11);
        AA12 = fmnaOp (tmp, AA82, AA12);
        AA13 = fmnaOp (tmp, AA83, AA13);
        AA14 = fmnaOp (tmp, AA84, AA14);
        AA15 = fmnaOp (tmp, AA85, AA15);
        AA16 = fmnaOp (tmp, AA86, AA16);
        AA17 = fmnaOp (tmp, AA87, AA17);
        AA18 = mulOp (negOp(tmp), AA88);

        tmp = AA28;
        AA20 = fmnaOp (tmp, AA80, AA20);
        AA21 = fmnaOp (tmp, AA81, AA21);
        AA22 = fmnaOp (tmp, AA82, AA22);
        AA23 = fmnaOp (tmp, AA83, AA23);
        AA24 = fmnaOp (tmp, AA84, AA24);
        AA25 = fmnaOp (tmp, AA85, AA25);
        AA26 = fmnaOp (tmp, AA86, AA26);
        AA27 = fmnaOp (tmp, AA87, AA27);
        AA28 = mulOp (negOp(tmp), AA88);

        tmp = AA38;
        AA30 = fmnaOp (tmp, AA80, AA30);
        AA31 = fmnaOp (tmp, AA81, AA31);
        AA32 = fmnaOp (tmp, AA82, AA32);
        AA33 = fmnaOp (tmp, AA83, AA33);
        AA34 = fmnaOp (tmp, AA84, AA34);
        AA35 = fmnaOp (tmp, AA85, AA35);
        AA36 = fmnaOp (tmp, AA86, AA36);
        AA37 = fmnaOp (tmp, AA87, AA37);
        AA38 = mulOp (negOp(tmp), AA88);

        tmp = AA48;
        AA40 = fmnaOp (tmp, AA80, AA40);
        AA41 = fmnaOp (tmp, AA81, AA41);
        AA42 = fmnaOp (tmp, AA82, AA42);
        AA43 = fmnaOp (tmp, AA83, AA43);
        AA44 = fmnaOp (tmp, AA84, AA44);
        AA45 = fmnaOp (tmp, AA85, AA45);
        AA46 = fmnaOp (tmp, AA86, AA46);
        AA47 = fmnaOp (tmp, AA87, AA47);
        AA48 = mulOp (negOp(tmp), AA88);

        tmp = AA58;
        AA50 = fmnaOp (tmp, AA80, AA50);
        AA51 = fmnaOp (tmp, AA81, AA51);
        AA52 = fmnaOp (tmp, AA82, AA52);
        AA53 = fmnaOp (tmp, AA83, AA53);
        AA54 = fmnaOp (tmp, AA84, AA54);
        AA55 = fmnaOp (tmp, AA85, AA55);
        AA56 = fmnaOp (tmp, AA86, AA56);
        AA57 = fmnaOp (tmp, AA87, AA57);
        AA58 = mulOp (negOp(tmp), AA88);

        tmp = AA68;
        AA60 = fmnaOp (tmp, AA80, AA60);
        AA61 = fmnaOp (tmp, AA81, AA61);
        AA62 = fmnaOp (tmp, AA82, AA62);
        AA63 = fmnaOp (tmp, AA83, AA63);
        AA64 = fmnaOp (tmp, AA84, AA64);
        AA65 = fmnaOp (tmp, AA85, AA65);
        AA66 = fmnaOp (tmp, AA86, AA66);
        AA67 = fmnaOp (tmp, AA87, AA67);
        AA68 = mulOp (negOp(tmp), AA88);

        tmp = AA78;
        AA70 = fmnaOp (tmp, AA80, AA70);
        AA71 = fmnaOp (tmp, AA81, AA71);
        AA72 = fmnaOp (tmp, AA82, AA72);
        AA73 = fmnaOp (tmp, AA83, AA73);
        AA74 = fmnaOp (tmp, AA84, AA74);
        AA75 = fmnaOp (tmp, AA85, AA75);
        AA76 = fmnaOp (tmp, AA86, AA76);
        AA77 = fmnaOp (tmp, AA87, AA77);
        AA78 = mulOp (negOp(tmp), AA88);

        /* sort columns into the correct order */
       B[(0) * batch + thrdNum] = AA00;
       B[(1) * batch + thrdNum] = AA10;
       B[(2) * batch + thrdNum] = AA20;
       B[(3) * batch + thrdNum] = AA30;
       B[(4) * batch + thrdNum] = AA40;
       B[(5) * batch + thrdNum] = AA50;
       B[(6) * batch + thrdNum] = AA60;
       B[(7) * batch + thrdNum] = AA70;
       B[(8) * batch + thrdNum] = AA80;
       //B[(9*icol1 + 0) * batch + thrdNum] = AA01;
       B[(9) * batch + thrdNum] = AA11;
       B[(10) * batch + thrdNum] = AA21;
       B[(11) * batch + thrdNum] = AA31;
       B[(12) * batch + thrdNum] = AA41;
       B[(13) * batch + thrdNum] = AA51;
       B[(14) * batch + thrdNum] = AA61;
       B[(15) * batch + thrdNum] = AA71;
       B[(16) * batch + thrdNum] = AA81;       
	   //B[(9*icol2 + 0) * batch + thrdNum] = AA02;
       //B[(9*icol2 + 1) * batch + thrdNum] = AA12;
       B[(17) * batch + thrdNum] = AA22;
       B[(18) * batch + thrdNum] = AA32;
       B[(19) * batch + thrdNum] = AA42;
       B[(20) * batch + thrdNum] = AA52;
       B[(21) * batch + thrdNum] = AA62;
       B[(22) * batch + thrdNum] = AA72;
       B[(23) * batch + thrdNum] = AA82;
       //B[(9*icol3 + 0) * batch + thrdNum] = AA03;
       //B[(9*icol3 + 1) * batch + thrdNum] = AA13;
       //B[(9*icol3 + 2) * batch + thrdNum] = AA23;
       B[(24) * batch + thrdNum] = AA33;
       B[(25) * batch + thrdNum] = AA43;
       B[(26) * batch + thrdNum] = AA53;
       B[(27) * batch + thrdNum] = AA63;
       B[(28) * batch + thrdNum] = AA73;
       B[(29) * batch + thrdNum] = AA83;
       //B[(9*icol4 + 0) * batch + thrdNum] = AA04;
       //B[(9*icol4 + 1) * batch + thrdNum] = AA14;
       //B[(9*icol4 + 2) * batch + thrdNum] = AA24;
       //B[(9*icol4 + 3) * batch + thrdNum] = AA34;
       B[(30) * batch + thrdNum] = AA44;
       B[(31) * batch + thrdNum] = AA54;
       B[(32) * batch + thrdNum] = AA64;
       B[(33) * batch + thrdNum] = AA74;
       B[(34) * batch + thrdNum] = AA84;
       //B[(9*icol5 + 0) * batch + thrdNum] = AA05;
       //B[(9*icol5 + 1) * batch + thrdNum] = AA15;
       //B[(9*icol5 + 2) * batch + thrdNum] = AA25;
       //B[(9*icol5 + 3) * batch + thrdNum] = AA35;
       //B[(9*icol5 + 4) * batch + thrdNum] = AA45;
       B[(35) * batch + thrdNum] = AA55;
       B[(36) * batch + thrdNum] = AA65;
       B[(37) * batch + thrdNum] = AA75;
       B[(38) * batch + thrdNum] = AA85;
       //B[(9*icol6 + 0) * batch + thrdNum] = AA06;
       //B[(9*icol6 + 1) * batch + thrdNum] = AA16;
       //B[(9*icol6 + 2) * batch + thrdNum] = AA26;
       //B[(9*icol6 + 3) * batch + thrdNum] = AA36;
       //B[(9*icol6 + 4) * batch + thrdNum] = AA46;
       //B[(9*icol6 + 5) * batch + thrdNum] = AA56;
       B[(39) * batch + thrdNum] = AA66;
       B[(40) * batch + thrdNum] = AA76;
       B[(41) * batch + thrdNum] = AA86;
       //B[(9*icol7 + 0) * batch + thrdNum] = AA07;
       //B[(9*icol7 + 1) * batch + thrdNum] = AA17;
       //B[(9*icol7 + 2) * batch + thrdNum] = AA27;
       //B[(9*icol7 + 3) * batch + thrdNum] = AA37;
       //B[(9*icol7 + 4) * batch + thrdNum] = AA47;
       //B[(9*icol7 + 5) * batch + thrdNum] = AA57;
       //B[(9*icol7 + 6) * batch + thrdNum] = AA67;
       B[(42) * batch + thrdNum] = AA77;
       B[(43) * batch + thrdNum] = AA87;
       //B[(9*icol8 + 0) * batch + thrdNum] = AA08;
       //B[(9*icol8 + 1) * batch + thrdNum] = AA18;
       //B[(9*icol8 + 2) * batch + thrdNum] = AA28;
       //B[(9*icol8 + 3) * batch + thrdNum] = AA38;
       //B[(9*icol8 + 4) * batch + thrdNum] = AA48;
       //B[(9*icol8 + 5) * batch + thrdNum] = AA58;
       //B[(9*icol8 + 6) * batch + thrdNum] = AA68;
       //B[(9*icol8 + 7) * batch + thrdNum] = AA78;
       B[(44) * batch + thrdNum] = AA88;
    }
}
