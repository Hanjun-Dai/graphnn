/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



#ifndef FWT_KERNEL_CUH_DOUBLE
#define FWT_KERNEL_CUH_DOUBLE
#ifndef fwt_kernel_cuh_double
#define fwt_kernel_cuh_double



///////////////////////////////////////////////////////////////////////////////
// Elementary(for vectors less than elementary size) in-shared memory
// combined radix-2 + radix-4 Fast Walsh Transform
///////////////////////////////////////////////////////////////////////////////
#define ELEMENTARY_LOG2SIZE 11

__global__ void fwtBatch1Kernel(double *d_Output, double *d_Input, int log2N)
{
    const int    N = 1 << log2N;
    const int base = blockIdx.x << log2N;

    //(2 ** 11) * 4 bytes == 8KB -- maximum d_data[] size for G80
    extern __shared__ double d_data[];
    double *d_Src = d_Input  + base;
    double *d_Dst = d_Output + base;

    for (int pos = threadIdx.x; pos < N; pos += blockDim.x)
    {
        d_data[pos] = d_Src[pos];
    }

    //Main radix-4 stages
    const int pos = threadIdx.x;

    for (int stride = N >> 2; stride > 0; stride >>= 2)
    {
        int lo = pos & (stride - 1);
        int i0 = ((pos - lo) << 2) + lo;
        int i1 = i0 + stride;
        int i2 = i1 + stride;
        int i3 = i2 + stride;

        __syncthreads();
        double D0 = d_data[i0];
        double D1 = d_data[i1];
        double D2 = d_data[i2];
        double D3 = d_data[i3];

        double T;
        T = D0;
        D0         = D0 + D2;
        D2         = T - D2;
        T = D1;
        D1         = D1 + D3;
        D3         = T - D3;
        T = D0;
        d_data[i0] = D0 + D1;
        d_data[i1] = T - D1;
        T = D2;
        d_data[i2] = D2 + D3;
        d_data[i3] = T - D3;
    }

    //Do single radix-2 stage for odd power of two
    if (log2N & 1)
    {
        __syncthreads();

        for (int pos = threadIdx.x; pos < N / 2; pos += blockDim.x)
        {
            int i0 = pos << 1;
            int i1 = i0 + 1;

            double D0 = d_data[i0];
            double D1 = d_data[i1];
            d_data[i0] = D0 + D1;
            d_data[i1] = D0 - D1;
        }
    }

    __syncthreads();

    for (int pos = threadIdx.x; pos < N; pos += blockDim.x)
    {
        d_Dst[pos] = d_data[pos];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Single in-global memory radix-4 Fast Walsh Transform pass
// (for strides exceeding elementary vector size)
////////////////////////////////////////////////////////////////////////////////
__global__ void fwtBatch2Kernel(
    double *d_Output,
    double *d_Input,
    int stride
)
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int   N = blockDim.x *  gridDim.x * 4;

    double *d_Src = d_Input  + blockIdx.y * N;
    double *d_Dst = d_Output + blockIdx.y * N;

    int lo = pos & (stride - 1);
    int i0 = ((pos - lo) << 2) + lo;
    int i1 = i0 + stride;
    int i2 = i1 + stride;
    int i3 = i2 + stride;

    double D0 = d_Src[i0];
    double D1 = d_Src[i1];
    double D2 = d_Src[i2];
    double D3 = d_Src[i3];

    double T;
    T = D0;
    D0        = D0 + D2;
    D2        = T - D2;
    T = D1;
    D1        = D1 + D3;
    D3        = T - D3;
    T = D0;
    d_Dst[i0] = D0 + D1;
    d_Dst[i1] = T - D1;
    T = D2;
    d_Dst[i2] = D2 + D3;
    d_Dst[i3] = T - D3;
}

////////////////////////////////////////////////////////////////////////////////
// Put everything together: batched Fast Walsh Transform CPU front-end
////////////////////////////////////////////////////////////////////////////////
void fwtBatchGPU(double *d_Data, int M, int log2N)
{
    const int THREAD_N = 1024;

    int N = 1 << log2N;
    dim3 grid((1 << log2N) / (4 * THREAD_N), M, 1);

    for (; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2)
    {
        fwtBatch2Kernel<<<grid, THREAD_N>>>(d_Data, d_Data, N / 4);
    }

    fwtBatch1Kernel<<<M, N / 4, N *sizeof(double)>>>(
        d_Data,
        d_Data,
        log2N
    );
}


////////////////////////////////////////////////////////////////////////////////
// Modulate two arrays
////////////////////////////////////////////////////////////////////////////////
__global__ void modulateKernel(double *d_A, double *d_B, int N)
{
    int        tid = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * gridDim.x;
    double     rcpN = 1.0f / (double)N;

    for (int pos = tid; pos < N; pos += numThreads)
    {
        d_A[pos] *= d_B[pos] * rcpN;
    }
}

//Interface to modulateKernel()
void modulateGPU(double *d_A, double *d_B, int N)
{
    modulateKernel<<<128, 256>>>(d_A, d_B, N);
}



#endif
#endif
