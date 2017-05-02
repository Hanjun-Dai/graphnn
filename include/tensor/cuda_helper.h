#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#ifdef USE_GPU
#include <curand.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

__device__ inline int get_sp_row_idx(int i, int* row_ptr, int n_rows)
{
    int l = 0, r = n_rows - 1, row;
    while (l <= r)
    {
        row = (l + r) / 2;
        if (row_ptr[row] <= i)
        {
            if (row_ptr[row + 1] > i)
                break;
            else 
                l = row + 1;
        } else r = row - 1;
    }
    return row;
}

__device__ inline float cuda_rsqrt(const float& src)
{
        return rsqrtf(src);
}

__device__ inline float cuda_rsqrt(const double& src)
{
        return rsqrt(src);
}

__device__ inline float cuda_pow(const float& x, const float& y)
{
        return powf(x, y);
}

__device__ inline double cuda_pow(const double& x, const double& y)
{
        return pow(x, y);
}

__device__ inline float cuda_fabs(const float& src)
{
        return fabsf(src);
}

__device__ inline double cuda_fabs(const double& src)
{
        return fabs(src);
}

__device__ inline float cuda_exp(const float& src)
{
        return expf(src);
}
    
__device__ inline double cuda_exp(const double& src)
{
        return exp(src);
}

__device__ inline float cuda_log(const float& src)
{
        return logf(src);
}
    
__device__ inline double cuda_log(const double& src)
{
        return log(src);
}

inline float Cuda_Dot(cublasHandle_t& handle, int n, const float *x, const float* y)
{
        float result;
        cublasSdot(handle, n, x, 1, y, 1, &result);
        return result;
}

inline double Cuda_Dot(cublasHandle_t& handle, int n, const double *x, const double* y)
{
        double result;
        cublasDdot(handle, n, x, 1, y, 1, &result);
        return result;
}

inline float Cuda_Norm2(cublasHandle_t& handle, int n, const float *x)
{
		float result;
		cublasSnrm2(handle, n, x, 1, &result);
		return result;
}

inline double Cuda_Norm2(cublasHandle_t& handle, int n, const double *x)
{
		double result;
		cublasDnrm2(handle, n, x, 1, &result);
		return result;
}

inline void Cuda_Amax(cublasHandle_t& handle, int n, const float *x, int* result)
{
        cublasIsamax(handle, n, x, 1, result);
}

inline void Cuda_Amax(cublasHandle_t& handle, int n, const double *x, int* result)
{
        cublasIdamax(handle, n, x, 1, result);
}

inline float Cuda_Asum(cublasHandle_t& handle, int n, const float *x)
{
		float result;
		cublasSasum(handle, n, x, 1, &result);
		return result;
}

inline double Cuda_Asum(cublasHandle_t& handle, int n, const double *x)
{
		double result;
		cublasDasum(handle, n, x, 1, &result);
		return result;
}

inline void Cuda_Ger(cublasHandle_t& handle, int m, int n, const float* alpha, const float* x, const float* y, float* A)
{
		cublasSger(handle, m, n, alpha, x, 1, y, 1, A, m);
}

inline void Cuda_Ger(cublasHandle_t& handle, int m, int n, const double* alpha, const double* x, const double* y, double* A)
{
		cublasDger(handle, m, n, alpha, x, 1, y, 1, A, m);
}

inline void Cuda_GeMV(cublasHandle_t& handle, cublasOperation_t trans, 
                            int m, int n, 
                            const float* alpha, const float* A, int lda, 
                            const float *x, int incx, 
                            const float *beta, float* y, int incy)
{
        cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline void Cuda_GeMV(cublasHandle_t& handle, cublasOperation_t trans, 
                            int m, int n, 
                            const double* alpha, const double* A, int lda, 
                            const double *x, int incx, 
                            const double *beta, double* y, int incy)
{
        cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline void Cuda_GeaM(cublasHandle_t& handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
							const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb,  
							float* C, int ldc)
{
		cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

inline void Cuda_GeaM(cublasHandle_t& handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
							const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, 
							double* C, int ldc)
{
		cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

inline void Cuda_CSRMM(cusparseHandle_t& handle, cusparseOperation_t transA, int m, int n, int k, int nnz, 
                    const float *alpha, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, 
                    const float *B, int ldb, const float *beta, float *C, int ldc)
{
        cusparseMatDescr_t descrA;
        cusparseCreateMatDescr(&descrA);
        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
        cusparseScsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
}

inline void Cuda_CSRMM(cusparseHandle_t& handle, cusparseOperation_t transA, int m, int n, int k, int nnz, 
                    const double *alpha, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, 
                    const double *B, int ldb, const double *beta, double *C, int ldc)
{
        cusparseMatDescr_t descrA;
        cusparseCreateMatDescr(&descrA);
        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
        cusparseDcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);       
}                    

inline void Cuda_Axpy(cublasHandle_t& handle, int n, const float *alpha, const float *x, float *y)
{
		cublasSaxpy(handle, n, alpha, x, 1, y, 1);
}

inline void Cuda_Axpy(cublasHandle_t& handle, int n, const double *alpha, const double *x, double *y)
{
		cublasDaxpy(handle, n, alpha, x, 1, y, 1);
}

inline void Cuda_SetRandNormal(curandGenerator_t& generator, float* outputPtr, size_t n, float mean, float stddev)
{
		curandGenerateNormal(generator, outputPtr, n, mean, stddev);
}

inline void Cuda_SetRandNormal(curandGenerator_t& generator, double* outputPtr, size_t n, double mean, double stddev)
{
		curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
}

#endif

#endif