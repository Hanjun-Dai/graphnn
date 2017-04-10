#ifndef MKL_HELPER_H
#define MKL_HELPER_H

#include <mkl.h>

namespace gnn
{

inline float MKL_Dot(const MKL_INT n, const float* x, const float* y)
{
    return cblas_sdot(n, x, 1, y, 1);
}

inline double MKL_Dot(const MKL_INT n, const double* x, const double* y)
{
    return cblas_ddot(n, x, 1, y, 1);
}

inline CBLAS_INDEX MKL_Amax(const MKL_INT n, const float *x)
{
    return cblas_isamax(n, x, 1);
}

inline CBLAS_INDEX MKL_Amax(const MKL_INT n, const double *x)
{
    return cblas_idamax(n, x, 1);
}

inline float MKL_ASum(const MKL_INT n, const float *x)
{
		return cblas_sasum(n, x, 1);
}

inline double MKL_ASum(const MKL_INT n, const double *x)
{
		return cblas_dasum(n, x, 1);
}

inline float MKL_Norm2(const MKL_INT n, const float *x)
{
		return cblas_snrm2(n, x, 1);
}

inline double MKL_Norm2(const MKL_INT n, const double *x)
{
		return cblas_dnrm2(n, x, 1);
}

inline void MKL_Ger(const CBLAS_LAYOUT Layout, const MKL_INT m, const MKL_INT n, 
						const float alpha, const float *x, const float *y, float *a)
{
	const MKL_INT lda = Layout == CblasRowMajor ? n : m;
	cblas_sger(Layout, m, n, alpha, x, 1, y, 1, a, lda);
}

inline void MKL_Ger(const CBLAS_LAYOUT Layout, const MKL_INT m, const MKL_INT n, 
						const double alpha, const double *x, const double *y, double *a)
{
	const MKL_INT lda = Layout == CblasRowMajor ? n : m;
	cblas_dger(Layout, m, n, alpha, x, 1, y, 1, a, lda);
}

inline void MKL_Axpy(const MKL_INT n, const float a, const float *x, float *y)
{
	cblas_saxpy(n, a, x, 1, y, 1);
}

inline void MKL_Axpy(const MKL_INT n, const double a, const double *x, double *y)
{
	cblas_daxpy(n, a, x, 1, y, 1);
}

inline void MKL_Axpby(const MKL_INT n, const float a, const float *x, const float b, float *y)
{
	cblas_saxpby(n, a, x, 1, b, y, 1);
}

inline void MKL_Axpby(const MKL_INT n, const double a, const double *x, const double b, double *y)
{
	cblas_daxpby(n, a, x, 1, b, y, 1);
}

inline void MKL_Omatadd(char ordering, char transa, char transb, size_t m, size_t n, 
							const float alpha, const float * A, size_t lda, 
							const float beta, const float * B, size_t ldb, 
							float * C, size_t ldc)
{
	mkl_somatadd(ordering, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

inline void MKL_Omatadd(char ordering, char transa, char transb, size_t m, size_t n, 
							const double alpha, const double * A, size_t lda, 
							const double beta, const double * B, size_t ldb, 
							double * C, size_t ldc)
{
	mkl_domatadd(ordering, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

inline void MKL_GeMV(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE trans, 
                           const MKL_INT m, const MKL_INT n,
                           const float alpha, const float *a, const MKL_INT lda, const float *x, const MKL_INT incx,
                           const float beta, float *y, const MKL_INT incy)
{
    cblas_sgemv(Layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);   
}

inline void MKL_GeMV(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE trans, 
                           const MKL_INT m, const MKL_INT n,
                           const double alpha, const double *a, const MKL_INT lda, const double *x, const MKL_INT incx,
                           const double beta, double *y, const MKL_INT incy)
{
    cblas_dgemv(Layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);   
} 

inline void MKL_GeMM(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, 
							const MKL_INT m, const MKL_INT n, const MKL_INT k, 
							const float alpha, const float *a, const MKL_INT lda, const float *b, const MKL_INT ldb, 
							const float beta, float *c, const MKL_INT ldc)
{
	cblas_sgemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void MKL_GeMM(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, 
							const MKL_INT m, const MKL_INT n, const MKL_INT k, 
							const double alpha, const double *a, const MKL_INT lda, const double *b, const MKL_INT ldb, 
							const double beta, double *c, const MKL_INT ldc)
{
	cblas_dgemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void MKL_CSRMM(char trans, MKL_INT m, MKL_INT n, MKL_INT k, float alpha, 
							char *matdescra, float *val, MKL_INT *indx, MKL_INT *pntrb, MKL_INT *pntre, 
						    float *b, MKL_INT ldb, float beta, float *c, MKL_INT ldc)
{
	mkl_scsrmm(&trans, &m, &n, &k, &alpha,
			matdescra, val, indx, pntrb, pntre, 
			b, &ldb, &beta, c, &ldc);
}

inline void MKL_CSRMM(char trans, MKL_INT m, MKL_INT n, MKL_INT k, double alpha, 
							char *matdescra, double *val, MKL_INT *indx, MKL_INT *pntrb, MKL_INT *pntre, 
							double *b, MKL_INT ldb, double beta, double *c, MKL_INT ldc)
{
	mkl_dcsrmm(&trans, &m, &n, &k, &alpha,
			matdescra, val, indx, pntrb, pntre, 
			b, &ldb, &beta, c, &ldc);
}

inline void MKL_Abs(const MKL_INT n, float* a, float* y)
{
    vsAbs(n, a, y);
}

inline void MKL_Abs(const MKL_INT n, double* a, double* y)
{
    vdAbs(n, a, y);
}

inline void MKL_Sin(const MKL_INT n, float* a, float* y)
{
    vsSin(n, a, y);
}

inline void MKL_Sin(const MKL_INT n, double* a, double* y)
{
    vdSin(n, a, y);
}

inline void MKL_Cos(const MKL_INT n, float* a, float* y)
{
    vsCos(n, a, y);
}

inline void MKL_Cos(const MKL_INT n, double* a, double* y)
{
    vdCos(n, a, y);
}

inline void MKL_Exp(const MKL_INT n, float* a, float* y)
{
	vsExp(n, a, y);
}

inline void MKL_Exp(const MKL_INT n, double* a, double* y)
{
	vdExp(n, a, y);
}

inline void MKL_Log(const MKL_INT n, float* a, float* y)
{
	vsLn(n, a, y);
}

inline void MKL_Log(const MKL_INT n, double* a, double* y)
{
	vdLn(n, a, y);
}

inline void MKL_Mul(const MKL_INT n, float* a, float* b, float* y)
{
	vsMul(n, a, b, y);
}

inline void MKL_Mul(const MKL_INT n, double* a, double* b, double* y)
{
	vdMul(n, a, b, y);
}

inline void MKL_Div(const MKL_INT n, float* a, float* b, float* y)
{
	vsDiv(n, a, b, y);
}

inline void MKL_Div(const MKL_INT n, double* a, double* b, double* y)
{
	vdDiv(n, a, b, y);
}

inline void MKL_Sqrt(const MKL_INT n, float* a, float* y)
{
	vsSqrt(n, a, y);
}

inline void MKL_Sqrt(const MKL_INT n, double* a, double* y)
{
	vdSqrt(n, a, y);
}

inline void MKL_InvSqrt(const MKL_INT n, float* a, float* y)
{
	vsInvSqrt(n, a, y);
}

inline void MKL_InvSqrt(const MKL_INT n, double* a, double* y)
{
	vdInvSqrt(n, a, y);
}

inline void MKL_Inv(const MKL_INT n, float* a, float* y)
{
	vsInv(n, a, y);
}

inline void MKL_Inv(const MKL_INT n, double* a, double* y)
{
	vdInv(n, a, y);
}

inline void MKL_Square(const MKL_INT n, float* a, float* y)
{
    vsSqr(n, a, y);
}

inline void MKL_Square(const MKL_INT n, double* a, double* y)
{
    vdSqr(n, a, y);
}

inline void MKL_PowerX(const MKL_INT n, float* a, float b, float* y)
{
    vsPowx(n, a, b, y);
}

inline void MKL_PowerX(const MKL_INT n, double* a, double b, double* y)
{
    vdPowx(n, a, b, y);
}

}

#endif