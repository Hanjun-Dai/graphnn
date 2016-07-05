#ifndef IMATRIX_H
#define IMATRIX_H

#include "matrix_utils.h"
#include "gpuhandle.h"
#include <stdexcept>
#define GPU_T(x) (x == Trans::N ? cublasOperation_t::CUBLAS_OP_N : cublasOperation_t::CUBLAS_OP_T)
#define CUSP_T(x) (x == Trans::N ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE)
#define CPU_CharT(x) (x == Trans::N ? 'N' : 'T')
#define CPU_T(x) (x == Trans::N ? CblasNoTrans : CblasTrans)

template<MatMode mode, typename Dtype>
class SparseMat;
template<MatMode mode, typename Dtype>
class DenseMat;

template<MatMode mode, typename Dtype>
class IMatrix 
{
public:
		virtual MatType GetMatType() = 0;
		virtual ~IMatrix() {}
		
		virtual void Serialize(FILE* fid) 
		{
			assert(fwrite(&rows, sizeof(size_t), 1, fid) == 1);
			assert(fwrite(&cols, sizeof(size_t), 1, fid) == 1);
			assert(fwrite(&count, sizeof(size_t), 1, fid) == 1);
		}
		
		virtual void Deserialize(FILE* fid)
		{
			assert(fread(&rows, sizeof(size_t), 1, fid) == 1);
			assert(fread(&cols, sizeof(size_t), 1, fid) == 1);
			assert(fread(&count, sizeof(size_t), 1, fid) == 1);
		}
        
        virtual void Print2Screen() = 0;
		
		virtual DenseMat<mode, Dtype>& DenseDerived() 
		{
			throw "Can not derive Dense Matrix from CSR Matrix";
		}
		
		virtual const DenseMat<mode, Dtype>& DenseDerived() const 
		{
			throw "Can not derive Dense Matrix from CSR Matrix";
		}
		
		virtual SparseMat<mode, Dtype>& SparseDerived()
		{
			throw "Can not derive CSR Matrix from Dense Matrix";
		}
		
		virtual const SparseMat<mode, Dtype>& SparseDerived() const 
		{
			throw "Can not derive CSR Matrix from Dense Matrix";
		}
		
		size_t rows, cols, count;		
};

#endif