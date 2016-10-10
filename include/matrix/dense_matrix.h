#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H

#include "imatrix.h"
#include "vector.h"
#include <cassert>
#include <vector>
#include <thrust/device_vector.h>
#define SOFTMAX_THREADS 128

template<MatMode mode, typename Dtype>
class SparseMat;

template<MatMode mode, typename Dtype>
class DenseMat : public IMatrix<mode, Dtype>
{
public:
};

template<typename Dtype>
class DenseMat<CPU, Dtype> : public IMatrix<CPU, Dtype>
{
public:
		virtual ~DenseMat() override;
		DenseMat();
        DenseMat(const DenseMat<CPU, Dtype>& that) = delete;        
        
		DenseMat(size_t _rows, size_t _cols);
		
		virtual void Serialize(FILE* fid) override;		
		virtual void Deserialize(FILE* fid) override;
		
		inline virtual MatType GetMatType() override
		{
			return DENSE;
		}		
		inline virtual DenseMat<CPU, Dtype>& DenseDerived() override
		{
			return *this;
		}
		inline virtual const DenseMat<CPU, Dtype>& DenseDerived() const override
		{
			return *this;
		}
		
		void Resize(size_t _newRows, size_t _newCols);	
		
		void CopyFrom(DenseMat<CPU, Dtype>& src);		
		void CopyFrom(DenseMat<GPU, Dtype>& src);
		
		void CopyFrom(SparseMat<CPU, Dtype>& src);
		void CopyFrom(SparseMat<GPU, Dtype>& src);
		
		template<MatMode mode>
		void CopyFrom(IMatrix<mode, Dtype>* src)
		{
			if (src->GetMatType() == DENSE)
			{
				CopyFrom(src->DenseDerived());
			} else {
				CopyFrom(src->SparseDerived());
			}
		}
        
        void SetRandU(Dtype lb, Dtype ub, size_t _rows = 0, size_t _cols = 0);
		void SetRandN(Dtype mean, Dtype std, size_t _rows = 0, size_t _cols = 0);
		void SetRandSign(size_t _rows = 0, size_t _cols = 0);
		void SetRandChi2(Dtype degree, size_t _rows = 0, size_t _cols = 0);
		
        void Identity(size_t dim = 0);        
		void Zeros(size_t _rows, size_t _cols);
        void Zeros();
		void Fill(Dtype scalar);
		void Scale(Dtype scalar);		
		void Power(Dtype scalar);        
        void Softmax();
		void Sqrt();
		void Square();
        void InvSqrt();
        void Inv();
        void Log();
        void Log(DenseMat<CPU, Dtype>& src);
        void Exp();
        void Exp(DenseMat<CPU, Dtype>& src);
        void Sin();
        void Sin(DenseMat<CPU, Dtype>& src);
        void Cos();
        void Cos(DenseMat<CPU, Dtype>& src);
        
        Dtype Dot(DenseMat<CPU, Dtype>& rhs);
        Dtype AsScalar();
		Dtype Norm2();		
		Dtype Asum();
		Dtype Amax();
        Dtype Sum();
        void Clip(Dtype max_abs);
        
        void Add(Dtype scalar);
		void AddRowVec(DenseMat<CPU, Dtype>& x, Dtype alpha);		
		void AddColVec(DenseMat<CPU, Dtype>& x, Dtype alpha);
        void Mean(DenseMat<CPU, Dtype>& src);
        void RowSum(DenseMat<CPU, Dtype>& src);
        void ReduceByRow(DenseMat<CPU, Dtype>& src, Dtype scalar);
        
        void GetColsFrom(DenseMat<CPU, Dtype>& src, size_t col_start, size_t col_cnt); 
        
		void MulRowVec(DenseMat<CPU, Dtype>& src, DenseMat<CPU, Dtype>& x, Dtype beta = 0);
        void MulRowVec(DenseMat<CPU, Dtype>& x);
		void MulColVec(DenseMat<CPU, Dtype>& src, DenseMat<CPU, Dtype>& x);

		void AddSubmat(DenseMat<CPU, Dtype>& src, size_t row_start, size_t col_start, Dtype beta);
		void SubmatAdd(size_t row_start, size_t col_start, DenseMat<CPU, Dtype>& src, Dtype beta);
		void SubmatAdd(size_t row_start, size_t col_start, SparseMat<CPU, Dtype>& src, Dtype beta);		
		void SubmatAdd(size_t row_start, size_t col_start, IMatrix<CPU, Dtype>* src, Dtype beta)
		{
			if (src->GetMatType() == DENSE)
			{
				SubmatAdd(row_start, col_start, src->DenseDerived(), beta);
			} else {
				SubmatAdd(row_start, col_start, src->SparseDerived(), beta);
			}
		}
		void Repmat(DenseMat<CPU, Dtype>& src, size_t times_rows, size_t times_cols);
		void ScatterCols(std::vector< DenseMat<CPU, Dtype>* >& dst_list);
		void ConcatCols(DenseMat<CPU, Dtype>& src);
        void ConcatCols(std::vector< DenseMat<CPU, Dtype>* > src_list);
        
        void EleWiseDiv(DenseMat<CPU, Dtype>& src);
        void EleWiseDiv(DenseMat<CPU, Dtype>& lhs, DenseMat<CPU, Dtype>& rhs);
        
		void EleWiseMul(DenseMat<CPU, Dtype>& src);
        void EleWiseMul(DenseMat<CPU, Dtype>& lhs, DenseMat<CPU, Dtype>& rhs);
        void EleWiseMul(SparseMat<CPU, Dtype>& src);
        
		void ShuffleCols(DenseMat<CPU, Dtype>& src, const int* perm);
		void ReduceCols(DenseMat<CPU, Dtype>& src);
		
		void GeaM(Dtype alpha, Trans transa, DenseMat<CPU, Dtype>& A, Dtype beta, Trans transb, DenseMat<CPU, Dtype>& B);
        void Axpy(Dtype alpha, DenseMat<CPU, Dtype>& x);
        void Axpy(Dtype alpha, SparseMat<CPU, Dtype>& x);
		void Axpby(Dtype a, DenseMat<CPU, Dtype>& x, Dtype b);
		size_t GetRowMaxIdx(size_t row_idx);
        
		void GeMM(DenseMat<CPU, Dtype>& A, DenseMat<CPU, Dtype>& B, Trans transa, Trans transb, Dtype alpha, Dtype beta);
				
		void SparseMM(SparseMat<CPU, Dtype>& A, DenseMat<CPU, Dtype>& B, Trans transa, Trans transb, Dtype alpha, Dtype beta);
		
		virtual void Print2Screen() override;
		
		Dtype* data;
		size_t mem_size;
private:
		bool is_submat;
		Vector<CPU, Dtype> bias_mult;
};

template<typename Dtype>
class DenseMat<GPU, Dtype> : public IMatrix<GPU, Dtype>
{
public:
		virtual ~DenseMat() override;
		DenseMat(unsigned int _streamid = 0U);        
        DenseMat(const DenseMat<GPU, Dtype>& that) = delete;
        
		DenseMat(size_t _rows, size_t _cols, unsigned int _streamid = 0U);
		
		inline virtual MatType GetMatType() override
		{
			return DENSE;
		}
		
		inline virtual DenseMat<GPU, Dtype>& DenseDerived() override
		{
			return *this;
		}
		inline virtual const DenseMat<GPU, Dtype>& DenseDerived() const override
		{
			return *this;
		}		
        virtual void Serialize(FILE* fid) override;		
		virtual void Deserialize(FILE* fid) override;
        
		void Resize(size_t _newRows, size_t _newCols);
		
		void CopyFrom(DenseMat<CPU, Dtype>& src);		
		void CopyFrom(DenseMat<GPU, Dtype>& src);
		
		void CopyFrom(SparseMat<CPU, Dtype>& src);
		void CopyFrom(SparseMat<GPU, Dtype>& src);
		
		template<MatMode mode>
		void CopyFrom(IMatrix<mode, Dtype>* src)
		{
			if (src->GetMatType() == DENSE)
			{
				CopyFrom(src->DenseDerived());
			} else {
				CopyFrom(src->SparseDerived());
			}
		}
        				
        void SetRandU(Dtype lb, Dtype ub, size_t _rows = 0, size_t _cols = 0);                      
		void SetRandN(Dtype mean, Dtype std, size_t _rows = 0, size_t _cols = 0);
		void SetRandSign(size_t _rows = 0, size_t _cols = 0);
		void SetRandChi2(Dtype degree, size_t _rows = 0, size_t _cols = 0);
        
		void Softmax();
        void Identity(size_t dim = 0);
		void Zeros(size_t _rows, size_t _cols);
        void Zeros();
		void Fill(Dtype scalar);
		void Scale(Dtype scalar);
		void Power(Dtype scalar);
		void Sqrt();
        void InvSqrt();
        void Inv();
		void Square();
        void Log();
        void Log(DenseMat<GPU, Dtype>& src);
        void Exp();
        void Exp(DenseMat<GPU, Dtype>& src);
        void Sin();
        void Sin(DenseMat<GPU, Dtype>& src);
        void Cos();
        void Cos(DenseMat<GPU, Dtype>& src);
        
        Dtype Dot(DenseMat<GPU, Dtype>& rhs);
        Dtype AsScalar();
		Dtype Norm2();
		Dtype Asum();
		Dtype Amax();
        Dtype Sum();
        void Clip(Dtype max_abs);
        
        void Add(Dtype scalar);
		void ShuffleCols(DenseMat<GPU, Dtype>& src, const int* perm);
		void ReduceCols(DenseMat<GPU, Dtype>& src);
		
		void AddRowVec(DenseMat<GPU, Dtype>& x, Dtype alpha);		
		void AddColVec(DenseMat<GPU, Dtype>& x, Dtype alpha);
		
		void AddSubmat(DenseMat<GPU, Dtype>& src, size_t row_start, size_t col_start, Dtype beta);
		void GetColsFrom(DenseMat<GPU, Dtype>& src, size_t col_start, size_t col_cnt); 
        void Repmat(DenseMat<GPU, Dtype>& src, size_t times_rows, size_t times_cols);
		void SubmatAdd(size_t row_start, size_t col_start, SparseMat<GPU, Dtype>& src, Dtype beta);
		void SubmatAdd(size_t row_start, size_t col_start, DenseMat<GPU, Dtype>& src, Dtype beta);
		void SubmatAdd(size_t row_start, size_t col_start, IMatrix<GPU, Dtype>* src, Dtype beta)
		{
			if (src->GetMatType() == DENSE)
			{
				SubmatAdd(row_start, col_start, src->DenseDerived(), beta);
			} else {
				SubmatAdd(row_start, col_start, src->SparseDerived(), beta);
			}
		}
        void ScatterCols(std::vector< DenseMat<GPU, Dtype>* >& dst_list);
		void ConcatCols(DenseMat<GPU, Dtype>& src);
        void ConcatCols(std::vector< DenseMat<GPU, Dtype>* > src_list);
        
        void EleWiseDiv(DenseMat<GPU, Dtype>& src);
        void EleWiseDiv(DenseMat<GPU, Dtype>& lhs, DenseMat<GPU, Dtype>& rhs);
                
		void EleWiseMul(DenseMat<GPU, Dtype>& src);
        void EleWiseMul(DenseMat<GPU, Dtype>& lhs, DenseMat<GPU, Dtype>& rhs);
        void EleWiseMul(SparseMat<GPU, Dtype>& src);
        void Mean(DenseMat<GPU, Dtype>& src);
        void RowSum(DenseMat<GPU, Dtype>& src);
        void ReduceByRow(DenseMat<GPU, Dtype>& src, Dtype scalar);
        
		void MulRowVec(DenseMat<GPU, Dtype>& src, DenseMat<GPU, Dtype>& x, Dtype beta = 0);
		void MulRowVec(DenseMat<GPU, Dtype>& x);
		void MulColVec(DenseMat<GPU, Dtype>& src, DenseMat<GPU, Dtype>& x);
		void GeaM(Dtype alpha, Trans transa, DenseMat<GPU, Dtype>& A, Dtype beta, Trans transb, DenseMat<GPU, Dtype>& B);
		void Axpy(Dtype alpha, DenseMat<GPU, Dtype>& x);
        void Axpy(Dtype alpha, SparseMat<GPU, Dtype>& x);        
		void Axpby(Dtype a, DenseMat<GPU, Dtype>& x, Dtype b);
        size_t GetRowMaxIdx(size_t row_idx);
        Dtype GetRowMax(size_t row_idx);
                        
		void GeMM(DenseMat<GPU, Dtype>& A, DenseMat<GPU, Dtype>& B, Trans transa, Trans transb, Dtype alpha, Dtype beta);
		
		void SparseMM(SparseMat<GPU, Dtype>& A, DenseMat<GPU, Dtype>& B, Trans transa, Trans transb, Dtype alpha, Dtype beta);
		
		virtual void Print2Screen() override;
        
		unsigned streamid;
		Dtype* data;
		size_t mem_size;
						
private:		
        void GetPointerBuf(std::vector< DenseMat<GPU, Dtype>* >& mat_list);
        thrust::device_vector<Dtype*> pointer_buf;
        thrust::device_ptr<Dtype> dev_ptr;
        bool is_submat;
		Vector<GPU, Dtype> bias_mult;
};

	
#endif