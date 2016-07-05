#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include "imatrix.h"
#include "sp_data.h"
#include <memory>

template<MatMode mode, typename Dtype>
class SparseMat : public IMatrix<mode, Dtype>
{
public:
	
};


template<typename Dtype>
class SparseMat<CPU, Dtype> : public IMatrix<CPU, Dtype>
{
public:
		SparseMat();
		~SparseMat();
        template<MatMode otherMode>
		SparseMat(SparseMat<otherMode, Dtype>& src)
        {
            CopyFrom(src);
        }
		SparseMat(size_t _rows, size_t cols);
		inline virtual MatType GetMatType() override
		{
			return SPARSE;
		}
		inline virtual SparseMat<CPU, Dtype>& SparseDerived() override
		{
			return *this;
		}
		
		inline virtual const SparseMat<CPU, Dtype>& SparseDerived() const override
		{
			return *this;
		}
        
		Dtype Asum();
        
        virtual void Print2Screen() override;
        
		virtual void Serialize(FILE* fid) override;
		virtual void Deserialize(FILE* fid) override;
		
		void Resize(size_t newRos, size_t newCols);		
		void ResizeSp(int newNNZ, int newNPtr); 
	
		void CopyFrom(SparseMat<CPU, Dtype>& src);
		void CopyFrom(SparseMat<GPU, Dtype>& src);
        
		std::shared_ptr< SpData<CPU, Dtype> > data;
};

template<typename Dtype>
class SparseMat<GPU, Dtype> : public IMatrix<GPU, Dtype>
{
public:
		SparseMat();
		~SparseMat();
		template<MatMode otherMode>
		SparseMat(SparseMat<otherMode, Dtype>& src)
        {
            CopyFrom(src);
        }
		SparseMat(size_t _rows, size_t _cols, unsigned _streamid = 0U);
		inline virtual MatType GetMatType() override
		{
			return SPARSE;
		}	
		inline virtual SparseMat<GPU, Dtype>& SparseDerived() override
		{
			return *this;
		}
		
		inline virtual const SparseMat<GPU, Dtype>& SparseDerived() const override
		{
			return *this;
		}
        
        virtual void Serialize(FILE* fid) override;
		virtual void Deserialize(FILE* fid) override;
		virtual void Print2Screen() override;
        
		void Resize(size_t newRos, size_t newCols);		
		void ResizeSp(int newNNZ, int newNPtr);
	
		Dtype Asum();
                		
		void CopyFrom(SparseMat<CPU, Dtype>& src);
		void CopyFrom(SparseMat<GPU, Dtype>& src);
				
		std::shared_ptr< SpData<GPU, Dtype> > data;
		unsigned int streamid;
		cusparseMatDescr_t descr;		
};
#endif