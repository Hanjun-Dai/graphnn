#include "dense_matrix.h"
#include "mkl_helper.h"
#include "sparse_matrix.h"
#include <iostream>
#include <cstring>
#include <random>
#include <chrono>

template<typename Dtype>
DenseMat<CPU, Dtype>::~DenseMat()
{
	if (!is_submat)
		MatUtils<CPU>::DelArr(data);
}

template<typename Dtype>
DenseMat<CPU, Dtype>::DenseMat()
{
	mem_size = this->count = this->rows = this->cols = 0U;
	data = nullptr;
	is_submat = false;
}

template<typename Dtype>
DenseMat<CPU, Dtype>::DenseMat(size_t _rows, size_t _cols)
{
	this->rows = _rows; 
	this->cols = _cols;
	this->count = _rows * _cols;
	mem_size = this->count + (this->count & 1);
	MatUtils<CPU>::MallocArr(data, sizeof(Dtype) * mem_size);
	is_submat = false;
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Resize(size_t _newRows, size_t _newCols)
{
	this->rows = _newRows; 
	this->cols = _newCols;
	this->count = this->rows * this->cols;
	if (this->count > mem_size)
	{
		mem_size = this->count + (this->count & 1); 
		MatUtils<CPU>::DelArr(data);
		MatUtils<CPU>::MallocArr(data, sizeof(Dtype) * mem_size);
	}
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Serialize(FILE* fid) 
{
	IMatrix<CPU, Dtype>::Serialize(fid);
	assert(fwrite(&mem_size, sizeof(size_t), 1, fid) == 1);
	assert(fwrite(data, sizeof(Dtype), mem_size, fid) == mem_size);
	assert(fwrite(&is_submat, sizeof(bool), 1, fid) == 1);
	bias_mult.Serialize(fid);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Deserialize(FILE* fid) 
{
	IMatrix<CPU, Dtype>::Deserialize(fid);
	assert(fread(&mem_size, sizeof(size_t), 1, fid) == 1);
	MatUtils<CPU>::DelArr(data);
	MatUtils<CPU>::MallocArr(data, sizeof(Dtype) * mem_size);	
	assert(fread(data, sizeof(Dtype), mem_size, fid) == mem_size);
	assert(fread(&is_submat, sizeof(bool), 1, fid) == 1);
	bias_mult.Deserialize(fid);			
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::CopyFrom(DenseMat<CPU, Dtype>& src)
{
	Resize(src.rows, src.cols);
	memcpy(data, src.data, sizeof(Dtype) * this->count);
}
		
template<typename Dtype>
void DenseMat<CPU, Dtype>::CopyFrom(DenseMat<GPU, Dtype>& src)
{
	Resize(src.rows, src.cols);
	cudaMemcpyAsync(data, src.data, sizeof(Dtype) * this->count, cudaMemcpyDeviceToHost, GPUHandle::streams[src.streamid]);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::CopyFrom(SparseMat<CPU, Dtype>& src)
{
	Resize(src.rows, src.cols);
	memset(data, 0, sizeof(Dtype) * this->count);
	size_t idx;
	for (size_t i = 0; i < src.data->len_ptr - 1; ++i)
	{
		for (int k = src.data->ptr[i]; k < src.data->ptr[i + 1]; ++k)
		{
			idx = this->cols * i + src.data->col_idx[k];
			data[idx] = src.data->val[k];
		}
	}
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::CopyFrom(SparseMat<GPU, Dtype>& src)
{
	Resize(src.rows, src.cols);
	throw "not implemented";
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::SetRandU(Dtype lb, Dtype ub, size_t _rows, size_t _cols)
{
	if (_rows && _cols)
	{
		Resize(_rows, _cols);
    }
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
    std::uniform_real_distribution<Dtype> distribution(lb, ub);
    for (int i = 0; i < this->count; ++i)
		data[i] = distribution(generator);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::SetRandN(Dtype mean, Dtype std, size_t _rows, size_t _cols)
{
	if (_rows && _cols)
	{
		Resize(_rows, _cols);
	}	
	auto seed = std::chrono::system_clock::now().time_since_epoch().count();    
	std::default_random_engine generator(seed);
	std::normal_distribution<Dtype> distribution(mean, std);
	for (int i = 0; i < this->count; ++i)
		data[i] = distribution(generator);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::SetRandSign(size_t _rows, size_t _cols)
{
	if (_rows && _cols)
	{
		Resize(_rows, _cols);
	}
	for (int i = 0; i < this->count; ++i)
		data[i] = rand() % 2 == 0 ? 1 : -1;
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::SetRandChi2(Dtype degree, size_t _rows, size_t _cols)
{
	if (_rows && _cols)
	{
		Resize(_rows, _cols);
	}	
	auto seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::chi_squared_distribution<Dtype> distribution(degree);
	for (int i = 0; i < this->count; ++i)
		data[i] = distribution(generator);	
}

template<typename Dtype>			
void DenseMat<CPU, Dtype>::Identity(size_t dim)
{
    if (dim)
        Resize(dim, dim);
    assert(this->rows == this->cols);
    Fill(0.0);
    for (size_t i = 0; i < this->rows; ++i)
        this->data[i * this->cols + i] = 1.0;
}

template<typename Dtype>			
void DenseMat<CPU, Dtype>::Zeros(size_t _rows, size_t _cols)
{
	Resize(_rows, _cols);	
	memset(data, 0, sizeof(Dtype) * this->count);
}

template<typename Dtype>			
void DenseMat<CPU, Dtype>::Zeros()
{
	if (this->count)
	   memset(data, 0, sizeof(Dtype) * this->count);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Add(Dtype scalar)
{
    for (int i = 0; i < this->count; ++i)
		data[i] += scalar;
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Fill(Dtype scalar)
{
	for (int i = 0; i < this->count; ++i)
		data[i] = scalar;
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Scale(Dtype scalar)
{
	if (scalar == 0)
	{
		memset(data, 0, sizeof(Dtype) * this->count);
		return;
	}
	
	if (scalar != 1)
	{
		for (int i = 0; i < this->count; ++i)
			data[i] *= scalar;
	}
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Power(Dtype scalar)
{
	MKLHelper_PowerX(this->count, this->data, scalar, this->data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Softmax()
{
    	Dtype sum, max_v;
		size_t i, j;
        Dtype* data_ptr;
        for (i = 0, data_ptr = this->data; i < this->rows; ++i, data_ptr += this->cols)
        {
            max_v = data_ptr[0];
            for (j = 1; j < this->cols; ++j)
                if (data_ptr[j] > max_v)
                    max_v = data_ptr[j];
            for (j = 0; j < this->cols; ++j)    
                data_ptr[j] -= max_v;
        }
        
        MKLHelper_Exp(this->count, this->data, this->data);
        for (i = 0, data_ptr = this->data; i < this->rows; ++i, data_ptr += this->cols)
        {
            sum = MKLHelper_Asum(this->cols, data_ptr);
            for (j = 0; j < this->cols; ++j)
                data_ptr[j] /= sum;
        }
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Log(DenseMat<CPU, Dtype>& src)
{
    Resize(src.rows, src.cols);
    MKLHelper_Log(this->count, src.data, this->data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Log()
{
    MKLHelper_Log(this->count, this->data, this->data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Exp(DenseMat<CPU, Dtype>& src)
{
    Resize(src.rows, src.cols);
    MKLHelper_Exp(this->count, src.data, this->data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Exp()
{
    MKLHelper_Exp(this->count, this->data, this->data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Sin(DenseMat<CPU, Dtype>& src)
{
    Resize(src.rows, src.cols);
    MKLHelper_Sin(this->count, src.data, this->data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Sin()
{
    MKLHelper_Sin(this->count, this->data, this->data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Cos(DenseMat<CPU, Dtype>& src)
{
    Resize(src.rows, src.cols);
    MKLHelper_Cos(this->count, src.data, this->data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Cos()
{
    MKLHelper_Cos(this->count, this->data, this->data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Sqrt()
{
	MKLHelper_Sqrt(this->count, data, data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Inv()
{
    MKLHelper_Inv(this->count, data, data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::InvSqrt()
{
    MKLHelper_InvSqrt(this->count, data, data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Square()
{
	MKLHelper_Square(this->count, data, data);
}

template<typename Dtype>
Dtype DenseMat<CPU, Dtype>::Norm2()
{
	return MKLHelper_Norm2(this->count, data); 
}

template<typename Dtype>
Dtype DenseMat<CPU, Dtype>::Asum()
{
	return MKLHelper_Asum(this->count, data);
}

template<typename Dtype>
Dtype DenseMat<CPU, Dtype>::Sum()
{
    Dtype sum = 0;
    for (size_t i = 0; i < this->count; ++i)
        sum += this->data[i];
    return sum;
}

template<typename Dtype>
Dtype DenseMat<CPU, Dtype>::Dot(DenseMat<CPU, Dtype>& rhs)
{
    assert(this->count == rhs.count);
    
    return MKLHelper_Dot(this->count, this->data, rhs.data);
}

template<typename Dtype>
Dtype DenseMat<CPU, Dtype>::AsScalar()
{
    assert(this->rows == this->cols && this->cols == 1);
    return this->data[0];
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Clip(Dtype max_abs)
{
    assert(max_abs >= 0);
    for (auto i = 0; i < this->count; ++i)
    {
        if (this->data[i] < -max_abs)
            this->data[i] = -max_abs;
        if (this->data[i] > max_abs)
            this->data[i] = max_abs;
    }
}

template<typename Dtype>
Dtype DenseMat<CPU, Dtype>::Amax()
{
	return fabs(this->data[MKLHelper_Amax(this->count, data)]);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::ReduceByRow(DenseMat<CPU, Dtype>& src, Dtype scalar)
{
    Resize(1, src.cols);
    if (bias_mult.count < src.rows)
		bias_mult.Resize(src.rows);
	bias_mult.Fill(scalar);
    
    MKLHelper_GeMV(CblasRowMajor, CblasTrans, src.rows, src.cols, 1.0, src.data, src.cols, 
                   bias_mult.data, 1, 0.0, this->data, 1);    
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Mean(DenseMat<CPU, Dtype>& src)
{
    ReduceByRow(src, 1.0 / src.rows);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::RowSum(DenseMat<CPU, Dtype>& src)
{
    ReduceByRow(src, 1.0);
}


template<typename Dtype>
void DenseMat<CPU, Dtype>::AddRowVec(DenseMat<CPU, Dtype>& x, Dtype alpha)
{	
	assert(x.count == this->cols);
	
	if (bias_mult.count < this->rows)
	{
		bias_mult.Resize(this->rows);
		bias_mult.Fill(1.0);
	}		
	
	MKLHelper_Ger(CblasRowMajor, this->rows, this->cols, alpha, bias_mult.data, x.data, data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::AddColVec(DenseMat<CPU, Dtype>& x, Dtype alpha)
{
	assert(x.count == this->rows);
	
	if (bias_mult.count < this->cols)
	{
		bias_mult.Resize(this->cols);
		bias_mult.Fill(1.0);
	}	
	
	MKLHelper_Ger(CblasRowMajor, this->rows, this->cols, alpha, x.data, bias_mult.data, data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::SubmatAdd(size_t row_start, size_t col_start, DenseMat<CPU, Dtype>& src, Dtype beta)
{	
	if (src.rows == this->rows && src.cols == this->cols && row_start == 0 && col_start == 0)
	{
		Axpby(1.0, src, beta);
		return;
	}
	assert(row_start + src.rows <= this->rows);
	assert(col_start + src.cols <= this->cols);
	Scale(beta);
	
	for (size_t row = 0; row < src.rows; ++row)
	{
		MKLHelper_Axpy(src.cols, 1.0, src.data + row * src.cols, this->data + (row_start + row) * this->cols + col_start);
	}
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::SubmatAdd(size_t row_start, size_t col_start, SparseMat<CPU, Dtype>& src, Dtype beta)
{
	assert(row_start + src.rows <= this->rows);
	assert(col_start + src.cols <= this->cols);
	Scale(beta);

	size_t j, idx;
	for (size_t i = 0; i < src.data->len_ptr - 1; ++i)
	{
		idx = this->cols * (row_start + i) + col_start;
		for (int k = src.data->ptr[i]; k < src.data->ptr[i + 1]; ++k)
		{
			j = src.data->col_idx[k];
			data[idx + j] += src.data->val[k]; 
		}
	}
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::ScatterCols(std::vector< DenseMat<CPU, Dtype>* >& dst_list)
{
    assert(dst_list.size() > 0);
    assert(this->cols % dst_list.size() == 0);    
    for (size_t i = 0; i < dst_list.size(); ++i)
        dst_list[i]->Resize(this->rows, this->cols / dst_list.size());
    
    size_t src_offset = 0, dst_cols = dst_list[0]->cols;
    for (size_t row = 0; row < this->rows; ++row)
    {
        for (size_t p = 0; p < dst_list.size(); ++p)
        {
            memcpy(dst_list[p]->data + row * dst_cols, this->data + src_offset, sizeof(Dtype) * dst_cols);
            src_offset += dst_cols;
        }
    }
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::ConcatCols(DenseMat<CPU, Dtype>& src)
{
	assert(this->rows == src.rows);
	assert(this->cols % src.cols == 0);
	
	Dtype* src_data = src.data, *dst_data = this->data;
	for (size_t row = 0; row < this->rows; ++row)
	{		
		for (size_t j = 0; j < this->cols / src.cols; ++j)
		{			
			memcpy(dst_data, src_data, sizeof(Dtype) * src.cols);
			dst_data += src.cols;
		}
		src_data += src.cols;
	}
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::ConcatCols(std::vector< DenseMat<CPU, Dtype>* > src_list)
{
    assert(src_list.size() > 0);
    size_t new_rows = src_list[0]->rows, new_cols = src_list[0]->cols; 
    for (size_t i = 1; i < src_list.size(); ++i)
    {
        assert(src_list[i]->rows == new_rows);
        new_cols += src_list[i]->cols;
    }
    Resize(new_rows, new_cols);    
    size_t dst_offset = 0;
    for (size_t row = 0; row < this->rows; ++row)
    {
        for (size_t p = 0; p < src_list.size(); ++p)
        {
            size_t col_cnt = src_list[p]->cols;
            memcpy(this->data + dst_offset, src_list[p]->data + row * col_cnt, sizeof(Dtype) * col_cnt);
            dst_offset += col_cnt;
        }            
    }
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::EleWiseMul(SparseMat<CPU, Dtype>& src)
{
	assert(this->rows == src.rows && this->cols == src.cols);
    
    int st, ed;
    Dtype* pointer = this->data;
    for (size_t i = 0; i < this->rows; ++i)
    {
        st = src.data->ptr[i];
        ed = src.data->ptr[i + 1]; 
        
        for (size_t j = 0; j < this->cols; ++j)
        {
            if (st == ed || j != src.data->col_idx[st])
                pointer[j] = 0;
            else {
                pointer[j] *= src.data->val[st];
                st++;
            }
        }
        pointer += this->cols;
    }
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::EleWiseDiv(DenseMat<CPU, Dtype>& src)
{
	assert(this->rows == src.rows && this->cols == src.cols);
	MKLHelper_Div(this->count, this->data, src.data, this->data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::EleWiseDiv(DenseMat<CPU, Dtype>& lhs, DenseMat<CPU, Dtype>& rhs)
{
	assert(lhs.rows == rhs.rows && lhs.cols == rhs.cols);
    Resize(lhs.rows, lhs.cols);
	MKLHelper_Div(this->count, lhs.data, rhs.data, this->data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::EleWiseMul(DenseMat<CPU, Dtype>& src)
{
	assert(this->rows == src.rows && this->cols == src.cols);
	MKLHelper_Mul(this->count, src.data, this->data, this->data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::EleWiseMul(DenseMat<CPU, Dtype>& lhs, DenseMat<CPU, Dtype>& rhs)
{
	assert(lhs.rows == rhs.rows && lhs.cols == rhs.cols);
    Resize(lhs.rows, lhs.cols);
	MKLHelper_Mul(this->count, lhs.data, rhs.data, this->data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::AddSubmat(DenseMat<CPU, Dtype>& src, size_t row_start, size_t col_start, Dtype beta)
{
	if (src.rows == this->rows && src.cols == this->cols && row_start == 0 && col_start == 0)
	{
		Axpby(1.0, src, beta);
		return;
	}
	
	assert(row_start + this->rows <= src.rows);
	assert(col_start + this->cols <= src.cols);
	
	for (size_t row = 0; row < this->rows; ++row)
	{
		MKLHelper_Axpby(this->cols, 1.0, src.data + (row_start + row) * src.cols + col_start, beta, this->data + row * this->cols);
	}
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::GetColsFrom(DenseMat<CPU, Dtype>& src, size_t col_start, size_t col_cnt)
{
    assert(col_start + col_cnt <= src.cols);    
    this->Resize(src.rows, col_cnt);
    
    size_t offset = col_start;
    for (size_t i = 0; i < src.rows; ++i)
    {
        memcpy(this->data + i * col_cnt, src.data + offset, sizeof(Dtype) * col_cnt);
        offset += src.cols;
    }
} 

template<typename Dtype>
void DenseMat<CPU, Dtype>::ShuffleCols(DenseMat<CPU, Dtype>& src, const int* perm)
{
	Resize(src.rows, src.cols);
	
	for (size_t i = 0; i < this->rows; ++i)
	{
		Dtype* ptr = this->data + i * this->cols;
		Dtype* src_ptr = src.data + i * this->cols;
		for (size_t j = 0; j < this->cols; ++j)
			ptr[j] = src_ptr[perm[j]];
	}
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Repmat(DenseMat<CPU, Dtype>& src, size_t times_rows, size_t times_cols)
{
	assert(times_cols == 1);
	assert(times_rows >= 1);

	this->Resize(src.rows * times_rows, src.cols * times_cols);
	Dtype* cur_pos = this->data;
	for (size_t i = 0; i < times_rows; ++i)
	{
		memcpy(cur_pos, src.data, sizeof(Dtype) * src.count);
		cur_pos += src.count;
	}
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::ReduceCols(DenseMat<CPU, Dtype>& src)
{
	assert(src.cols % this->cols == 0);
	size_t num_parts = src.cols / this->cols;
	
	if (bias_mult.count < num_parts)
	{
		bias_mult.Resize(num_parts);
		bias_mult.Fill(1.0);
	}
	
	size_t offset = 0;
	for (size_t i = 0; i < this->rows; ++i)
	{
		MKLHelper_GeMM(CblasRowMajor, CPU_T(Trans::N), CPU_T(Trans::N), 
					1, this->cols, num_parts, 1.0, 
					bias_mult.data, num_parts,
					src.data + offset, this->cols, 
					0.0, data + i * this->cols, this->cols);
		offset += src.cols;
	}
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::MulRowVec(DenseMat<CPU, Dtype>& src, DenseMat<CPU, Dtype>& x, Dtype beta)
{
	Resize(src.rows, src.cols);
    assert(beta == 0); 	
	assert(x.count == this->cols);
	
	size_t offset = 0; 
	for (size_t row_idx = 0; row_idx < this->rows; ++row_idx)
	{	
		MKLHelper_Mul(x.count, x.data, src.data + offset, this->data + offset);
		offset += this->cols;
	}
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::MulColVec(DenseMat<CPU, Dtype>& src, DenseMat<CPU, Dtype>& x)
{
	Resize(src.rows, src.cols);
	assert(x.count == this->rows);
	size_t offset = 0; 
	for (size_t row_idx = 0; row_idx < this->rows; ++row_idx)
	{	
		for (size_t col_idx = 0; col_idx < this->cols; ++col_idx)
			this->data[offset + col_idx] = src.data[offset + col_idx] * x.data[row_idx];
		offset += this->cols;
	}
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::MulRowVec(DenseMat<CPU, Dtype>& x)
{
    this->MulRowVec(*this, x);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::GeaM(Dtype alpha, Trans transa, DenseMat<CPU, Dtype>& A, Dtype beta, Trans transb, DenseMat<CPU, Dtype>& B)
{
	if (&A == this || &B == this)
	{
		assert(transa == Trans::N && transb == Trans::N);
		Resize(A.rows, A.cols);
		if (&A == this)
			Axpby(beta, B, alpha);
		else
			Axpby(alpha, A, beta);
	} else {
		if (transa == Trans::N)
			Resize(A.rows, A.cols);
		else 
			Resize(A.cols, A.rows);
		MKLHelper_Omatadd('R', CPU_CharT(transa), CPU_CharT(transb),
						this->rows, this->cols,
						alpha, A.data, A.cols, 
						beta, B.data, B.cols, 
						data, this->cols);
	}
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Axpby(Dtype a, DenseMat<CPU, Dtype>& x, Dtype b)
{
    assert(x.rows == this->rows && x.cols == this->cols);
	MKLHelper_Axpby(this->count, a, x.data, b, data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Axpy(Dtype a, DenseMat<CPU, Dtype>& x)
{
    assert(x.rows == this->rows && x.cols == this->cols);
	MKLHelper_Axpy(this->count, a, x.data, data);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Axpy(Dtype a, SparseMat<CPU, Dtype>& x)
{
	assert(x.rows == this->rows && x.cols == this->cols);
    for (size_t i = 0; i < x.rows; ++i)
	{
		for (int k = x.data->ptr[i]; k < x.data->ptr[i + 1]; ++k)
		{
			this->data[x.cols * i + x.data->col_idx[k]] += a * x.data->val[k];
		}
	}
}

template<typename Dtype>
size_t DenseMat<CPU, Dtype>::GetRowMaxIdx(size_t row_idx)
{
    assert(row_idx < this->rows);
    size_t result = 0;
    Dtype* cur_ptr = this->data + row_idx * this->cols;
    Dtype cur_best = cur_ptr[0];
    for (size_t j = 1; j < this->cols; ++j)
        if (cur_ptr[j] > cur_best)
        {
            cur_best = cur_ptr[j];
            result = j;
        }
    return result;        
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::GeMM(DenseMat<CPU, Dtype>& A, DenseMat<CPU, Dtype>& B, Trans transa, Trans transb, Dtype alpha, Dtype beta)
{
	size_t m, n, k;
	GetDims(A.rows, A.cols, transa, B.rows, B.cols, transb, m, n, k);
	Resize(m, n);
		
	MKLHelper_GeMM(CblasRowMajor, CPU_T(transa), CPU_T(transb), 
					m, n, k, alpha, 
					A.data, A.cols, 
					B.data, B.cols, 
					beta, data, this->cols);
}


template<typename Dtype>
void DenseMat<CPU, Dtype>::SparseMM(SparseMat<CPU, Dtype>& A, DenseMat<CPU, Dtype>& B, Trans transa, Trans transb, Dtype alpha, Dtype beta)
{
	assert(transb == Trans::N);
	size_t m, n, k;
	GetDims(A.rows, A.cols, transa, B.rows, B.cols, transb, m, n, k);
	Resize(m, n);
	MKLHelper_CSRMM(CPU_CharT(transa), A.rows, this->cols, A.cols, alpha,
				(char*)"GLNC", A.data->val, A.data->col_idx, A.data->ptr, A.data->ptr + 1,
				B.data, B.cols, 
				beta, data, this->cols);
}

template<typename Dtype>
void DenseMat<CPU, Dtype>::Print2Screen() //debug only
{
	std::cerr << "========mat content========" << std::endl; 			
	for (size_t i = 0; i < this->rows; ++i)
	{
		for (size_t j = 0; j < this->cols; ++j)
	       std::cerr << data[i * this->cols + j] << " ";
		std::cerr << std::endl;
	}
	std::cerr << "========    end    ========" << std::endl;
}

template class DenseMat<CPU, double>;
template class DenseMat<CPU, float>;