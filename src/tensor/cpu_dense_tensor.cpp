#include "tensor/cpu_dense_tensor.h"
#include "tensor/cpu_sparse_tensor.h"
#include "tensor/cpu_row_sparse_tensor.h"
#include "tensor/t_data.h"
#include "tensor/cpu_unary_functor.h"
#include "tensor/mkl_helper.h"
#include "util/mem_holder.h"
#include <cstring>
#include <cassert>
#include <functional>
#include "tbb/tbb.h"

#ifdef USE_GPU
#include "tensor/gpu_dense_tensor.h"
#endif

namespace gnn 
{

template<typename Dtype>
TensorTemplate<CPU, DENSE, Dtype>::TensorTemplate() : Tensor(), data(nullptr)
{
}

template<typename Dtype>
TensorTemplate<CPU, DENSE, Dtype>::TensorTemplate(std::vector<size_t> l, Dtype* _data) : Tensor()
{
	if (_data)
	{
		this->shape.Reshape(l);
		this->data = std::make_shared< DenseData<CPU, Dtype> >(_data, 0, this->shape.Count());
	}
	else
		Reshape(l);
}

template<typename Dtype>
TensorTemplate<CPU, DENSE, Dtype>::TensorTemplate(TShape s, Dtype* _data) : Tensor()
{
	if (_data)
	{
		this->shape.Reshape(s.dims);
		this->data = std::make_shared< DenseData<CPU, Dtype> >(_data, 0, this->shape.Count());
	}
	else
		Reshape(s.dims);
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Reshape(std::vector<size_t> l)
{
	this->shape.Reshape(l);

	if (this->data == nullptr)
		this->data = std::make_shared< DenseData<CPU, Dtype> >();

	this->data->Resize(this->shape.Count());
}

template<typename Dtype>
MatType TensorTemplate<CPU, DENSE, Dtype>::GetMatType()
{
	return MatType::dense;
}

template<typename Dtype>
MatMode TensorTemplate<CPU, DENSE, Dtype>::GetMatMode()
{
	return MatMode::cpu;
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Serialize(FILE* fid)
{
	Tensor::Serialize(fid);
	assert(fwrite(&(data->mem_size), sizeof(size_t), 1, fid) == 1);
	assert(fwrite(data->ptr, sizeof(Dtype), data->mem_size, fid) == data->mem_size);
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Deserialize(FILE* fid)
{
	Tensor::Deserialize(fid);	
	size_t new_mem_size;
	assert(fread(&(new_mem_size), sizeof(size_t), 1, fid) == 1);
	this->data->Resize(new_mem_size);
	assert(fread(data->ptr, sizeof(Dtype), new_mem_size, fid) == new_mem_size);	
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::CopyFrom(DTensor<CPU, Dtype>& src)
{
	Reshape(src.shape.dims);
	memcpy(data->ptr, src.data->ptr, sizeof(Dtype) * shape.Count());
}

template<typename Dtype>
DTensor<CPU, Dtype> TensorTemplate<CPU, DENSE, Dtype>::GetRowRef(size_t row_start, size_t row_cnt)
{
	DTensor<CPU, Dtype> result;
	size_t col;
	if ((int)shape.dims.size() > 1)
		col = shape.Count(1);
	else
		col = 1;
	result.data = std::make_shared< DenseData<CPU, Dtype> >( data->ptr, row_start * col, row_cnt * col);
	
	auto dims = this->shape.dims;
	dims[0] = row_cnt;
	result.shape.Reshape(dims);

	return result;	
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::ShallowCopy(DTensor<CPU, Dtype>& src)
{
	this->shape = src.shape;
	this->data = src.data;
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Zeros()
{
	if (shape.Count())
	   memset(this->data->ptr, 0, sizeof(Dtype) * shape.Count());
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::RowSelectiveZeros(DTensor<CPU, int>& row_idxes)
{
	size_t row_cnt = row_idxes.shape.Count();
	size_t dim = this->shape.Count(1);
	tbb::parallel_for(size_t(0), row_cnt, size_t(1), [&](size_t i){
		size_t row_idx = row_idxes.data->ptr[i];
		auto* row_ptr = data->ptr + row_idx * dim;
		memset(row_ptr, 0, sizeof(Dtype) * dim);
	});
}

template<typename Dtype>
Dtype TensorTemplate<CPU, DENSE, Dtype>::AsScalar()
{
	ASSERT(this->shape.Count() == 1, "can only convert trivial tensor to scalar");
	return this->data->ptr[0];	
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::SetRandN(Dtype mean, Dtype std)
{
	UnaryEngine<CPU>::Exec<UnaryRandNorm>(this->data->ptr, this->shape.Count(), mean, std);
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::SetRandU(Dtype lb, Dtype ub)
{
	UnaryEngine<CPU>::Exec<UnaryRandUniform>(this->data->ptr, this->shape.Count(), lb, ub);
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Fill(Dtype scalar)
{
	if (scalar == 0)
		this->Zeros();
	else {
		UnaryEngine<CPU>::Exec<UnarySet>(this->data->ptr, this->shape.Count(), scalar);
	}
}

template<typename Dtype>
Dtype TensorTemplate<CPU, DENSE, Dtype>::ASum()
{
	return MKL_ASum(this->shape.Count(), this->data->ptr);
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::ArgMax(DTensor<CPU, int>& dst, uint axis)
{
	ASSERT(axis == 0, "not supported for axis > 0 in CPU DENSE Tensor");
	dst.Reshape({this->shape[0]});

	Dtype* ptr = data->ptr;
	for (size_t i = 0; i < this->shape[0]; ++i)
	{
		dst.data->ptr[i] = 0;
		Dtype cur_max = *ptr;
		auto cnt = this->shape.Count(1);
		for (size_t j = 1; j < cnt; ++j)
			if (ptr[j] > cur_max)
			{
				cur_max = ptr[j];
				dst.data->ptr[i] = j;
			}
		ptr += cnt;
	}
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::MM(DTensor<CPU, Dtype>& a, DTensor<CPU, Dtype>& b, Trans transA, Trans transB, Dtype alpha, Dtype beta)
{
	ASSERT(a.rank() == 2 && b.rank() == 2, "only support mat x mat now");
	size_t m, n, k;
	GetDims(a.rows(), a.cols(), transA, b.rows(), b.cols(), transB, m, n, k);
	
	Reshape({m, n});
	MKL_GeMM(CblasRowMajor, CPU_T(transA), CPU_T(transB), 
			m, n, k, alpha, 
			a.data->ptr, a.cols(), 
			b.data->ptr, b.cols(), 
			beta, data->ptr, this->cols());
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::MM(SpTensor<CPU, Dtype>& a, DTensor<CPU, Dtype>& b, Trans transA, Trans transB, Dtype alpha, Dtype beta)
{
	assert(transB == Trans::N);
	size_t m, n, k;
	GetDims(a.rows(), a.cols(), transA, b.rows(), b.cols(), transB, m, n, k);

	Reshape({m, n});
	MKL_CSRMM(CPU_CharT(transA), a.rows(), this->cols(), a.cols(), alpha,
				(char*)"GLNC", a.data->val, a.data->col_idx, a.data->row_ptr, a.data->row_ptr + 1,
				b.data->ptr, b.cols(), 
				beta, data->ptr, this->cols());
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Softmax()
{
	ASSERT(this->rank() == 2, "Softmax is assumed to exec on matrix");
    Dtype sum, max_v;
	size_t i, j;
	Dtype* data_ptr;
    for (i = 0, data_ptr = this->data->ptr; i < this->rows(); ++i, data_ptr += this->cols())
    {
        max_v = data_ptr[0];
        for (j = 1; j < this->cols(); ++j)
            if (data_ptr[j] > max_v)
                max_v = data_ptr[j];
        for (j = 0; j < this->cols(); ++j)    
            data_ptr[j] -= max_v;
    }
    
    MKL_Exp(this->shape.Count(), this->data->ptr, this->data->ptr);
    for (i = 0, data_ptr = this->data->ptr; i < this->rows(); ++i, data_ptr += this->cols())
    {
        sum = MKL_ASum(this->cols(), data_ptr);
        for (j = 0; j < this->cols(); ++j)
            data_ptr[j] /= sum;
    }
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::JaggedSoftmax(DTensor<CPU, int>& lens)
{
	ASSERT(rows() == shape.Count(), "input must be a column vector");

	int total = 0;
	for (size_t i = 0; i < lens.shape.Count(); ++i)
	{
		auto cur_simplex = GetRowRef(total, lens.data->ptr[i]);
		cur_simplex.Reshape({(size_t)1, cur_simplex.shape.Count()});
		cur_simplex.Softmax();
		total += lens.data->ptr[i];
	}
	ASSERT( total == (int)shape.Count(), "length mismatch in jagged softmax");	
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Sum(DTensor<CPU, Dtype>& a, int axis)
{
	ASSERT(axis == -1, "currently only support global sum");
	Reshape({1});

	Dtype s = 0;
	auto cnt = a.shape.Count();
	for (size_t i = 0; i < cnt; ++i)
		s += a.data->ptr[i];
	data->ptr[0] = s;
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Mean(DTensor<CPU, Dtype>& a, int axis)
{
	ASSERT(axis == -1, "currently only support global mean");
	Reshape({1});

	Dtype s = 0;
	auto cnt = a.shape.Count();
	for (size_t i = 0; i < cnt; ++i)
		s += a.data->ptr[i];
	data->ptr[0] = s / cnt;
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Add(Dtype scalar)
{
	auto cnt = shape.Count();
	for (size_t i = 0; i < cnt; ++i)
		data->ptr[i] += scalar;
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Axpy(Dtype a, DTensor<CPU, Dtype>& x)
{
	ASSERT(this->shape == x.shape, "shape doesn't match in Axpy");
	MKL_Axpy(shape.Count(), a, x.data->ptr, data->ptr);
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::RowSelectiveAxpy(DTensor<CPU, int>& row_idxes, Dtype a, DTensor<CPU, Dtype>& x)
{
	ASSERT(this->shape == x.shape, "shape doesn't match in Axpy");
	ASSERT(row_idxes.shape.Count(), "wrong usage in row selective axpy");

	size_t dim = this->shape.Count(1);
	tbb::parallel_for(size_t(0), row_idxes.shape.Count(), size_t(1), [&](size_t i){
		size_t row_idx = row_idxes.data->ptr[i];

		MKL_Axpy(dim, a, x.data->ptr + row_idx * dim, data->ptr + row_idx * dim);
	});
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Axpy(Dtype a, SpTensor<CPU, Dtype>& x)
{
	ASSERT(this->shape == x.shape, "shape doesn't match in Axpy");
	for (size_t i = 0; i < x.rows(); ++i)
	{
		for (int k = x.data->row_ptr[i]; k < x.data->row_ptr[i + 1]; ++k)
			data->ptr[x.cols() * i + x.data->col_idx[k]] += a * x.data->val[k];
	}
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::RowSparseAxpby(Dtype a, RowSpTensor<CPU, Dtype>& x, Dtype b)
{
	if (x.is_full)
	{
		auto dtensor = x.Full();
		Axpby(a, dtensor, b);
	} else if (x.row_idxes.shape.Count())
	{
		size_t row_cnt = x.row_idxes.shape.Count();
		size_t dim = this->shape.Count(1);
		tbb::parallel_for(size_t(0), row_cnt, size_t(1), [&](size_t i){
			size_t row_idx = x.row_idxes.data->ptr[i];
			MKL_Axpby(dim, a, 
					x.data->ptr + row_idx * dim, 
					b,
					data->ptr + row_idx * dim);
		});
	}
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Axpby(Dtype a, DTensor<CPU, Dtype>& x, Dtype b)
{
	ASSERT(this->shape == x.shape, "shape doesn't match in Axpby");
	MKL_Axpby(this->shape.Count(), a, x.data->ptr, b, data->ptr);
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::ElewiseMul(SpTensor<CPU, Dtype>& src)
{
	ASSERT(this->shape == src.shape, "shape doesn't match in ElewiseMul");
    
    int st, ed;
    Dtype* pointer = this->data->ptr;
    for (size_t i = 0; i < this->rows(); ++i)
    {
        st = src.data->row_ptr[i];
        ed = src.data->row_ptr[i + 1]; 
        
        for (int j = 0; j < (int)this->cols(); ++j)
        {
            if (st == ed || j != src.data->col_idx[st])
                pointer[j] = 0;
            else {
                pointer[j] *= src.data->val[st];
                st++;
            }
        }
        pointer += this->cols();
    }
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::BCast(DTensor<CPU, Dtype>& src, std::function<void(Dtype&, Dtype&)> opr)
{
	ASSERT(this->rank() == src.rank(), "broadcasting only support same rank tensors; please do reshape manually");
	for (size_t i = 0; i < this->rank(); ++i)
		if (shape.dims[i] != src.shape.dims[i])
			ASSERT(src.shape.dims[i] == 1, "shape mismatch, broadcasting failed");
	
	int r = rank();
	size_t src_idx;
	std::vector<size_t> cur_pos(r), src_pos(r);
	for (auto& c : cur_pos)
		c = 0;
	auto ele_cnt = shape.Count();
	for (size_t i = 0; i < ele_cnt; ++i)
	{
		for (int i = 0; i < r; ++i)
			src_pos[i] = cur_pos[i] >= src.shape.dims[i] ? 0 : cur_pos[i];
		src_idx = src.shape.Coor2Idx(src_pos);
		opr(data->ptr[i], src.data->ptr[src_idx]);
		cur_pos[r - 1] += 1;
		for (int i = r - 1; i > 0; --i)
			if (cur_pos[i] >= shape.dims[i])
			{
				cur_pos[i] -= shape.dims[i];
				cur_pos[i - 1]++;
			}
	}
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::ConcatCols(std::vector< DTensor<CPU, Dtype>* > src_list)
{
    ASSERT(src_list.size(), "no operator for concat");    
    size_t new_rows = src_list[0]->rows(), new_cols = src_list[0]->cols(); 
    for (size_t i = 1; i < src_list.size(); ++i)
    {
        ASSERT(src_list[i]->rows() == new_rows, "should have same # rows");
        new_cols += src_list[i]->cols();
    }
    Reshape({new_rows, new_cols});

    size_t dst_offset = 0;
    for (size_t row = 0; row < this->rows(); ++row)
    {
        for (size_t p = 0; p < src_list.size(); ++p)
        {
            size_t col_cnt = src_list[p]->cols();
            memcpy(this->data->ptr + dst_offset, src_list[p]->data->ptr + row * col_cnt, sizeof(Dtype) * col_cnt);
            dst_offset += col_cnt;
        }
    }
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::CopyColsFrom(DTensor<CPU, Dtype>& src, size_t col_start, size_t col_cnt)
{
    ASSERT(col_start + col_cnt <= src.cols(), "cols out of range");   
    this->Reshape({src.rows(), col_cnt});
    
    size_t offset = col_start;
    for (size_t i = 0; i < src.rows(); ++i)
    {
        memcpy(this->data->ptr + i * col_cnt, src.data->ptr + offset, sizeof(Dtype) * col_cnt);
        offset += src.cols();
    }
} 

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::ElewiseMul(DTensor<CPU, Dtype>& src)
{
	if (this->shape == src.shape)
	{
		MKL_Mul(this->shape.Count(), src.data->ptr, this->data->ptr, this->data->ptr);
	} else { // require broadcasting
		BCast(src, [](Dtype& dst, Dtype& src){ 
			dst *= src; 
		});
	}
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::ElewiseDiv(DTensor<CPU, Dtype>& src)
{
	if (this->shape == src.shape)
	{
		MKL_Mul(this->shape.Count(), src.data->ptr, this->data->ptr, this->data->ptr);
	} else { // require broadcasting
		BCast(src, [](Dtype& dst, Dtype& src){ 
			dst /= src; 
		});
	}
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Scale(Dtype scalar)
{
	if (scalar == 0)
	{
		memset(data->ptr, 0, sizeof(Dtype) * this->shape.Count());
		return;
	}	
	if (scalar != 1)
	{
		auto cnt = shape.Count();
		for (size_t i = 0; i < cnt; ++i)
			data->ptr[i] *= scalar;
	}
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Abs()
{
	MKL_Abs(this->shape.Count(), data->ptr, data->ptr);
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Inv()
{
	MKL_Inv(this->shape.Count(), data->ptr, data->ptr);
}

template<typename Dtype>
Dtype TensorTemplate<CPU, DENSE, Dtype>::Norm2()
{
	return MKL_Norm2(this->shape.Count(), data->ptr); 
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Square()
{
	MKL_Square(this->shape.Count(), data->ptr, data->ptr);
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Sqrt()
{
	MKL_Sqrt(this->shape.Count(), data->ptr, data->ptr);
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::InvSqrt()
{
	MKL_InvSqrt(this->shape.Count(), data->ptr, data->ptr);
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Sigmoid()
{
	UnaryEngine<CPU>::Exec<UnarySigmoid>(this->data->ptr, this->shape.Count());
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Log()
{
	MKL_Log(this->shape.Count(), data->ptr, data->ptr);
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Exp()
{
	MKL_Exp(this->shape.Count(), data->ptr, data->ptr);
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Truncate(Dtype lb, Dtype ub)
{
	ASSERT(lb <= ub, "the interval is invalid");
	UnaryEngine<CPU>::Exec<UnaryTruncate>(this->data->ptr, this->shape.Count(), lb, ub);
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::Print2Screen()
{
	ASSERT(shape.dims.size() == 2, "can only print matrix");
	std::cerr << "========= " << rows() << " x " << cols() << " ==========" << std::endl;
	for (size_t i = 0; i < rows(); ++i)
	{
		for (size_t j = 0; j < cols(); ++j)
			std::cerr << data->ptr[i * cols() + j] << " ";
		std::cerr << std::endl;
	}
	std::cerr << std::endl;
}

///================================ int tensor ===================================

TensorTemplate<CPU, DENSE, int>::TensorTemplate() : data(nullptr)
{

}

void TensorTemplate<CPU, DENSE, int>::Reshape(std::vector<size_t> l)
{
	this->shape.Reshape(l);

	if (this->data == nullptr)
		this->data = std::make_shared< DenseData<CPU, int> >();

	this->data->Resize(this->shape.Count());
}

MatType TensorTemplate<CPU, DENSE, int>::GetMatType()
{
	return MatType::dense;
}

MatMode TensorTemplate<CPU, DENSE, int>::GetMatMode()
{
	return MatMode::cpu;
}

void TensorTemplate<CPU, DENSE, int>::Serialize(FILE* fid)
{
	Tensor::Serialize(fid);
	assert(fwrite(&(data->mem_size), sizeof(size_t), 1, fid) == 1);
	assert(fwrite(data->ptr, sizeof(int), data->mem_size, fid) == data->mem_size);
}

void TensorTemplate<CPU, DENSE, int>::Deserialize(FILE* fid)
{
	Tensor::Deserialize(fid);	
	size_t new_mem_size;
	assert(fread(&(new_mem_size), sizeof(size_t), 1, fid) == 1);
	this->data->Resize(new_mem_size);
	assert(fread(data->ptr, sizeof(int), new_mem_size, fid) == new_mem_size);
}

void TensorTemplate<CPU, DENSE, int>::CopyFrom(DTensor<CPU, int>& src)
{
	Reshape(src.shape.dims);
	memcpy(data->ptr, src.data->ptr, sizeof(int) * shape.Count());
}

void TensorTemplate<CPU, DENSE, int>::ShallowCopy(DTensor<CPU, int>& src)
{
	this->shape = src.shape;
	this->data = src.data;
}

void TensorTemplate<CPU, DENSE, int>::Zeros()
{
	if (this->data->mem_size)
	   memset(this->data->ptr, 0, sizeof(int) * this->data->mem_size);
}

void TensorTemplate<CPU, DENSE, int>::Fill(int scalar)
{
	if (scalar == 0)
		this->Zeros();
	else {
		UnaryEngine<CPU>::Exec<UnarySet>(this->data->ptr, this->shape.Count(), scalar);
	}
}

int TensorTemplate<CPU, DENSE, int>::AsScalar()
{
	assert(this->shape.Count() == 1);
	return this->data->ptr[0];	
}

#ifdef USE_GPU
template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::CopyFrom(DTensor<GPU, Dtype>& src)
{
	Reshape(src.shape.dims);
	cudaMemcpy(data->ptr, src.data->ptr, sizeof(Dtype) * this->shape.Count(), cudaMemcpyDeviceToHost);
}

void TensorTemplate<CPU, DENSE, int>::CopyFrom(DTensor<GPU, int>& src)
{
	Reshape(src.shape.dims);
	cudaMemcpy(data->ptr, src.data->ptr, sizeof(int) * this->shape.Count(), cudaMemcpyDeviceToHost);
}
#endif

template class TensorTemplate<CPU, DENSE, float>;
template class TensorTemplate<CPU, DENSE, double>;

template class TensorTemplate<CPU, DENSE, int>;

}