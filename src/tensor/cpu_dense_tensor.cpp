#include "tensor/cpu_dense_tensor.h"
#include "tensor/gpu_dense_tensor.h"
#include "tensor/cpu_sparse_tensor.h"
#include "tensor/t_data.h"
#include "tensor/cpu_unary_functor.h"
#include "tensor/mkl_helper.h"
#include "util/mem_holder.h"
#include <cstring>
#include <cassert>
#include <functional>

namespace gnn 
{

template<typename Dtype>
TensorTemplate<CPU, DENSE, Dtype>::TensorTemplate() : Tensor(), data(nullptr)
{
}

template<typename Dtype>
TensorTemplate<CPU, DENSE, Dtype>::TensorTemplate(std::vector<size_t> l) : Tensor()
{
	Reshape(l);
}

template<typename Dtype>
TensorTemplate<CPU, DENSE, Dtype>::TensorTemplate(TShape s) : Tensor()
{
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
void TensorTemplate<CPU, DENSE, Dtype>::CopyFrom(DTensor<CPU, Dtype>& src)
{
	Reshape(src.shape.dims);
	memcpy(data->ptr, src.data->ptr, sizeof(Dtype) * shape.Count());
}

template<typename Dtype>
void TensorTemplate<CPU, DENSE, Dtype>::CopyFrom(DTensor<GPU, Dtype>& src)
{
	Reshape(src.shape.dims);
	cudaMemcpy(data->ptr, src.data->ptr, sizeof(Dtype) * this->shape.Count(), cudaMemcpyDeviceToHost);
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

template class TensorTemplate<CPU, DENSE, float>;
template class TensorTemplate<CPU, DENSE, double>;

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

void TensorTemplate<CPU, DENSE, int>::CopyFrom(DTensor<CPU, int>& src)
{
	Reshape(src.shape.dims);
	memcpy(data->ptr, src.data->ptr, sizeof(int) * shape.Count());
}

void TensorTemplate<CPU, DENSE, int>::CopyFrom(DTensor<GPU, int>& src)
{
	Reshape(src.shape.dims);
	cudaMemcpy(data->ptr, src.data->ptr, sizeof(int) * this->shape.Count(), cudaMemcpyDeviceToHost);
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

template class TensorTemplate<CPU, DENSE, int>;

}