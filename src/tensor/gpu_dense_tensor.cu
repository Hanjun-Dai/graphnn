#include "tensor/gpu_dense_tensor.h"
#include "tensor/cpu_dense_tensor.h"
#include "tensor/t_data.h"
#include "tensor/cu_rand_kernel.h"
#include "tensor/cuda_helper.h"
#include "util/mem_holder.h"

namespace gnn
{
	
template<typename Dtype>
TensorTemplate<GPU, DENSE, Dtype>::TensorTemplate() : Tensor(), data(nullptr)
{
}

template<typename Dtype>
TensorTemplate<GPU, DENSE, Dtype>::TensorTemplate(std::vector<size_t> l) : Tensor()
{
	Reshape(l);
}

template<typename Dtype>
TensorTemplate<GPU, DENSE, Dtype>::TensorTemplate(TShape s) : Tensor()
{
	Reshape(s.dims);
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::Reshape(std::vector<size_t> l)
{
	this->shape.Reshape(l);

	if (this->data == nullptr)
		this->data = std::make_shared< DenseData<GPU, Dtype> >();

	this->data->Resize(this->shape.Count());
}

template<typename Dtype>
MatType TensorTemplate<GPU, DENSE, Dtype>::GetMatType()
{
	return MatType::dense;
}

template<typename Dtype>
MatMode TensorTemplate<GPU, DENSE, Dtype>::GetMatMode()
{
	return MatMode::gpu;
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::CopyFrom(DTensor<CPU, Dtype>& src)
{
	Reshape(src.shape.dims);
	cudaMemcpy(this->data->ptr, src.data->ptr, sizeof(Dtype) * shape.Count(), cudaMemcpyHostToDevice);
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::CopyFrom(DTensor<GPU, Dtype>& src)
{
	Reshape(src.shape.dims);
	cudaMemcpy(this->data->ptr, src.data->ptr, sizeof(Dtype) * shape.Count(), cudaMemcpyDeviceToDevice);
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::ShallowCopy(DTensor<GPU, Dtype>& src)
{
	this->shape = src.shape;
	this->data = src.data;
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::Zeros()
{
	if (this->data->mem_size)
		cudaMemset(data->ptr, 0, this->shape.Count() * sizeof(Dtype));
}

template<typename Dtype>
Dtype TensorTemplate<GPU, DENSE, Dtype>::AsScalar()
{
	ASSERT(this->shape.Count() == 1, "can only convert trivial tensor to scalar");
 	Dtype result;
 	cudaMemcpy(&result, this->data->ptr, sizeof(Dtype), cudaMemcpyDeviceToHost);
 	return result;
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::SetRandN(Dtype mean, Dtype std)
{
	SetRand(data->ptr, shape.Count(), NormalRandomizer<Dtype>(mean, std));
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::SetRandU(Dtype lb, Dtype ub)
{
	SetRand(data->ptr, shape.Count(), UniformRandomizer<Dtype>(lb, ub));
}

template<typename Dtype>
void TensorTemplate<GPU, DENSE, Dtype>::Fill(Dtype scalar)
{
	if (scalar == 0)
		this->Zeros();
	else {
		//UnaryEngine<GPU>::Exec<UnarySet>(this->data->ptr, this->shape.Count(), scalar);
	}
}

template<typename Dtype>
Dtype TensorTemplate<GPU, DENSE, Dtype>::ASum()
{
	Dtype result;
	WITH_GPUCTX(ctx, {
		result = Cuda_Asum(ctx.cublasHandle, shape.Count(), data->ptr);
	});
	return result;
}

template class TensorTemplate<GPU, DENSE, float>;
template class TensorTemplate<GPU, DENSE, double>;

///================================ int tensor ===================================

TensorTemplate<GPU, DENSE, int>::TensorTemplate() : data(nullptr)
{

}

void TensorTemplate<GPU, DENSE, int>::Reshape(std::vector<size_t> l)
{
	this->shape.Reshape(l);

	if (this->data == nullptr)
		this->data = std::make_shared< DenseData<GPU, int> >();

	if (this->shape.Count() > this->data->mem_size)
	{
		this->data->mem_size = this->shape.Count();
		MemHolder<GPU>::DelArr(this->data->ptr);
		MemHolder<GPU>::MallocArr(this->data->ptr, sizeof(int) * this->shape.Count());
	}
}

MatType TensorTemplate<GPU, DENSE, int>::GetMatType()
{
	return MatType::dense;
}

MatMode TensorTemplate<GPU, DENSE, int>::GetMatMode()
{
	return MatMode::gpu;
}

template class TensorTemplate<GPU, DENSE, int>;

}