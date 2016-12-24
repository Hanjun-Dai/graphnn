#include "gtest/gtest.h"
#include "tensor/tensor_all.h"
#include <type_traits>

using namespace gnn;

#define sqr(x) ((x) * (x))

TEST(GPUTensorTest, ReshapeSize)
{
	GpuHandle::Init(0, 1);
	Tensor* t = new DTensor<GPU, float>();
	t->Reshape({2, 3, 4});

	auto& mat = t->Derived<GPU, DENSE, float>();

	ASSERT_EQ(2 * 3 * 4, mat.data->mem_size);
	GpuHandle::Destroy();
}

TEST(GPUTensorTest, RandUniform)
{
	GpuHandle::Init(0, 1);
	Tensor* t = new DTensor<GPU, double>();
	t->Reshape({101, 101, 101});	
	auto& tmat = t->Derived<GPU, DENSE, double>();
	tmat.SetRandU(-1.0, 3.0);
	auto* tmp = new DTensor<CPU, double>();
	tmp->CopyFrom(tmat);
	auto& mat = tmp->Derived<CPU, DENSE, double>();	
	double s = 0.0;
	for (size_t i = 0; i < mat.data->mem_size; ++i)
		s += mat.data->ptr[i];
	s /= mat.shape.Count();
	double err = fabs(s - 1.0);
	EXPECT_LE(err, 1e-3);
	GpuHandle::Destroy();
}

TEST(GPUTensorTest, RandNorm)
{
	GpuHandle::Init(0, 1);
	Tensor* t = new DTensor<GPU, double>();
	t->Reshape({100, 500, 100});	
	auto& tmat = t->Derived<GPU, DENSE, double>();

	tmat.SetRandN(5.0, 0.1);
	auto* tmp = new DTensor<CPU, double>();
	tmp->CopyFrom(tmat);
	auto& mat = tmp->Derived<CPU, DENSE, double>();	

	double s = 0.0;
	for (size_t i = 0; i < mat.data->mem_size; ++i)
		s += mat.data->ptr[i];
	s /= mat.shape.Count();
	double err = fabs(s - 5.0);
	EXPECT_LE(err, 1e-4);

	double ss = 0.0;
	for (size_t i = 0; i < mat.data->mem_size; ++i)
		ss += sqr(mat.data->ptr[i] - s);
	ss = sqrt(ss / mat.shape.Count());
	err = fabs(ss - 0.1);
	EXPECT_LE(err, 1e-4);
	GpuHandle::Destroy();
}