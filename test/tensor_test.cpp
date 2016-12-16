#include "gtest/gtest.h"
#include "tensor/dense_tensor.h"
#include "tensor/t_data.h"
#include <type_traits>

using namespace gnn;

TEST(TensorTest, ReshapeSize)
{
	Tensor* t = new DenseTensor<CPU, float>();
	t->Reshape({2, 3, 4});

	auto& mat = t->Derived<CPU, DENSE, float>();

	ASSERT_EQ(2 * 3 * 4, mat.data->mem_size);
}

TEST(TensorTest, Zero)
{
	Tensor* t = new DenseTensor<CPU, float>();
	t->Reshape({2, 3, 4});
	auto& mat = t->Derived<CPU, DENSE, float>();
	mat.Zeros();

	int ans = mat.ASum();
	ASSERT_EQ(0, ans);
}

TEST(TensorTest, AsScalar)
{
	Tensor* t = new DenseTensor<CPU, float>();
	t->Reshape({1, 1, 1});
	auto& mat = t->Derived<CPU, DENSE, float>();
	mat.Zeros();

	ASSERT_EQ(0, mat.AsScalar());
}

TEST(TensorTest, Fill)
{
	Tensor* t = new DenseTensor<CPU, int>();
	t->Reshape({2, 3, 4});
	auto& mat = t->Derived<CPU, DENSE, int>();
	mat.Fill(2);

	float ans = 0;
	for (size_t i = 0; i < mat.data->mem_size; ++i)
		ans += mat.data->ptr[i];

	ASSERT_EQ(48, ans);
}

TEST(TensorTest, Compile)
{
	EXPECT_EQ(1, 1);
}