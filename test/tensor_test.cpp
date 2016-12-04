#include "gtest/gtest.h"
#include "tensor/dense_tensor.h"
#include "tensor/t_data.h"
#include <type_traits>

using namespace gnn;

TEST(TensorTest, ReshapeSize)
{
	Tensor* t = new DenseTensor<CPU>();
	t->Reshape({2, 3, 4});

	auto& t_data = t->data->Derived<CPU, DENSE>();

	ASSERT_EQ(2 * 3 * 4, t_data.mem_size);
}

TEST(TensorTest, Zero)
{
	Tensor* t = new DenseTensor<CPU>();
	t->Reshape({2, 3, 4});
	t->Zeros();

	int ans = t->ASum();
	ASSERT_EQ(0, ans);
}

TEST(TensorTest, AsScalar)
{
	Tensor* t = new DenseTensor<CPU>();
	t->Reshape({1, 1, 1});
	t->Zeros();

	ASSERT_EQ(0, t->AsScalar());
}

TEST(TensorTest, Fill)
{
	Tensor* t = new DenseTensor<CPU>();
	t->Reshape({2, 3, 4});
	t->Fill(2.0);

	auto& t_data = t->data->Derived<CPU, DENSE>();
	Dtype ans = 0;
	for (size_t i = 0; i < t_data.mem_size; ++i)
		ans += t_data.ptr[i];

	ASSERT_EQ(48, ans);
}

TEST(TensorTest, Compile)
{
	int s = rand();
	Tensor* t = new DenseTensor<CPU>();

	auto& tmp = t->Derived<CPU, DENSE>();
	tmp.Reshape({2, 2, 4});

	auto& t_data = tmp.data->Derived(&tmp);
	tmp.Zeros();

	EXPECT_EQ(1, 1);
}