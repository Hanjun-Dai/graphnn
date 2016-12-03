#include "gtest/gtest.h"
#include "dense_tensor.h"
#include "t_data.h"
#include <type_traits>

using namespace gnn;

TEST(TensorTest, ReshapeSize)
{
	Tensor* t = new DenseTensor<CPU, float>();
	t->Reshape({2, 3, 4});

	auto& t_data = t->data->Derived<CPU, DENSE, float>();

	ASSERT_EQ(2 * 3 * 4, t_data.mem_size);
}

TEST(TensorTest, ZeroInt)
{
	Tensor* t = new DenseTensor<CPU, int>();
	t->Reshape({2, 3, 4});
	t->Zeros();

	auto& t_data = t->data->Derived<CPU, DENSE, int>();
	int ans = 0;
	for (size_t i = 0; i < t_data.mem_size; ++i)
		ans += t_data.ptr[i];

	ASSERT_EQ(0, ans);
}

TEST(TensorTest, AsScalar)
{
	Tensor* t = new DenseTensor<CPU, int>();
	t->Reshape({1, 1, 1});
	t->Zeros();

	ASSERT_EQ(0, t->AsScalar<int>());
}

TEST(TensorTest, Compile)
{
	int s = rand();
	Tensor* t = new DenseTensor<CPU, float>();

	auto& tmp = t->Derived<CPU, DENSE, float>();
	tmp.Reshape({2, 2, 4});

	auto& t_data = tmp.data->Derived(&tmp);
	std::cerr << t_data.mem_size << std::endl;
	tmp.Zeros();

	EXPECT_EQ(1, 1);
}