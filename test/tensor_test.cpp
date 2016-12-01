#include "gtest/gtest.h"
#include "dense_tensor.h"
#include <type_traits>

using namespace gnn;


TEST(TensorTest, Compile)
{
	int s = rand();
	Tensor* t = new DenseTensor<CPU, FLOAT32>();

	auto& tmp = t->Derived<CPU, DENSE, FLOAT32>();
	tmp.Zeros();

	EXPECT_EQ(1, 1);
}