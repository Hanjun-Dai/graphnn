#include "gtest/gtest.h"
#include "dense_tensor.h"
#include <type_traits>

using namespace gnn;

void Print(Tensor& t)
{
	t.Print();
}

TEST(TensorTest, Compile)
{
	int s = rand();
	Tensor t;

	auto tmp = t.Get<CPU, DENSE, 1, FLOAT32>();
	Print(t);
	Print(tmp);
	EXPECT_EQ(1, 1);
}