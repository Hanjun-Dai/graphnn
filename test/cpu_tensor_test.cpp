#include "gtest/gtest.h"
#include "tensor/tensor_all.h"
#include "tensor/mkl_helper.h"
#include <type_traits>

using namespace gnn;

#define sqr(x) ((x) * (x))

TEST(CPUTensorTest, ReshapeSize)
{
	Tensor* t = new DTensor<CPU, float>();
	t->Reshape({2, 3, 4});

	auto& mat = t->Derived<CPU, DENSE, float>();

	ASSERT_EQ(2 * 3 * 4, mat.data->mem_size);
	delete t;
}

TEST(CPUTensorTest, norm2)
{
	Tensor* t = new DTensor<CPU, float>();
	t->Reshape({3, 3});
	auto& mat = t->Derived<CPU, DENSE, float>();
	mat.Fill(1.0);
	auto nn = MKL_Norm2(mat.shape.Count(), mat.data->ptr);
	std::cerr << nn << std::endl;
	delete t;
}

TEST(CPUTensorTest, Concat)
{
	DTensor<CPU, float> x, y, z;
	x.Reshape({5, 3});
	y.Reshape({5, 2});
	x.Fill(1.0);
	y.Fill(2.0);
	z.ConcatCols({&x, &y});

	for (int i = 0; i < (int)z.rows(); ++i)
		for (int j = 0; j < (int)z.cols(); ++j)
			if (j < 3)
				ASSERT_EQ(z.data->ptr[i * z.cols() + j], 1.0);
			else
				ASSERT_EQ(z.data->ptr[i * z.cols() + j], 2.0);
}

TEST(CPUTensorTest, BroadcastMulCol)
{
	DTensor<CPU, float> x, y;
	x.Reshape({5, 3}); 
	y.Reshape({5, 1});
	x.Fill(1.0);
	for (int i = 0; i < 5; ++i)
	{
		x.data->ptr[i * 3 + 1] = 2.0;
		x.data->ptr[i * 3 + 2] = 3.0;
	}
	for (int i = 0; i < 5; ++i)
		y.data->ptr[i] = i + 1;

	x.ElewiseMul(y);
	for (int i = 0; i < (int)x.rows(); ++i)
		for (int j = 0; j < (int)x.cols(); ++j)
			ASSERT_EQ(x.data->ptr[i * x.cols() + j], (i + 1) * (j + 1));
}

TEST(CPUTensorTest, BroadcastMulRow)
{
	DTensor<CPU, float> x, y;
	x.Reshape({3, 5}); 
	y.Reshape({1, 5});
	x.Fill(1.0);
	for (int i = 0; i < 5; ++i)
	{
		x.data->ptr[5 + i] = 2.0;
		x.data->ptr[10 + i] = 3.0;
	}

	for (int i = 0; i < 5; ++i)
		y.data->ptr[i] = i + 1;
	x.ElewiseMul(y);
	for (int i = 0; i < (int)x.rows(); ++i)
		for (int j = 0; j < (int)x.cols(); ++j)
			ASSERT_EQ(x.data->ptr[i * x.cols() + j], (i + 1) * (j + 1));
}

TEST(CPUTensorTest, RandUniform)
{
	Tensor* t = new DTensor<CPU, double>();
	t->Reshape({100, 100, 100});	
	auto& mat = t->Derived<CPU, DENSE, double>();

	mat.SetRandU(-1.0, 3.0);

	double s = 0.0;
	for (size_t i = 0; i < mat.data->mem_size; ++i)
		s += mat.data->ptr[i];
	s /= mat.shape.Count();
	double err = fabs(s - 1.0);
	EXPECT_LE(err, 1e-3);
	delete t;
}

TEST(CPUTensorTest, RandNorm)
{
	Tensor* t = new DTensor<CPU, double>();
	t->Reshape({100, 500, 100});	
	auto& mat = t->Derived<CPU, DENSE, double>();

	mat.SetRandN(5.0, 0.1);

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
	delete t;
}

TEST(CPUTensorTest, Zero)
{
	Tensor* t = new DTensor<CPU, float>();
	t->Reshape({2, 3, 4});
	auto& mat = t->Derived<CPU, DENSE, float>();
	mat.Zeros();

	int ans = mat.ASum();
	ASSERT_EQ(0, ans);
	delete t;
}

TEST(CPUTensorTest, AsScalar)
{
	Tensor* t = new DTensor<CPU, float>();
	t->Reshape({1, 1, 1});
	auto& mat = t->Derived<CPU, DENSE, float>();
	mat.Zeros();

	ASSERT_EQ(0, mat.AsScalar());
	delete t;
}

TEST(CPUTensorTest, Fill)
{
	Tensor* t = new DTensor<CPU, int>();
	t->Reshape({2, 3, 4});
	auto& mat = t->Derived<CPU, DENSE, int>();
	mat.Fill(2);

	float ans = 0;
	for (size_t i = 0; i < mat.data->mem_size; ++i)
		ans += mat.data->ptr[i];

	ASSERT_EQ(48, ans);
	delete t;
}