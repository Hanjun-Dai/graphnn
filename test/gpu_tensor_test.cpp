#include "gtest/gtest.h"
#include "tensor/tensor_all.h"
#include <type_traits>

using namespace gnn;

#define sqr(x) ((x) * (x))

TEST(GPUTensorTest, ReshapeSize)
{
	Tensor* t = new DTensor<GPU, float>();
	t->Reshape({2, 3, 4});

	auto& mat = t->Derived<GPU, DENSE, float>();

	ASSERT_EQ(2 * 3 * 4, mat.data->mem_size);	
}

TEST(GPUTensorTest, Concat)
{
	DTensor<CPU, float> x, y, z, tmp;
	x.Reshape({5, 6});
	y.Reshape({5, 3});

	x.SetRandU(-1.0, 3.0);
	y.SetRandU(-1.0, 3.0);	
	z.ConcatCols({&x, &y});

	DTensor<GPU, float> gx, gy, gz;
	gx.CopyFrom(x);
	gy.CopyFrom(y);

	gz.ConcatCols({&gx, &gy});

	tmp.CopyFrom(gz);
	tmp.Axpy(-1.0, z);

	EXPECT_LE(tmp.ASum(), 1e-6);
}

TEST(GPUTensorTest, RowRef)
{
	DTensor<CPU, float> x, tmp;
	DTensor<GPU, float> y;
	x.Reshape({5, 3});
	x.SetRandU(-1.0, 3.0);
	y.CopyFrom(x);

	auto p = x.GetRowRef(1, 2);
	auto q = y.GetRowRef((size_t)1, (size_t)2);
	tmp.CopyFrom(q);

	tmp.Axpy(-1.0, p);
	EXPECT_LE(tmp.ASum(), 1e-6);
}

TEST(GPUTensorTest, RandUniform)
{
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
}

TEST(GPUTensorTest, RandNorm)
{
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
}

TEST(GPUTensorTest, Fill)
{
	DTensor<GPU, double> mat;
	mat.Reshape({100, 100, 100});
	mat.Fill(2.0);

	double ans = mat.ASum();

	ASSERT_EQ(2 * 100 * 100 * 100, ans);
}

TEST(GPUTensorTest, ArgMax)
{
	DTensor<CPU, double> d_cpu;
	DTensor<CPU, int> idx_cpu, buf;
	d_cpu.Reshape({10, 1023});
	d_cpu.SetRandN(0.0, 1.0);
	d_cpu.ArgMax(idx_cpu);

	DTensor<GPU, double> d_gpu;
	DTensor<GPU, int> idx_gpu;
	d_gpu.CopyFrom(d_cpu);
	d_gpu.ArgMax(idx_gpu);
	buf.CopyFrom(idx_gpu);

	for (size_t i = 0; i < idx_gpu.shape.Count(); ++i)
	{
		ASSERT_EQ(idx_cpu.data->ptr[i], buf.data->ptr[i]);
	}
}

TEST(GPUTensorTest, GeMM)
{
	DTensor<CPU, double> x, y, z, zz;
	x.Reshape({10, 20});
	y.Reshape({30, 20});

	x.SetRandN(0.0, 1.0);
	y.SetRandN(0.0, 1.0);
	z.MM(x, y, Trans::N, Trans::T, 1.0, 0.0);

	DTensor<GPU, double> gx, gy, gz;
	gx.CopyFrom(x);
	gy.CopyFrom(y);
	gz.MM(gx, gy, Trans::N, Trans::T, 1.0, 0.0);

	zz.CopyFrom(gz);
	auto a1 = z.ASum(), a2 = gz.ASum();
	EXPECT_LE(fabs(a1 - a2), 1e-4);
}

TEST(GPUTensorTest, Softmax)
{
	DTensor<CPU, double> x, y;
	DTensor<GPU, double> gx;
	x.Reshape({20, 200});
	x.SetRandN(0.0, 1.0);
	gx.CopyFrom(x);

	x.Softmax();
	gx.Softmax();
	y.CopyFrom(gx);

	x.Axpy(-1.0, y);

	EXPECT_LE(x.ASum(), 1e-4);
}

TEST(GPUTensorTest, Mean)
{
	DTensor<CPU, float> x, dst_x;
	DTensor<GPU, float> gx, dst_gx;
	x.Reshape({20, 200});
	x.SetRandU(1, 2);
	gx.CopyFrom(x);

	dst_x.Mean(x);
	dst_gx.Mean(gx);

	EXPECT_LE(fabs(dst_x.AsScalar() - dst_gx.AsScalar()), 1e-5);
}

TEST(GPUTensorTest, JaggedSoftmax)
{
	DTensor<CPU, float> x;
	DTensor<CPU, int> len;

	DTensor<GPU, float> gx;
	DTensor<GPU, int> glen;	

	x.Reshape({10, 1});
	x.SetRandU(1, 2);
	gx.CopyFrom(x);

	len.Reshape({3, 1});
	len.data->ptr[0] = 2;
	len.data->ptr[1] = 3;
	len.data->ptr[2] = 5;

	glen.CopyFrom(len);

	x.JaggedSoftmax(len);
	gx.JaggedSoftmax(glen);

	DTensor<CPU, float> y;
	y.CopyFrom(gx);
	y.Axpy(-1.0, x);

	EXPECT_LE(y.ASum(), 1e-5);
}

TEST(GPUTensorTest, ElewiseMul)
{
	DTensor<CPU, float> x, y, tmp;
	DTensor<GPU, float> gx, gy;
	x.Reshape({20, 200});
	x.SetRandU(1, 2);
	y.Reshape({20, 200});
	y.SetRandN(0, 2);		

	gx.CopyFrom(x);
	gy.CopyFrom(y);

	x.ElewiseMul(y);
	gx.ElewiseMul(gy);

	tmp.CopyFrom(gx);

	x.Axpy(-1.0, tmp);

	EXPECT_LE(x.ASum(), 1e-4);
}

TEST(GPUTensorTest, BroadcastMulCol)
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

	DTensor<GPU, float> gx, gy;
	gx.CopyFrom(x);
	gy.CopyFrom(y);

	gx.ElewiseMul(gy);
	x.Fill(0);
	x.CopyFrom(gx);
	for (int i = 0; i < (int)x.rows(); ++i)
		for (int j = 0; j < (int)x.cols(); ++j)
			ASSERT_EQ(x.data->ptr[i * x.cols() + j], (i + 1) * (j + 1));
}

TEST(GPUTensorTest, BroadcastMulRow)
{
	DTensor<CPU, float> x, y;
	x.Reshape({30, 50}); 
	y.Reshape({1, 50});

	x.SetRandN(0, 1);
	y.SetRandN(0, 1);

	DTensor<GPU, float> gx, gy;
	gx.CopyFrom(x);
	gy.CopyFrom(y);

	gx.ElewiseMul(gy);	
	x.ElewiseMul(y);

	DTensor<CPU, float> tx;
	tx.CopyFrom(gx);
	x.Axpy(-1, tx);

	EXPECT_LE(x.ASum(), 1e-4);
}

TEST(GPUTensorTest, InvSqrSqrtNorm2)
{
	DTensor<CPU, float> x, tmp;
	x.Reshape({10, 10}); 
	x.SetRandU(1, 3);
	DTensor<GPU, float> gx;
	gx.CopyFrom(x);

	x.Square();
	gx.Square();
	tmp.CopyFrom(gx);
	tmp.Axpy(-1, x);
	EXPECT_LE(tmp.ASum(), 1e-4);

	x.Sqrt();
	gx.Sqrt();
	tmp.CopyFrom(gx);
	tmp.Axpy(-1, x);
	EXPECT_LE(tmp.ASum(), 1e-4);

	x.Inv();
	gx.Inv();
	tmp.CopyFrom(gx);
	tmp.Axpy(-1, x);
	EXPECT_LE(tmp.ASum(), 1e-4);
	
	EXPECT_LE(fabs(x.Norm2() - gx.Norm2()), 1e-4);
}

TEST(GPUTensorTest, SparseMM)
{
	SpTensor<CPU, float> a;
	a.Reshape({2, 3});
	a.ResizeSp(3, 3);
	a.data->row_ptr[0] = 0; 
	a.data->row_ptr[1] = 1;
	a.data->row_ptr[2] = 3;
	a.data->val[0] = 2.0;
	a.data->val[1] = 1.0;
	a.data->val[2] = 3.0;
	a.data->col_idx[0] = 1;
	a.data->col_idx[1] = 0;
	a.data->col_idx[2] = 2;

	DTensor<CPU, float> b({2, 4});
	for (int i = 0; i < 8; ++i)
		b.data->ptr[i] = i;

	SpTensor<GPU, float> ga;
	ga.CopyFrom(a);
	DTensor<GPU, float> gb;
	gb.CopyFrom(b);

	DTensor<GPU, float> gc;
	gc.MM(ga, gb, Trans::T, Trans::N, 1.0, 0.0);

	DTensor<CPU, float> c;
	c.CopyFrom(gc);

	//c.MM(a, b, Trans::T, Trans::N, 1.0, 0.0);
	for (size_t i = 0; i < c.rows(); ++i)
		{
			for (size_t j = 0; j < c.cols(); ++j)
				std::cerr << c.data->ptr[i * c.cols() + j] << " ";
			std::cerr << std::endl;
		}
}