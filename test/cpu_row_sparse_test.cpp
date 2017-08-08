#include "gtest/gtest.h"
#include "tensor/tensor_all.h"
#include "tensor/mkl_helper.h"
#include "tensor/cpu_row_sparse_tensor.h"
#include <type_traits>
#include <algorithm>
#include <chrono>
#include "tbb/tbb.h"

using namespace gnn;

#define sqr(x) ((x) * (x))

TEST(CPURpwSpTensorTest, norm2)
{
	typedef double Dtype;
	RowSpTensor<CPU, Dtype> t;

	t.Reshape({5000000, 100});
	t.Full().SetRandU(0, 1);

	DTensor<CPU, int> idxes;
	idxes.Reshape({t.rows()});
	for (size_t i = 0; i < t.rows(); ++i)
		idxes.data->ptr[i] = i;

	std::random_shuffle(idxes.data->ptr, idxes.data->ptr + t.rows());

	size_t cur_pos = t.rows();
	for (size_t i = cur_pos; i < t.rows(); ++i)
	{
		size_t row_idx = idxes.data->ptr[i];
		memset(t.data->ptr + row_idx * t.cols(), 0, sizeof(Dtype) * t.cols());
	}

	// t.Full().Print2Screen();

	auto t_start = std::chrono::high_resolution_clock::now();
	auto norm1 = t.Full().Norm2();
	auto t_end = std::chrono::high_resolution_clock::now();
	auto totalTime1 = std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count();

	idxes.Reshape({cur_pos});
	t.InsertRowIdxes(idxes.shape.Count(), idxes.data->ptr);

	t_start = std::chrono::high_resolution_clock::now();
	t.is_full = false;
	auto norm2 = t.Norm2();

	t_end = std::chrono::high_resolution_clock::now();
	auto totalTime2 = std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count();

	auto err = fabs(norm1 - norm2);
	EXPECT_LE(err, 1e-4);
	EXPECT_LE(totalTime2, totalTime1);
}
