#include "gtest/gtest.h"
#include "tensor/dense_tensor.h"
#include "tensor/t_data.h"
#include "tensor/gpu_handle.h"
#include <type_traits>

using namespace gnn;

TEST(CudaTest, Init)
{
	uint n_thread = 2;
	GpuHandle::Init(0, n_thread);

	std::vector< GpuContext > c;
	for (uint i = 0; i < n_thread; ++i)
		c.push_back(GpuHandle::AquireCtx());

	for (uint i = 0; i < n_thread; ++i)
		GpuHandle::ReleaseCtx(c[i]);
	for (uint i = 0; i < n_thread; ++i)
		GpuHandle::AquireCtx();
	GpuHandle::Destroy();
}
