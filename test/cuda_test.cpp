// #include "gtest/gtest.h"
// #include "tensor/tensor_all.h"
// #include <type_traits>
// #include <thread>
// #include <chrono>

// using namespace gnn;

// TEST(CudaTest, Init)
// {
// 	uint n_thread = 2;
// 	GpuHandle::Init(0, n_thread);

// 	std::vector< GpuContext > c;
// 	for (uint i = 0; i < n_thread; ++i)
// 		c.push_back(GpuHandle::AquireCtx());

// 	for (uint i = 0; i < n_thread; ++i)
// 		GpuHandle::ReleaseCtx(c[i]);
// 	for (uint i = 0; i < n_thread; ++i)
// 		GpuHandle::AquireCtx();
// 	GpuHandle::Destroy();
// }

// TEST(CudaTest, With)
// {
// 	uint n_thread = 2;
// 	GpuHandle::Init(0, n_thread);

// 	for (uint i = 0; i < 10000; ++i)
// 	{
// 		WITH_GPUCTX(ctx, {
// 			assert(ctx.id == (int)i % (int)n_thread);
// 		});	
// 	}
// 	GpuHandle::Destroy();
// }

// TEST(CudaTest, Thread)
// {
// 	uint n_thread = 4;
// 	GpuHandle::Init(0, n_thread);

// 	std::vector< std::thread > pool;
// 	for (uint i = 0; i < n_thread; ++i)
// 	{
// 		pool.push_back(std::thread([](){
// 			WITH_GPUCTX(ctx, {
// 				std::this_thread::sleep_for(std::chrono::seconds(1));	
// 			});		
// 		}));
// 	}
// 	for (uint i = 0; i < n_thread; ++i)
// 		pool[i].join();
// 	GpuHandle::Destroy();
// }