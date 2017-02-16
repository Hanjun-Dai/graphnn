#include "gtest/gtest.h"
#include "tensor/gpu_handle.h"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  gnn::GpuHandle::Init(0, 1);

  return RUN_ALL_TESTS();
}