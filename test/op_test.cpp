#include "gtest/gtest.h"
#include "tensor/tensor_all.h"
#include "nn/nn_all.h"
#include <type_traits>
#include <thread>
#include <chrono>

using namespace gnn;

TEST(OpTest, MovingNorm)
{	
	DTensor<CPU, float> cx;
	cx.Reshape({5, 3});
	for (int i = 0; i < 5; ++i)
		for (int j = 0; j < 3; ++j)
			cx.data->ptr[i * 3 + j] = (i * 3 + j) * 0.01;

	DTensor<GPU, float> x, mean, inv_std;
	x.CopyFrom(cx);

	mean.Reshape({1, 3});
	mean.Fill(0.0);

	inv_std.Reshape({1, 3});
	inv_std.Fill(0.0);

	auto op = std::make_shared<	MovingNorm<GPU, float> >("moving_norm_test", 1);

	auto v_x = std::make_shared< DTensorVar<GPU, float> >("vx");
	auto v_mean = std::make_shared< DTensorVar<GPU, float> >("vm");
	auto v_inv_std = std::make_shared< DTensorVar<GPU, float> >("vi");
	auto v_o = std::make_shared< DTensorVar<GPU, float> >("vo");

	v_x->value = x;
	v_mean->value = mean;
	v_inv_std->value = inv_std;

	std::vector< std::shared_ptr<Variable> > ops = {v_x, v_mean, v_inv_std};
	std::vector< std::shared_ptr<Variable> > outs = {v_o};
	op->Forward(ops, outs, Phase::TRAIN);

	// v_x->value.Print2Screen();
	// v_mean->value.Print2Screen();
	// v_inv_std->value.Print2Screen();

	// v_o->value.Print2Screen();

	op->Forward(ops, outs, Phase::TEST);
	// v_x->value.Print2Screen();
	// v_mean->value.Print2Screen();
	// v_inv_std->value.Print2Screen();

	// v_o->value.Print2Screen();	
}
