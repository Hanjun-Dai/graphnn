#ifndef RELU_H
#define RELU_H

#include "util/gnn_macros.h"
#include "nn/factor.h"

namespace gnn
{

template<typename mode, typename Dtype>
class ReLU : public Factor
{
public:
	ReLU(std::string _name, PropErr _properr = PropErr::T);

};

}

#endif