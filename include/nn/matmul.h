#ifndef MATMUL_H
#define MATMUL_H

#include "util/gnn_macros.h"
#include "nn/factor.h"
#include "nn/variable.h"

namespace gnn
{

template <class... Args>
struct type_list
{
   template <std::size_t N>
   using type = typename std::tuple_element<N, std::tuple<Args...>>::type;
};

template<typename mode, typename Dtype>
class MatMul : public Factor
{
public:
	typedef type_list< DTensorVar<mode, Dtype> > t_list;
	
	MatMul(std::string _name, PropErr _properr = PropErr::T);

};

}

#endif