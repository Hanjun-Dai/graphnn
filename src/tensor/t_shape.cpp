#include "tensor/t_shape.h"

namespace gnn
{

TShape::TShape() : dims()
{

}

TShape::TShape(std::initializer_list<uint> l) : dims(l)
{

}

void TShape::Reshape(std::initializer_list<uint> l)
{
	this->dims = l;
}

size_t TShape::Count()
{
	size_t result = 1;
	for (size_t i = 0; i < this->dims.size(); ++i)
		result *= (size_t)this->dims[i];
	return result;
}

}