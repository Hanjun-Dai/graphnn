#include <iostream>
#include "tensor/t_shape.h"
#include "fmt/printf.h"

namespace gnn
{

TShape::TShape() : dims()
{

}

TShape::TShape(std::vector<size_t> l) : dims(l)
{

}

void TShape::Reshape(std::vector<size_t> l)
{
	this->dims = l;
}

size_t TShape::Count(uint dim)
{
	if (dim == 0 && this->dims.size() == 0)
		return 0;
	ASSERT(dim < this->dims.size(), fmt::sprintf("dim %d is out of range", dim));
	size_t result = 1;
	for (size_t i = dim; i < this->dims.size(); ++i)
		result *= (size_t)this->dims[i];
	return result;
}

size_t TShape::operator[](uint dim)
{
	ASSERT(dim < this->dims.size(), fmt::sprintf("dim %d is out of range", dim));
	return this->dims[dim];
}

}