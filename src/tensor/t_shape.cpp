#include <iostream>
#include "tensor/t_shape.h"
#include "util/fmt.h"

namespace gnn
{

TShape::TShape() : dims(), cnt()
{

}

TShape::TShape(std::vector<size_t> l)
{
	Reshape(l);
}

void TShape::Reshape(std::vector<size_t> l)
{
	this->dims = l;
	if (l.size())
	{
		cnt.resize(l.size());
		cnt[cnt.size() - 1] = this->dims[cnt.size() - 1];
		for (int i = (int)cnt.size() - 2; i >= 0; --i)
			cnt[i] = cnt[i + 1] * this->dims[i];
	}
}

std::string TShape::toString()
{
	std::string ans = "[";
	for (size_t i = 0; i < dims.size(); ++i)
	{
		ans += fmt::sprintf("%d", dims[i]);
		if (i + 1 < dims.size())
			ans += ",";
	}
	ans += "]";
	return ans;
}

// size_t TShape::Count(uint dim)
// {
// 	if (dim == 0 && this->dims.size() == 0)
// 		return 0;
// 	ASSERT(dim < this->dims.size(), fmt::sprintf("dim %d is out of range", dim));
// 	size_t result = 1;
// 	for (size_t i = dim; i < this->dims.size(); ++i)
// 		result *= (size_t)this->dims[i];
// 	return result;
// }

size_t TShape::operator[](uint dim)
{
	ASSERT(dim < this->dims.size(), fmt::sprintf("dim %d is out of range", dim));
	return this->dims[dim];
}

size_t TShape::Coor2Idx(const std::vector<size_t>& l)
{
	ASSERT(l.size() == dims.size(), "rank mismatch");
	size_t ans = 0;
	for (size_t i = 0; i < l.size(); ++i)
	{
		ASSERT(l[i] < dims[i], "coordinate out of range");
		ans = ans * dims[i] + l[i];
	}
	return ans;
}

}