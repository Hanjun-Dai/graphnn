#ifndef T_SHAPE_H
#define T_SHAPE_H

#include "util/gnn_macros.h"
#include <vector>

namespace gnn 
{

/**
 * @brief      Class for shape.
 */
class TShape
{
public:
	TShape();
	TShape(std::vector<size_t> l);

	void Reshape(std::vector<size_t> l);
	size_t Count(uint dim = 0);

	size_t operator[](uint dim);

	std::vector<size_t> dims;
};

}

#endif