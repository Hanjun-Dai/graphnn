#ifndef T_SHAPE_H
#define T_SHAPE_H

#include "gnn_macros.h"
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
	TShape(std::initializer_list<uint> l);

	void Reshape(std::initializer_list<uint> l);
	size_t Count();
	
	std::vector<uint> dims;
};

}

#endif