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
	/**
	 * @brief      constructor
	 *
	 * @param[in]  l     the list specifying the shape. For example, {3, 4} creates a 
	 * 					matrix with 3 rows and 4 cols.
	 */
	TShape(std::vector<size_t> l);

	/**
	 * @brief      reshape this shape representation
	 *
	 * @param[in]  l     a list specifying the new shape
	 */
	void Reshape(std::vector<size_t> l);

	/**
	 * @brief      count # elements starting from dim dimension
	 *
	 * @param[in]  dim   The dim to start (default is 0)
	 *
	 * @return     \prod_{i=dim}^{rank - 1} dims[i]
	 */
	inline size_t Count(uint dim = 0)
	{
		if (dim == 0 && this->dims.size() == 0)
			return 0;
		ASSERT(dim < this->dims.size(), "dim is out of range");
		return cnt[dim];
	}
	/**
	 * @brief      get the size of a certain dimension
	 *
	 * @param[in]  dim   The dim
	 *
	 * @return     the size of {dim} dimension
	 */
	size_t operator[](uint dim);

	/**
	 * @brief      coordicate to 1d index
	 *
	 * @param[in]  l     coordinates
	 *
	 * @return     scalar index
	 */
	size_t Coor2Idx(const std::vector<size_t>& l);

	/**
	 * @brief      Returns a string representation of the object.
	 *
	 * @return     String representation of the object.
	 */
	std::string toString();

	/**
	 * stores the size per each dimension
	 */
	std::vector<size_t> dims;

	std::vector<size_t> cnt;
};

inline bool operator==(const TShape& lhs, const TShape& rhs)
{
	if (lhs.dims.size() != rhs.dims.size())
		return false;
	for (size_t i = 0; i < lhs.dims.size(); ++i)
		if (lhs.dims[i] != rhs.dims[i])
			return false;
	return true;
}

}

#endif