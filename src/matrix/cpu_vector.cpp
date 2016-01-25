#include "vector.h"

template<typename Dtype>
Vector<CPU, Dtype>::~Vector()
{
	MatUtils<CPU>::DelArr(data);
}

template<typename Dtype>
Vector<CPU, Dtype>::Vector()
{
	count = mem_size = 0U;
	data = nullptr;
}

template<typename Dtype>
Vector<CPU, Dtype>::Vector(size_t _count)
{	
	count = _count;
	mem_size = count;
	MatUtils<CPU>::MallocArr(data, sizeof(Dtype) * mem_size);
}

template<typename Dtype>
void Vector<CPU, Dtype>::Resize(size_t _count)
{	
	count = _count;
	if (_count > mem_size)
	{
		mem_size = _count;
		MatUtils<CPU>::DelArr(data);
		MatUtils<CPU>::MallocArr(data, sizeof(Dtype) * mem_size);		
	}
}

template<typename Dtype>
void Vector<CPU, Dtype>::Fill(Dtype scalar)
{	
	for (size_t i = 0; i < count; ++i)
		data[i] = scalar;
}


template class Vector<CPU, double>;
template class Vector<CPU, float>;