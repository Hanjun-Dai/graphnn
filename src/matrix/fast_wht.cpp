#include "fast_wht.h"
#include <cmath>

template<typename Dtype>
FastWHT<CPU, Dtype>::FastWHT(unsigned int _degree)
					: degree(_degree)
{
	buf_len = 1 << _degree;
	wht_tree = wht_get_tree(_degree);
	InitBuffer();
}

template<typename Dtype>
FastWHT<CPU, Dtype>::~FastWHT()
{
	if (buffer)
		delete[] buffer;
}

template<>
void FastWHT<CPU, float>::InitBuffer()
{
	buffer = new double[buf_len];
}

template<>
void FastWHT<CPU, double>::InitBuffer()
{
	buffer = nullptr;
}

template<>
void FastWHT<CPU, float>::Transform(size_t num_rows, float* data)
{
	for (unsigned row = 0; row < num_rows; ++row)
	{
		for (unsigned i = 0; i < buf_len; ++i)
			buffer[i] = (double)data[i];
		wht_apply(wht_tree, 1, buffer);
		for (unsigned i = 0; i < buf_len; ++i)
			data[i] = (float)buffer[i];
		data += buf_len;
	}
}

template<>
void FastWHT<CPU, double>::Transform(size_t num_rows, double* data)
{
	for (unsigned i = 0; i < num_rows; ++i)
	{
		wht_apply(wht_tree, 1, data + i * buf_len);
	}
}

template class FastWHT<CPU, double>;
template class FastWHT<CPU, float>;