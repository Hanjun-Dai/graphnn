#ifndef FAST_WHT_H
#define FAST_WHT_H

#include "mat_typedef.h"

extern "C" {
#include "spiral_wht.h"
}

template<MatMode mode, typename Dtype>
class FastWHT {};

template<typename Dtype>
class FastWHT<CPU, Dtype>
{
public:
	FastWHT(unsigned int _degree);
	~FastWHT();
	
	void Transform(size_t num_rows, Dtype* data);
	
private:
	void InitBuffer();
	
	double* buffer;
	size_t buf_len;
	unsigned int degree;
	Wht* wht_tree;
};

template<typename Dtype>
class FastWHT<GPU, Dtype>
{
public:
	FastWHT(unsigned int _degree);
	~FastWHT();
	
	void Transform(size_t num_rows, Dtype* data);
	
private:
	unsigned int degree;
};

#endif