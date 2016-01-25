#ifndef DENSE_VECTOR_H
#define DENSE_VECTOR_H

#include "imatrix.h"
#include <cassert>

template<MatMode mode, typename Dtype>
class Vector
{
public:
};

template<typename Dtype>
class Vector<CPU, Dtype>
{
public:
		~Vector();
		Vector();		
		Vector(size_t _count);
		
		virtual void Serialize(FILE* fid) 
		{
			assert(fwrite(&count, sizeof(size_t), 1, fid) == 1);
			assert(fwrite(&mem_size, sizeof(size_t), 1, fid) == 1);
			assert(fwrite(data, sizeof(Dtype), mem_size, fid) == mem_size);
		}
		
		virtual void Deserialize(FILE* fid)
		{
			assert(fread(&count, sizeof(size_t), 1, fid) == 1);
			assert(fread(&mem_size, sizeof(size_t), 1, fid) == 1);
			MatUtils<CPU>::DelArr(data);
			MatUtils<CPU>::MallocArr(data, sizeof(Dtype) * mem_size);
			assert(fread(data, sizeof(Dtype), mem_size, fid) == mem_size);
		}
		
		void Resize(size_t _count);
		void Fill(Dtype scalar);
		
		Dtype* data;
		size_t count, mem_size;	
};

template<typename Dtype>
class Vector<GPU, Dtype>
{
public:
		~Vector();
		Vector();
		Vector(size_t _count, unsigned int _streamid = 0U);
        
		virtual void Serialize(FILE* fid) 
		{
			assert(fwrite(&count, sizeof(size_t), 1, fid) == 1);
			assert(fwrite(&mem_size, sizeof(size_t), 1, fid) == 1);
            Dtype* buf = new Dtype[mem_size];
            cudaMemcpy(buf, data, sizeof(Dtype) * mem_size, cudaMemcpyDeviceToHost);
			assert(fwrite(buf, sizeof(Dtype), mem_size, fid) == mem_size);
            delete[] buf;
		}
		
		virtual void Deserialize(FILE* fid)
		{
			assert(fread(&count, sizeof(size_t), 1, fid) == 1);
			assert(fread(&mem_size, sizeof(size_t), 1, fid) == 1);
			MatUtils<CPU>::DelArr(data);
			MatUtils<CPU>::MallocArr(data, sizeof(Dtype) * mem_size);
            Dtype* buf = new Dtype[mem_size];	        
			assert(fread(buf, sizeof(Dtype), mem_size, fid) == mem_size);
            cudaMemcpy(data, buf, sizeof(Dtype) * mem_size, cudaMemcpyHostToDevice);
            delete[] buf;
		}
        
		void Resize(size_t _count);
		void Fill(Dtype scalar);
		
		void CopyFrom(Vector<CPU, Dtype>& src);
		
		Dtype* data;
		size_t count, mem_size;	
		unsigned streamid;		
};

#endif