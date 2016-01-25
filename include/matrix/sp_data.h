#ifndef SP_DATA_H
#define SP_DATA_H

#include "matrix_utils.h"

template<MatMode mode, typename Dtype>
class SpData
{
public:
	inline SpData()
	{
		nnz = len_ptr = nzCap = ptrCap = 0;
		val = nullptr;
		col_idx = ptr = nullptr;
	}
	
	inline SpData(int newNzCap, int newPtrCap)
	{
		nnz = len_ptr = 0;
		nzCap = newNzCap; 
		ptrCap = newPtrCap;
		MatUtils<mode>::MallocArr(val, sizeof(Dtype) * nzCap);
        MatUtils<mode>::MallocArr(col_idx, sizeof(int) * nzCap);
		MatUtils<mode>::MallocArr(ptr, sizeof(int) * ptrCap);
	}
	
	void Serialize(FILE* fid)
	{
		assert(fwrite(&nnz, sizeof(int), 1, fid) == 1);
		assert(fwrite(&len_ptr, sizeof(int), 1, fid) == 1);
		assert(fwrite(&nzCap, sizeof(int), 1, fid) == 1);
		assert(fwrite(&ptrCap, sizeof(int), 1, fid) == 1);
	
        int *p_col_idx = col_idx, *p_ptr = ptr;
        Dtype* p_val = val;
        if (mode == GPU)
        {
            p_val = new Dtype[nzCap];
            p_col_idx = new int[nzCap];
            p_ptr = new int[ptrCap];
            cudaMemcpy(p_val, val, sizeof(Dtype) * nzCap, cudaMemcpyDeviceToHost);
            cudaMemcpy(p_col_idx, col_idx, sizeof(int) * nzCap, cudaMemcpyDeviceToHost);
            cudaMemcpy(p_ptr, ptr, sizeof(int) * ptrCap, cudaMemcpyDeviceToHost);
        }
		assert(fwrite(p_val, sizeof(Dtype), nzCap, fid) == nzCap);
		assert(fwrite(p_col_idx, sizeof(int), nzCap, fid) == nzCap);
		assert(fwrite(p_ptr, sizeof(int), ptrCap, fid) == ptrCap);
        if (mode == GPU)
        {
            delete[] p_val;
            delete[] p_col_idx;
            delete[] p_ptr;
        }
	}
	
	void Deserialize(FILE* fid)
	{
		assert(fread(&nnz, sizeof(int), 1, fid) == 1);
		assert(fread(&len_ptr, sizeof(int), 1, fid) == 1);
		assert(fread(&nzCap, sizeof(int), 1, fid) == 1);
		assert(fread(&ptrCap, sizeof(int), 1, fid) == 1);
		
		MatUtils<mode>::DelArr(val);
        MatUtils<mode>::DelArr(col_idx);
		MatUtils<mode>::DelArr(ptr);
		MatUtils<mode>::MallocArr(val, sizeof(Dtype) * nzCap);
        MatUtils<mode>::MallocArr(col_idx, sizeof(int) * nzCap);
		MatUtils<mode>::MallocArr(ptr, sizeof(int) * ptrCap);
	
        int *p_col_idx = col_idx, *p_ptr = ptr;
        Dtype* p_val = val;
        if (mode == GPU)
        {
            p_val = new Dtype[nzCap];
            p_col_idx = new int[nzCap];
            p_ptr = new int[ptrCap];                       
        }        
		assert(fread(p_val, sizeof(Dtype), nzCap, fid) == nzCap);
		assert(fread(p_col_idx, sizeof(int), nzCap, fid) == nzCap);
		assert(fread(p_ptr, sizeof(int), ptrCap, fid) == ptrCap);
        if (mode == GPU)
        {
            cudaMemcpy(val, p_val, sizeof(Dtype) * nzCap, cudaMemcpyHostToDevice);
            cudaMemcpy(col_idx, p_col_idx, sizeof(int) * nzCap, cudaMemcpyHostToDevice);
            cudaMemcpy(ptr, p_ptr, sizeof(int) * ptrCap, cudaMemcpyHostToDevice);
            delete[] p_val;
            delete[] p_col_idx;
            delete[] p_ptr;               
        }
	}

	~SpData()
	{
		MatUtils<mode>::DelArr(val);
        MatUtils<mode>::DelArr(col_idx);
		MatUtils<mode>::DelArr(ptr);
	}

	Dtype* val;
	int* col_idx;
	int* ptr;
	
	int nnz;
	int len_ptr;
	int nzCap;
	int ptrCap;
};

#endif