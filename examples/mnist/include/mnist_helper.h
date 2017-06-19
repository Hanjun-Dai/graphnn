#ifndef MNIST_HELPER_H
#define MNIST_HELPER_H

#include <algorithm>
#include <cstring>
#include <cassert>
#include <vector>

typedef float Dtype;

void LoadRaw(const char* f_image, const char* f_label, std::vector< Dtype* >& images, std::vector< int >& labels)
{
    FILE* fid = fopen(f_image, "r");
    int buf;
    assert(fread(&buf, sizeof(int), 1, fid) == 1); // magic number
    int num;
    assert(fread(&num, sizeof(int), 1, fid) == 1); // num
    num = __builtin_bswap32(num); // the raw data is high endian    
    assert(fread(&buf, sizeof(int), 1, fid) == 1); // rows 
    assert(fread(&buf, sizeof(int), 1, fid) == 1); // cols
    std::for_each(images.begin(), images.end(), [](Dtype*& val) {delete val;});
    images.clear();    
    unsigned char* buffer = new unsigned char[784];
    for (int i = 0; i < num; ++i)
    {
        assert(fread(buffer, sizeof(unsigned char), 784, fid) == 784);
        Dtype* img = new Dtype[784];
        for (unsigned j = 0; j < 784; ++j)
            img[j] = buffer[j];
        images.push_back(img);            
    }    
    delete[] buffer;
    fclose(fid);    
    
    fid = fopen(f_label, "r");
    assert(fread(&buf, sizeof(int), 1, fid) == 1); // magic number    
    assert(fread(&num, sizeof(int), 1, fid) == 1); // num
    num = __builtin_bswap32(num); // the raw data is high endian
    buffer = new unsigned char[num];
    assert(fread(buffer, sizeof(unsigned char), num, fid) == (unsigned)num);
    fclose(fid);
    labels.clear();
    for (int i = 0; i < num; ++i)
        labels.push_back(buffer[i]);    
    delete[] buffer;        
}

#endif
