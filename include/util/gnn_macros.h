#ifndef GNN_MACROS_H
#define GNN_MACROS_H

#include <cstddef>
#include <iostream>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstdio>

namespace gnn
{

#define NOT_IMPLEMENTED { throw std::logic_error(std::string("not implemented virtual func: ") + std::string(__FUNCTION__)); }

#define ASSERT(condition, message) \
   	do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            void *array[10]; \
            size_t size; \
            size = backtrace(array, 10); \
            backtrace_symbols_fd(array, size, STDERR_FILENO); \
            std::terminate(); \
        } \
    } while (false)

#define MAT_TYPE_SWITCH(type, matType, ...) \
    switch (type) {							\
    case MatType::dense:						\
      {										\
      	typedef DENSE matType;				\
       	{__VA_ARGS__}						\
      } 										\
      break;                                  \
    case MatType::sparse:						\
    	{										\
    		typedef SPARSE matType;				\
    		{__VA_ARGS__}						\
    	}										\
    	break;									\
    default:									\
    	throw std::logic_error("unknown type"); \
    }											\

#define ELE_TYPE_SWITCH(type, eleType, ...) \
    switch (type) {             \
    case EleType::FLOAT32:            \
      {                   \
        typedef float eleType;  \
        {__VA_ARGS__}           \
      }                     \
      break;                 \
    case EleType::FLOAT64:           \
      {                   \
        typedef double eleType;       \
        {__VA_ARGS__}           \
      }                   \
      break;                  \
    case EleType::INT32:           \
      {                   \
        typedef int eleType;       \
        {__VA_ARGS__}           \
      }                   \
      break;                  \
    default:                  \
      throw std::logic_error("unknown type"); \
    }                     \

typedef unsigned int uint;

enum class MatType
{
  dense,
  sparse
};

enum class PropErr
{
	N = 0,
	T = 1	
};

enum class Trans
{
  N = 0,
  T = 1
};

enum class MatMode
{
	cpu = 0,
	gpu = 1
};

enum class Phase
{
    TRAIN = 0,
    TEST = 1  
};

enum class EleType
{
  FLOAT32 = 0, 
  FLOAT64 = 1,
  INT32 = 2
};

template<typename Dtype>
inline EleType Dtype2Enum();

template<>
inline EleType Dtype2Enum<float>() { return EleType::FLOAT32; }

template<>
inline EleType Dtype2Enum<double>() { return EleType::FLOAT64; }

template<>
inline EleType Dtype2Enum<int>() { return EleType::INT32; }

struct CPU
{
  static const MatMode mode = MatMode::cpu;
};

struct GPU
{
	static const MatMode mode = MatMode::gpu;
};

struct DENSE
{
	static const MatType type = MatType::dense;
};

struct SPARSE
{
	static const MatType type = MatType::sparse;
};

}

#endif