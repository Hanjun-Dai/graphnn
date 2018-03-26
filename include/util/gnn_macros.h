#ifndef GNN_MACROS_H
#define GNN_MACROS_H

#include <cstddef>
#include <iostream>
#include <cstring>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstdio>

namespace gnn
{

#ifdef USE_GPU
  #define INSTANTIATE_CLASS(classname) \
    template class classname<CPU, float>; \
    template class classname<CPU, double>; \
    template class classname<GPU, float>; \
    template class classname<GPU, double>;
#else
  #define INSTANTIATE_CLASS(classname) \
    template class classname<CPU, float>; \
    template class classname<CPU, double>;
#endif

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
    		typedef CSR_SPARSE matType;				\
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

#ifdef USE_GPU
#define MAT_MODE_SWITCH(mode, matMode, ...) \
    switch (mode) { \
      case MatMode::cpu: \
      { \
        typedef CPU matMode; \
        {__VA_ARGS__} \
      } \
      break; \
      case MatMode::gpu: \
      { \
        typedef GPU matMode; \
        {__VA_ARGS__} \
      } \
      break; \
      default: \
        throw std::logic_error("unknown mode"); \
    }
#else
#define MAT_MODE_SWITCH(mode, matMode, ...) \
    switch (mode) { \
      case MatMode::cpu: \
      { \
        typedef CPU matMode; \
        {__VA_ARGS__} \
      } \
      break; \
      case MatMode::gpu: \
      { \
        throw std::logic_error("gpu mode is not enabled"); \
      } \
      break; \
      default: \
        throw std::logic_error("unknown mode"); \
    }
#endif

typedef unsigned int uint;

enum class MatType
{
  dense,
  sparse,
  row_sparse
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
  INT32 = 2,
  UNKNOWN = 3,
};

template<typename Dtype>
inline EleType Dtype2Enum();

template<>
inline EleType Dtype2Enum<float>() { return EleType::FLOAT32; }

template<>
inline EleType Dtype2Enum<double>() { return EleType::FLOAT64; }

template<>
inline EleType Dtype2Enum<int>() { return EleType::INT32; }

/**
 * @brief      CPU token; used for template parsing
 */
struct CPU
{
  static const MatMode type = MatMode::cpu;
};

/**
 * @brief      GPU token; used for template parsing
 */
struct GPU
{
	static const MatMode type = MatMode::gpu;
};

/**
 * @brief      DENSE tensor token; used for template parsing
 */
struct DENSE
{
	static const MatType type = MatType::dense;
};

/**
 * @brief      CSR SPARSE tensor token; used for template parsing
 */
struct CSR_SPARSE
{
	static const MatType type = MatType::sparse;
};

/**
 * @brief      ROW SPARSE tensor token; used for template parsing
 */
struct ROW_SPARSE
{
  static const MatType type = MatType::row_sparse;
};
}

#endif
