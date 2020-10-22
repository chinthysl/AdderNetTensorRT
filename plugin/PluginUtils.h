#ifndef PLUGIN_UTILS_H
#define PLUGIN_UTILS_H

#include "NvInfer.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"

#include <iostream>
#include <numeric>

// this is for debug, and you can find a lot assert in plugin implementation,
// it will reduce the time spend on debug
#define ASSERT(assertion)                                                                   \
{                                                                                           \
    if (!(assertion))                                                                       \
    {                                                                                       \
        std::cerr << "#assertion fail " << __FILE__ << " line " << __LINE__ << std::endl;   \
        abort();                                                                            \
    }                                                                                       \
}

#define UNUSED(unusedVariable) (void)(unusedVariable)
// suppress compiler warning: unused parameter

inline int64_t volume(const nvinfer1::Dims& d){
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t){
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        default: throw std::runtime_error("Invalid DataType.");
    }
}


#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                                 \
    {                                                                                                       \
        cudaError_t error_code = callstr;                                                                   \
        if (error_code != cudaSuccess) {                                                                    \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(0);                                                                                        \
        }                                                                                                   \
    }
#endif

inline void* safeCudaMalloc(size_t memSize) {
    void* deviceMem;
    CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr) {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

inline void safeCudaFree(void* deviceMem) {
    CUDA_CHECK(cudaFree(deviceMem));
}

inline void error(const std::string& message, const int line, const std::string& function, const std::string& file) {
    std::cout << message << " at " << line << " in " << function << " in " << file << std::endl;
}


// write value to buffer
template <typename T>
void write(char *&buffer, const T &val)
{
    *reinterpret_cast<T *>(buffer) = val;
    buffer += sizeof(T);
}

// read value from buffer
template <typename T>
void read(const char *&buffer, T &val)
{
    val = *reinterpret_cast<const T *>(buffer);
    buffer += sizeof(T);
}



// return needed space of a datatype
size_t type2size(nvinfer1::DataType type);

// copy data to device memory
void* copyToDevice(const void* data, size_t count);

// copy data to buffer.
void copyToBuffer(char*& buffer, const void* data, size_t count);

// convert data to datatype and copy it to device
void convertAndCopyToDeivce(void*& deviceWeights, const nvinfer1::Weights &weights,
                            nvinfer1::DataType datatype);

// convert data to datatype and copy it to buffer
void convertAndCopyToBuffer(char*& buffer, const nvinfer1::Weights weights,
                            nvinfer1::DataType datatype);

// deserialize buffer to device memory.
void deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size);

#endif //PLUGIN_UTILS_H