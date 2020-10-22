#include <cstring>
#include <vector>
#include "cuda_runtime.h"
#include "cuda_fp16.h"

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "PluginUtils.h"
#include "Adder2dPlugin.h"


////adder cuda kernel
//template <typename Ftype, unsigned int blockSize>
//__global__ void filterSum(int *g_idata, int *g_odata, unsigned int n)
//{
//    extern __shared__ int sdata[];
//    unsigned int tid = threadIdx.x;
//    unsigned int i = blockIdx.x*(blockSize*2) + tid;
//    unsigned int gridSize = blockSize*2*gridDim.x;
//    sdata[tid] = 0;
//
//    while (i < n)
//    {
//        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
//        i += gridSize;
//    }
//
//    __syncthreads();
//
//    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
//    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
//    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
//
//    if (tid < 32)
//    {
//        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
//        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
//        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
//        if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
//        if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
//        if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
//    }
//
//    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
//}

// for consistency I recommend all plugin have same namesapce and version
const char* G_PLUGIN_NAMESPACE = "_TRT";
const char* G_PLUGIN_VERSION = "1";
const char* G_ADDER2D_TYPE = "Adder2d";
const char* G_ADDER2D_NAME = "Adder2d_TRT"; //plugin_name = plugin_type + plugin_namespace


Adder2dPlugin::Adder2dPlugin(const nvinfer1::Weights *weights, int nbWeights, int filterSize, int nbFilters, int stride,
                             int padding) {
    mWeights = weights[0];
    mWeights.values = malloc(mWeights.count * type2size(mWeights.type));
    memcpy(const_cast<void *>(mWeights.values), weights[0].values, mWeights.count * type2size(mWeights.type));
    mNbWeights = nbWeights;
    mFilterSize = filterSize;
    mNbFilters = nbFilters;
    mStride = stride;
    mPadding = padding;

}

// create the plugin at runtime from a byte stream
Adder2dPlugin::Adder2dPlugin(const void *data, size_t length) {
    const char *d = static_cast<const char *>(data), *a = d;
    read<int>(d, mNbInputChannels);
    read<int>(d, mNbInputHeight);
    read<int>(d, mNbInputWidth);
    read<int>(d, mNbWeights);
    read<int>(d, mFilterSize);
    read<int>(d, mNbFilters);
    read<int>(d, mStride);
    read<int>(d, mPadding);
    read<nvinfer1::DataType>(d, mDataType);
    read<int64_t>(d, mWeights.count);
    read<nvinfer1::DataType>(d, mWeights.type);
    mWeights.values = nullptr;
    mWeights.values = malloc(mWeights.count * type2size(mWeights.type));
    memcpy(const_cast<void *>(mWeights.values), d, mWeights.count * type2size(mWeights.type));
    d = d + mWeights.count * type2size(mWeights.type);
    ASSERT(d == a + length);
}

size_t Adder2dPlugin::getSerializationSize() const {
    return sizeof(mNbInputChannels) + sizeof(mNbInputWidth) + sizeof(mNbInputHeight) + sizeof(mFilterSize) +
           sizeof(mNbFilters) + sizeof(mStride) + sizeof(mPadding) + sizeof(mDataType) + sizeof(mWeights.count) +
           sizeof(mWeights.type) + mWeights.count * type2size(mWeights.type);
}

void Adder2dPlugin::serialize(void *buffer) const {
    char *d = static_cast<char *>(buffer), *a = d;
    write(d, mNbInputChannels);
    write(d, mNbInputHeight);
    write(d, mNbInputWidth);
    write(d, mNbWeights);
    write(d, mFilterSize);
    write(d, mNbFilters);
    write(d, mStride);
    write(d, mPadding);
    write(d, mDataType);
    write(d, mWeights.count);
    write(d, mWeights.type);
    convertAndCopyToBuffer(d, mWeights, mWeights.type);
    ASSERT(d == a + getSerializationSize());
}

Adder2dPlugin::~Adder2dPlugin() {
    if (mWeights.values)
    {
        free(const_cast<void *>(mWeights.values));
        mWeights.values = nullptr;
    }

//    if (mDeviceKernel)
//    {
//        cudaFree(mDeviceKernel);
//        mDeviceKernel = nullptr;
//    }
}

int Adder2dPlugin::getNbOutputs() const {
    return 1;
}

nvinfer1::Dims Adder2dPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) {
    if(index == 0) {
        // CHW
        nvinfer1::Dims dimsOutput;
        dimsOutput.nbDims = inputs->nbDims;
        dimsOutput.d[0] = inputs->d[0];
        dimsOutput.d[1] = (inputs->d[1] + 2 * mPadding - mFilterSize) / mStride + 1;
        dimsOutput.d[2] = (inputs->d[2] + 2 * mPadding - mFilterSize) / mStride + 1;
        dimsOutput.d[3] = mNbFilters;
        return dimsOutput;
    } // else if(index == n) {
        // for other outputs if exists.
    // }
    else {
        ASSERT(false);
    }
}


bool Adder2dPlugin::supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const {
    return (type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF)
            && format == nvinfer1::PluginFormat::kNCHW;
}

void Adder2dPlugin::configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs,
                                        const nvinfer1::Dims* outputDims, int nbOutputs,
                                        nvinfer1::DataType type, nvinfer1::PluginFormat format,
                                        int maxBatchSize) {
    ASSERT((type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF)
            && format == nvinfer1::PluginFormat::kNCHW);
    mNbInputChannels = inputDims[0].d[0];
    mNbInputHeight = inputDims[0].d[1];
    mNbInputWidth = inputDims[0].d[2];
    mDataType = type;
}

int Adder2dPlugin::initialize() {
//    convertAndCopyToDeivce(mDeviceKernel, mWeights, mDataType);
    return 0;
}

void Adder2dPlugin::terminate() {
    if (mWeights.values)
    {
        free(const_cast<void *>(mWeights.values));
        mWeights.values = nullptr;
    }
//    if (mDeviceKernel)
//    {
//        cudaFree(mDeviceKernel);
//        mDeviceKernel = nullptr;
//    }
}

size_t Adder2dPlugin::getWorkspaceSize(int maxBatchSize) const{
    return 0;
}

int Adder2dPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream)
{
//    const int count = batchSize * mNbInputChannels * mNbInputWidth * mNbInputHeight;
//    const int channels = mNbInputChannels;
//    const int dim = mNbInputWidth * mNbInputHeight;
//    const int div_factor = 1;
//    if (mDataType == nvinfer1::DataType::kFLOAT)
//    {
//        const float zerof{0.0f};
//        CUDA_CHECK(Forward_gpu(count, channels, dim,
//                            reinterpret_cast<const float *>(mDeviceKernel),
//                            reinterpret_cast<const float *>(inputs[0]),
//                            reinterpret_cast<float *>(outputs[0]),
//                            zerof,
//                            div_factor,
//                            stream));
//    }
//#ifdef FP16_PRELU
//    else
//    {
//        const __half zeroh = __half(0.0f);
//        CUDA_CHECK(Forward_gpu(count, channels, dim,
//                            reinterpret_cast<const __half *>(mDeviceKernel),
//                            reinterpret_cast<const __half *>(inputs[0]),
//                            reinterpret_cast<__half *>(outputs[0]),
//                            zeroh,
//                            div_factor,
//                            stream));
//    }
//#else
//    else
//    {
//        spdlog::error("fp16 prelu is unsupported");
//        ASSERT(false);
//    }
//#endif
    return 0;
}

const char *Adder2dPlugin::getPluginType() const {
    return G_ADDER2D_TYPE;
}

const char *Adder2dPlugin::getPluginVersion() const {
    return G_PLUGIN_VERSION;
}

void Adder2dPlugin::destroy() {
    delete this;
}

nvinfer1::IPluginV2* Adder2dPlugin::clone() const {
    return new Adder2dPlugin(&mWeights, mNbWeights, mFilterSize, mNbFilters, mStride, mPadding);
}

void Adder2dPlugin::setPluginNamespace(const char* pluginNamespace) {

}

const char* Adder2dPlugin::getPluginNamespace() const {
    return G_PLUGIN_NAMESPACE;
}




Adder2dPluginCreator::Adder2dPluginCreator()  {
    mPluginAttributes.emplace_back(nvinfer1::PluginField("weights", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("nbWeight", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("filterSize", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("nbFilters", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("stride", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("padding", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

// return ADDER2D_PLUGIN_TYPE + ADDER2D_PLUGIN_NAMESPACE
const char* Adder2dPluginCreator::getPluginName() const {
    // std::string plugin_type{G_ADDER2D_TYPE};
    // std::string plugin_namespace{G_PLUGIN_NAMESPACE};
    // return (plugin_type+plugin_namespace).c_str();
    return G_ADDER2D_NAME;
}

const char* Adder2dPluginCreator::getPluginVersion() const {
    return G_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* Adder2dPluginCreator::getFieldNames() {
    return &mFC;
}

nvinfer1::IPluginV2* Adder2dPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) {
    int nbWeights, filterSize, nbFilters, stride, padding;
    std::vector<float> weightValues;
    const nvinfer1::PluginField* fields = fc->fields;
    for (int i=0; i<fc->nbFields; i++) {
        const char* attrName = fields[i].name;
        if(strcmp(attrName, "weights")) {
            ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
            weightValues.reserve(fields[i].length);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < weightValues.size(); j++)
            {
                weightValues.push_back(*w);
                w++;
            }
        }
        if(strcmp(attrName, "nbWeights")) {
            ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
            nbWeights = *(static_cast<const int*>(fields[i].data));
        }
        if(strcmp(attrName, "filterSize")) {
            ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
            filterSize = *(static_cast<const int*>(fields[i].data));
        }
        if(strcmp(attrName, "nbFilters")) {
            ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
            nbFilters = *(static_cast<const int*>(fields[i].data));
        }
        if(strcmp(attrName, "stride")) {
            ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
            stride = *(static_cast<const int*>(fields[i].data));
        }
        if(strcmp(attrName, "padding")) {
            ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
            padding = *(static_cast<const int*>(fields[i].data));
        }
    }
    nvinfer1::Weights weights{nvinfer1::DataType::kFLOAT, weightValues.data(), (int64_t)weightValues.size()};
    return new Adder2dPlugin(&weights, nbWeights, filterSize, nbFilters, stride, padding);
}

// deserialization plugin implementation
nvinfer1::IPluginV2* Adder2dPluginCreator::deserializePlugin(const char *layerName, const void *serialData, size_t serialLength) {
    return new Adder2dPlugin(serialData, serialLength);
}

const char* Adder2dPluginCreator::getPluginNamespace() const {
    return G_PLUGIN_NAMESPACE;
}

REGISTER_TENSORRT_PLUGIN(Adder2dPluginCreator);
