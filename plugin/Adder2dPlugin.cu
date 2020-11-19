#include <cstring>
#include <vector>
#include "cuda_runtime.h"
#include "cuda_fp16.h"

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "PluginUtils.h"
#include "Adder2dPlugin.h"


//cuda kernel of the Adder Filter
template <typename Ftype>
__global__ void AdderFilter(int in_c, int in_h, int in_w, int k, int stride, int padding,
                            int out_h, int out_w, const Ftype* input, Ftype* output, const Ftype* weights)
{
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid = tid_y*out_w + tid_x;
    int filterIdx = blockIdx.x;
    int out_idx = out_h * out_w * filterIdx + tid;
    output[out_idx] = 0.0;

    for(int a=0; a<in_c; a++)
    {
        for(int i=0; i<k; i++)
        {
            for(int j=0; j<k; j++)
            {
                Ftype val;

                int shift = k/2;
                if(padding==0)
                {
                    shift = 0;
                }
                int input_pos_y = tid_y*stride + i - shift;
                int input_pos_x = tid_x*stride + j - shift;
                int input_idx = a*(in_h*in_w) + input_pos_y*in_w + input_pos_x;

                if(input_pos_y<0 || input_pos_y>in_h-1 || input_pos_x<0 || input_pos_x>in_w-1)
                {
                    val = 0.0;
                }
                else
                {
                    val = input[input_idx];
                }

//                printf("tid_x:%d, tid_y:%d, tid:%d, out_idx:%d, input_pos_y:%d, input_pos_x:%d, input_idx:%d, val:%d\n",
//                        tid_x, tid_y, tid, out_idx, input_pos_y, input_pos_x, input_idx, val);

                int weight_idx = filterIdx*in_c*k*k + a*k*k + i*k+ j;
                output[out_idx] += fabs(val - weights[weight_idx]);
            }
        }
    }
}

//cuda kernel to make values zero
template <typename Ftype>
__global__ void MakeOutputNegative(int elements, Ftype* output)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    if(tid < elements)
    {
        output[tid] = -output[tid];
    }
}

template <typename Dtype>
cudaError_t ForwardGpu(int n_filters,int in_c, int in_h, int in_w, int k, int stride, int pad,
                       const Dtype* input, Dtype* output, const Dtype* weights, const cudaStream_t stream)
{
    int out_h = (in_h + 2*pad - k) / stride + 1;
    int out_w = (in_w + 2*pad - k) / stride + 1;
    int nOutElements =  n_filters * out_h * out_w;

    dim3 blkDim(out_w,out_h, 1);
    dim3 gridDim(n_filters,1,1);

    AdderFilter<<<n_filters, blkDim, 0, stream>>>(in_c, in_h, in_w, k, stride, pad, out_h, out_w, input, output, weights);
    MakeOutputNegative<<<nOutElements/256+1, 256, 0, stream>>>(nOutElements, output);
    cudaError_t err = cudaGetLastError();
    return err;
}


// for consistency I recommend all plugin have same namesapce and version
const char* G_PLUGIN_NAMESPACE = "_TRT";
const char* G_PLUGIN_VERSION = "1";
const char* G_ADDER2D_TYPE = "Adder2d";
const char* G_ADDER2D_NAME = "Adder2d_TRT"; //plugin_name = plugin_type + plugin_namespace


Adder2dPlugin::Adder2dPlugin(const nvinfer1::Weights *weights, int nbWeights, int nbInputChannels, int inputHeight,
                             int inputWidth, int filterSize, int nbFilters, int stride, int padding) {
    mWeights = weights[0];
    mWeights.values = malloc(mWeights.count * type2size(mWeights.type));
    memcpy(const_cast<void *>(mWeights.values), weights[0].values, mWeights.count * type2size(mWeights.type));
    mNbWeights = nbWeights;
    mNbInputChannels = nbInputChannels;
    mInputHeight = inputHeight;
    mInputWidth = inputWidth;
    mFilterSize = filterSize;
    mNbFilters = nbFilters;
    mStride = stride;
    mPadding = padding;
}

// create the plugin at runtime from a byte stream
Adder2dPlugin::Adder2dPlugin(const void *data, size_t length) {
    const char *d = static_cast<const char *>(data), *a = d;
    read<int>(d, mNbWeights);
    read<int>(d, mNbInputChannels);
    read<int>(d, mInputHeight);
    read<int>(d, mInputWidth);
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
    return sizeof(mNbWeights) + sizeof(mNbInputChannels) + sizeof(mInputHeight) + sizeof(mInputWidth) +
           sizeof(mFilterSize) + sizeof(mNbFilters) + sizeof(mStride) + sizeof(mPadding) + sizeof(mDataType) +
           sizeof(mWeights.count) + sizeof(mWeights.type) + mWeights.count * type2size(mWeights.type);
}

void Adder2dPlugin::serialize(void *buffer) const {
    char *d = static_cast<char *>(buffer), *a = d;
    write(d, mNbWeights);
    write(d, mNbInputChannels);
    write(d, mInputHeight);
    write(d, mInputWidth);
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

    if (mDeviceWeightPtr)
    {
        cudaFree(mDeviceWeightPtr);
        mDeviceWeightPtr = nullptr;
    }
}

int Adder2dPlugin::getNbOutputs() const {
    return 1;
}

nvinfer1::Dims Adder2dPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) {
    if(index == 0) {
        // CHW
        nvinfer1::Dims dimsOutput;
        dimsOutput.nbDims = inputs->nbDims;
//        std::cout << "Input nbDims:" << inputs->nbDims << std::endl;
        dimsOutput.d[0] = mNbFilters;
        dimsOutput.d[1] = (inputs->d[1] + 2 * mPadding - mFilterSize) / mStride + 1;
        dimsOutput.d[2] = (inputs->d[2] + 2 * mPadding - mFilterSize) / mStride + 1;

//        std::cout << "mPadding:" << mPadding << ",mFilterSize:" << mFilterSize << ",mStride:" << mStride << std::endl;
//        std::cout << "InputDimention:" << inputs->d[0] << "," << inputs->d[1] << "," <<  inputs->d[2] << std::endl;
//        std::cout << "getOutputDimensions:" << dimsOutput.d[0] << "," << dimsOutput.d[1] << "," <<  dimsOutput.d[2] << std::endl;
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
    mInputHeight = inputDims[0].d[1];
    mInputWidth = inputDims[0].d[2];
    mDataType = type;
}

int Adder2dPlugin::initialize() {
    convertAndCopyToDeivce(mDeviceWeightPtr, mWeights, mDataType);
    return 0;
}

void Adder2dPlugin::terminate() {
    if (mWeights.values)
    {
        free(const_cast<void *>(mWeights.values));
        mWeights.values = nullptr;
    }
    if (mDeviceWeightPtr)
    {
        cudaFree(mDeviceWeightPtr);
        mDeviceWeightPtr = nullptr;
    }
}

size_t Adder2dPlugin::getWorkspaceSize(int maxBatchSize) const{
    return 0;
}

int Adder2dPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream)
{
    switch(mDataType)
    {
        case nvinfer1::DataType::kFLOAT:
            CUDA_CHECK(ForwardGpu<float>(mNbFilters, mNbInputChannels, mInputHeight, mInputWidth, mFilterSize, mStride,
                                         mPadding, (const float *)inputs[0], (float *)outputs[0],
                                         (const float *)mDeviceWeightPtr, stream));
            break;
//        case nvinfer1::DataType::kHALF:
//            CUDA_CHECK(ForwardGpu<__half>(mNbFilters, mNbInputChannels, mInputHeight, mInputWidth, mFilterSize, mStride,
//                                          mPadding, (const __half *)inputs[0], (__half *)outputs[0],
//                                          (const __half *)mDeviceWeightPtr, stream));
//            break;
//        case nvinfer1::DataType::kINT8:
//            CUDA_CHECK(ForwardGpu<u_int8_t>(mNbFilters, mNbInputChannels, mInputHeight, mInputWidth, mFilterSize, mStride,
//                                            mPadding, (const u_int8_t *)inputs[0], (u_int8_t *)outputs[0],
//                                            (const u_int8_t *)mDeviceWeightPtr, stream));
//            break;
        default:
            std::cerr << "error data type" << std::endl;
            ASSERT(false);
    }
    return 0;
}

const char *Adder2dPlugin::getPluginType() const {
    return G_ADDER2D_TYPE;
}

const char *Adder2dPlugin::getPluginVersion() const {
    return G_PLUGIN_VERSION;
}

void Adder2dPlugin::setPluginNamespace(const char* pluginNamespace) {

}

const char* Adder2dPlugin::getPluginNamespace() const {
    return G_PLUGIN_NAMESPACE;
}

void Adder2dPlugin::destroy() {
    delete this;
}

nvinfer1::IPluginV2* Adder2dPlugin::clone() const {
    return new Adder2dPlugin(&mWeights, mNbWeights, mNbInputChannels, mInputHeight, mInputWidth,
                             mFilterSize, mNbFilters, mStride, mPadding);
}



// Adder2dPluginCreator Implementation
Adder2dPluginCreator::Adder2dPluginCreator()  {
    mPluginAttributes.emplace_back(nvinfer1::PluginField("weights", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("nbWeight", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("nbInputChannels", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("inputHeight", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("inputWeight", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("filterSize", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("nbFilters", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("stride", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("padding", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* Adder2dPluginCreator::getPluginName() const {
    return G_ADDER2D_NAME;
}

const char* Adder2dPluginCreator::getPluginVersion() const {
    return G_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* Adder2dPluginCreator::getFieldNames() {
    return &mFC;
}

nvinfer1::IPluginV2* Adder2dPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) {
    int nbWeights, filterSize, nbInputChannels, inputHeight, inputWidth, nbFilters, stride, padding;
    std::vector<float> weightValues;
    const nvinfer1::PluginField* fields = fc->fields;

//    std::cout << "FieldType:kFlOAT32 - " << int(nvinfer1::PluginFieldType::kFLOAT32) << std::endl;
//    std::cout << "FieldType:kINT32 - " << int(nvinfer1::PluginFieldType::kINT32) << std::endl;
    for (int i=0; i<fc->nbFields; i++) {
        const char* attrName = fields[i].name;
//        std::cout << "FieldName:" << attrName << std::endl;
//        std::cout << "FieldType:" << int(fields[i].type) << std::endl;
//        std::cout << "FieldLength:" << int(fields[i].length) << std::endl;

        if(strcmp(attrName, "weights") == 0) {
            ASSERT(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < fields[i].length; j++)
            {
                weightValues.push_back(w[j]);
            }

            for (int j = 0; j < 10; j++)
            {
                std::cout << weightValues[j] << ",";
            }
            std::cout << std::endl;

        }
        if(strcmp(attrName, "nbWeights") == 0) {
            ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
            nbWeights = *(static_cast<const int*>(fields[i].data));
//            std::cout  << "nbWeights:" << nbWeights << std::endl;
        }
        if(strcmp(attrName, "filterSize") == 0) {
            ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
            filterSize = *(static_cast<const int*>(fields[i].data));
//            std::cout  << "filterSize:" << filterSize << std::endl;
        }
        if(strcmp(attrName, "nbInputChannels") == 0) {
            ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
            nbInputChannels = *(static_cast<const int*>(fields[i].data));
//            std::cout  << "nbInputChannels:" << nbInputChannels << std::endl;
        }
        if(strcmp(attrName, "inputHeight") == 0) {
            ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
            inputHeight = *(static_cast<const int*>(fields[i].data));
//            std::cout  << "inputHeight:" << inputHeight << std::endl;
        }
        if(strcmp(attrName, "inputWeight") == 0) {
            ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
            inputWidth = *(static_cast<const int*>(fields[i].data));
//            std::cout  << "inputWidth:" << inputWidth << std::endl;
        }
        if(strcmp(attrName, "nbFilters") == 0) {
            ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
            nbFilters = *(static_cast<const int*>(fields[i].data));
//            std::cout  << "nbFilters:" << nbFilters << std::endl;
        }
        if(strcmp(attrName, "stride") == 0) {
            ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
            stride = *(static_cast<const int*>(fields[i].data));
//            std::cout  << "stride:" << stride << std::endl;
        }
        if(strcmp(attrName, "padding") == 0) {
            ASSERT(fields[i].type == nvinfer1::PluginFieldType::kINT32);
            padding = *(static_cast<const int*>(fields[i].data));
//            std::cout  << "padding:" << padding << std::endl;
        }
    }
    nvinfer1::Weights weights{nvinfer1::DataType::kFLOAT, weightValues.data(), (int64_t)weightValues.size()};
    return new Adder2dPlugin(&weights, nbWeights, nbInputChannels, inputHeight, inputWidth,
                             filterSize, nbFilters, stride, padding);
}

// deserialization plugin implementation
nvinfer1::IPluginV2* Adder2dPluginCreator::deserializePlugin(const char *layerName, const void *serialData, size_t serialLength) {
    return new Adder2dPlugin(serialData, serialLength);
}

const char* Adder2dPluginCreator::getPluginNamespace() const {
    return G_PLUGIN_NAMESPACE;
}

REGISTER_TENSORRT_PLUGIN(Adder2dPluginCreator);
