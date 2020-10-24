#ifndef ADDER2D_PLUGIN_H
#define ADDER2D_PLUGIN_H

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include <vector>

class Adder2dPlugin : public nvinfer1::IPluginV2
{
public:
    Adder2dPlugin(const nvinfer1::Weights *weights, int nbWeights, int nbInputChannels, int inputHeight,
                  int inputWidth, int filterSize, int nbFilters, int stride, int padding);

    Adder2dPlugin(const void *data, size_t length);

    ~Adder2dPlugin();

    virtual int getNbOutputs() const override;

    virtual nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;

    virtual bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override;

    virtual void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims,
                                     int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format,
                                     int maxBatchSize) override;

    virtual int initialize() override;

    virtual void terminate() override;

    virtual size_t getWorkspaceSize(int maxBatchSize) const override;

    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    virtual size_t getSerializationSize() const override;

    virtual void serialize(void* buffer) const override;

    virtual const char* getPluginType() const override;

    virtual const char* getPluginVersion() const override;

    virtual void destroy();

    virtual IPluginV2* clone() const override;

    virtual void setPluginNamespace(const char* pluginNamespace) override;

    virtual const char* getPluginNamespace() const override;

private:
    int mNbWeights, mNbInputChannels, mInputHeight, mInputWidth, mFilterSize, mNbFilters, mStride, mPadding;
    nvinfer1::Weights mWeights;
    nvinfer1::DataType mDataType{nvinfer1::DataType::kFLOAT};
    void* mDeviceWeightPtr{nullptr};
};


class Adder2dPluginCreator : public nvinfer1::IPluginCreator {
public:
    Adder2dPluginCreator();

    // ------------------inherit from IPluginCreator-------------------
    // return the plugin type + plugin namesapce
    virtual const char* getPluginName() const override;

    // return the plugin version
    virtual const char* getPluginVersion() const override;

    // return a list of fields that needs to be passed to createPlugin
    virtual const nvinfer1::PluginFieldCollection* getFieldNames() override;

    // return nullptr in case of error
    virtual nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection *fc) override;

    // Called during deserialization of plugin layer. Return a plugin object.
    virtual nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLenth) override;

    // Set the namespace of the plugin creator based on the plugin library it belongs to. This can be set while registering the plugin creator
    virtual void setPluginNamespace(const char* pluginNamespace) override {}

    // Return the namespace of the plugin creator object.
    virtual const char* getPluginNamespace() const override;

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
};


#endif //ADDER2D_PLUGIN_H