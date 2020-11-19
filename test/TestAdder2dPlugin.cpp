#include "Adder2dPlugin.h"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {

    int nbWeights, nbInputChannels, nInputHeight, nInputWidth, filterSize, nbFilters, stride, padding;
    nbInputChannels=4; nInputHeight=32; nInputWidth=32;
    filterSize=3; nbFilters=64; stride=1; padding=0;
    nbWeights = nbInputChannels*filterSize*filterSize*nbFilters;
    std::vector<float> weightValues;
    for(int i=0; i<nbWeights; i++){
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        weightValues.push_back(r);
    }
    nvinfer1::Weights weights{nvinfer1::DataType::kFLOAT, weightValues.data(), (int64_t)weightValues.size()};

    auto plugin_obj = new Adder2dPlugin(&weights, nbWeights, nbInputChannels, nInputHeight, nInputWidth,
                                        filterSize, nbFilters, stride, padding);
    std::cout << "Adder2dPlugin Obj Created" << std::endl;

    delete plugin_obj;

    std::cout << "Adder2dPlugin Obj Deleted" << std::endl;

    std::cout << "Creating Adder2dPlugin using c++ is successful" << std::endl;

    return 0;
}