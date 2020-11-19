from PIL import Image
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

import torch
import tensorrt as trt

import sys
sys.path.append("../")
import common
from test_adder_layer import SingleAdder

sys.path.append("../plugin/build")
from adder2dpytrt import Adder2dPlugin

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list


def get_adder2d_plugin(weights, nbWeights, nbInCh, nInH, nInW, filterSize, nbFilters, stride, padding):
    plugin = None
    plugin_name = "Adder2d_TRT"
    for plugin_creator in PLUGIN_CREATORS:
        if plugin_creator.name == plugin_name:
            x = np.ascontiguousarray(np.ravel(weights),dtype=np.float32)
            weight_field = trt.PluginField("weights", np.array(weights.flatten(), dtype=np.float32), trt.PluginFieldType.FLOAT32)
            nbWeights_field = trt.PluginField("nbWeights", np.array([nbWeights], dtype=np.int32), trt.PluginFieldType.INT32)
            nbInCh_field = trt.PluginField("nbInputChannels", np.array([nbInCh], dtype=np.int32), trt.PluginFieldType.INT32)
            nInH_field = trt.PluginField("nInputHeight", np.array([nInH], dtype=np.int32), trt.PluginFieldType.INT32)
            nInW_field = trt.PluginField("nInputWidth", np.array([nInW], dtype=np.int32), trt.PluginFieldType.INT32)
            filterSize_field = trt.PluginField("filterSize", np.array([filterSize], dtype=np.int32), trt.PluginFieldType.INT32)
            nbFilters_field = trt.PluginField("nbFilters", np.array([nbFilters], dtype=np.int32), trt.PluginFieldType.INT32)
            stride_field = trt.PluginField("stride", np.array([stride], dtype=np.int32), trt.PluginFieldType.INT32)
            padding_field = trt.PluginField("padding", np.array([padding], dtype=np.int32), trt.PluginFieldType.INT32)

            field_collection = trt.PluginFieldCollection([weight_field, nbWeights_field, filterSize_field,
                                                          nbInCh_field, nInH_field, nInW_field,
                                                          nbFilters_field, stride_field, padding_field])
            plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
    return plugin


class ModelData(object):
    INPUT_NAME = "input_fmap"
    INPUT_SHAPE = (5, 5, 5)
    OUTPUT_NAME = "output_fmap"
    DTYPE = trt.float32


def populate_network(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    # Creating and adding the Adder1 layer
    adder1_w = weights['adder1.adder'].numpy()
    adder1_w = adder1_w.astype('float32')
    in_ch = input_tensor.shape[0]
    in_h = input_tensor.shape[1]
    in_w = input_tensor.shape[2]
    adder1_plugin = get_adder2d_plugin(weights=adder1_w, nbWeights=adder1_w.size, nbInCh=in_ch, nInH=in_h, nInW=in_w,
                                       filterSize=3, nbFilters=5, stride=1, padding=0)
    adder1 = network.add_plugin_v2(inputs=[input_tensor], plugin=adder1_plugin)
    # ********************************************

    adder1.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=adder1.get_output(0))  # for debugging


def build_engine(weights):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_workspace_size = common.GiB(1)
        # Populate the network using weights from the PyTorch model.
        populate_network(network, weights)
        # Build and return an engine.
        return builder.build_cuda_engine(network)


def print_feature_map(feature_array, c, h, w):
    for i in range(c):
        print('[', end='')
        for j in range(h):
            print('[', end='')
            for k in range(w):
                print(feature_array[i*h*w + j*w + k], end=',')
            print(']')
        print(']')


def load_single_test_case(pagelocked_buffer):
    # Select an image at random to be the test case.
    input = np.empty([5, 5, 5])
    input.fill(1)
    for i in range(5):
        print(input[i])
    # Copy to the pagelocked input buffer
    input = input.ravel().astype(np.float32)
    np.copyto(pagelocked_buffer, input)


def single_image_inference():
    print("####Testing the output of TensorRT implementation of the Adder Layer###")
    adder_model = SingleAdder()
    weights = adder_model.get_weights()

    # Do inference with TensorRT.
    with build_engine(weights) as engine:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            load_single_test_case(pagelocked_buffer=inputs[0].host)
            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            print("####Output Feature Map####")
            print_feature_map(output, 5, 3, 3)


def main():
    single_image_inference()


if __name__ == '__main__':
    main()
