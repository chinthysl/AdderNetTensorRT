from PIL import Image
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

import torch
import tensorrt as trt

import sys
sys.path.append("../../")
import common
from addermnist import MnistModel

sys.path.append("../build")
from adder2dpytrt import Adder2dPlugin

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list


def get_adder2d_plugin(weights, nbWeights, filterSize, nbFilters, stride, padding):
    plugin = None
    plugin_name = "Adder2d_TRT"
    for plugin_creator in PLUGIN_CREATORS:
        print(plugin_creator.name)
        if plugin_creator.name == plugin_name:
            x = np.ascontiguousarray(np.ravel(weights),dtype=np.float32)
            print(x[0:10])
            weight_field = trt.PluginField("weights", np.array(weights.flatten(), dtype=np.float32), trt.PluginFieldType.FLOAT32)
            nbWeights_field = trt.PluginField("nbWeights", np.array([nbWeights], dtype=np.int32), trt.PluginFieldType.INT32)
            filterSize_field = trt.PluginField("filterSize", np.array([filterSize], dtype=np.int32), trt.PluginFieldType.INT32)
            nbFilters_field = trt.PluginField("nbFilters", np.array([nbFilters], dtype=np.int32), trt.PluginFieldType.INT32)
            stride_field = trt.PluginField("stride", np.array([stride], dtype=np.int32), trt.PluginFieldType.INT32)
            padding_field = trt.PluginField("padding", np.array([padding], dtype=np.int32), trt.PluginFieldType.INT32)

            field_collection = trt.PluginFieldCollection([weight_field, nbWeights_field, filterSize_field,
                                                          nbFilters_field, stride_field, padding_field])
            plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
    return plugin


class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32


def populate_network(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    conv1_w = weights['conv1.weight'].numpy()
    conv1_b = weights['conv1.bias'].numpy()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=20, kernel_shape=(5, 5), kernel=conv1_w, bias=conv1_b)
    conv1.stride = (1, 1)

    pool1 = network.add_pooling(input=conv1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool1.stride = (2, 2)

    # Creating and adding the Adder2d layer plugin
    adder1_w = weights['adder.adder'].numpy()
    adder1_w = adder1_w.astype('float64')
    print(adder1_w.shape)
    print(adder1_w[0, 0, 0, :], adder1_w[0, 0, 1, :])
    print(pool1.get_output(0).shape, pool1.get_output(0).location, "{0:b}".format(pool1.get_output(0).allowed_formats))
    print(pool1.get_output(0))
    adder_plugin = get_adder2d_plugin(weights=adder1_w, nbWeights=adder1_w.size, filterSize=5, nbFilters=50, stride=1, padding=0)
    print(adder_plugin)
    adder1 = network.add_plugin_v2(inputs=[pool1.get_output(0)], plugin=adder_plugin)
    # ********************************************

    pool2 = network.add_pooling(adder1.get_output(0), trt.PoolingType.MAX, (2, 2))
    pool2.stride = (2, 2)

    fc1_w = weights['fc1.weight'].numpy()
    fc1_b = weights['fc1.bias'].numpy()
    fc1 = network.add_fully_connected(input=pool2.get_output(0), num_outputs=500, kernel=fc1_w, bias=fc1_b)

    relu1 = network.add_activation(input=fc1.get_output(0), type=trt.ActivationType.RELU)

    fc2_w = weights['fc2.weight'].numpy()
    fc2_b = weights['fc2.bias'].numpy()
    fc2 = network.add_fully_connected(input=relu1.get_output(0), num_outputs=ModelData.OUTPUT_SIZE, kernel=fc2_w, bias=fc2_b)

    fc2.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=fc2.get_output(0))


def build_engine(weights):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_workspace_size = common.GiB(1)
        # Populate the network using weights from the PyTorch model.
        populate_network(network, weights)
        # Build and return an engine.
        return builder.build_cuda_engine(network)


# Loads a random test case from pytorch's DataLoader
def load_random_test_case(model, pagelocked_buffer):
    # Select an image at random to be the test case.
    img, expected_output = model.get_random_testcase()
    # Copy to the pagelocked input buffer
    np.copyto(pagelocked_buffer, img)
    return expected_output


def main():
    mnist_model = MnistModel()
    # mnist_model.learn()
    mnist_model.network.load_state_dict(torch.load('adder_mnist.pth'))
    weights = mnist_model.get_weights()

    # Do inference with TensorRT.
    with build_engine(weights) as engine:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            case_num = load_random_test_case(mnist_model, pagelocked_buffer=inputs[0].host)
            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            pred = np.argmax(output)
            print("Test Case: " + str(case_num))
            print("Prediction: " + str(pred))


if __name__ == '__main__':
    main()
