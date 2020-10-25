from PIL import Image
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

import torch
import tensorrt as trt
import torch.nn.functional as F

import sys
sys.path.append("../../")
import common
from addermnist_v2 import MnistModel

sys.path.append("../build")
from adder2dpytrt import Adder2dPlugin

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list


def get_adder2d_plugin(weights, nbWeights, nbInCh, nInH, nInW, filterSize, nbFilters, stride, padding):
    plugin = None
    plugin_name = "Adder2d_TRT"
    for plugin_creator in PLUGIN_CREATORS:
        print(plugin_creator.name)
        if plugin_creator.name == plugin_name:
            x = np.ascontiguousarray(np.ravel(weights),dtype=np.float32)
            print(x[0:10])
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
    INPUT_NAME = "data"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32


def populate_network(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)
    print("input shape:", input_tensor.shape, "input formats:{0:b}".format(input_tensor.allowed_formats))
    input_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
    print("input shape after:", input_tensor.shape, "pool1 formats:{0:b}".format(input_tensor.allowed_formats))

    # Creating and adding the Adder1 layer
    adder1_w = weights['adder1.adder'].numpy()
    adder1_w = adder1_w.astype('float32')
    print(adder1_w.shape)
    print(adder1_w[0, 0, 0, :], adder1_w[0, 0, 1, :])
    in_ch = input_tensor.shape[0]
    in_h = input_tensor.shape[1]
    in_w = input_tensor.shape[2]
    adder1_plugin = get_adder2d_plugin(weights=adder1_w, nbWeights=adder1_w.size, nbInCh=in_ch, nInH=in_h, nInW=in_w,
                                       filterSize=5, nbFilters=20, stride=1, padding=0)
    adder1 = network.add_plugin_v2(inputs=[input_tensor], plugin=adder1_plugin)
    # ********************************************

    pool1 = network.add_pooling(input=adder1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool1.stride = (2, 2)
    print("pool1 shape:", pool1.get_output(0).shape, "pool1 formats:{0:b}".format(pool1.get_output(0).allowed_formats))
    pool1.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
    print("pool1 shape after:", pool1.get_output(0).shape, "pool1 formats:{0:b}".format(pool1.get_output(0).allowed_formats))

    # Creating and adding the Adder2 layer
    adder2_w = weights['adder2.adder'].numpy()
    adder2_w = adder2_w.astype('float32')
    print(adder2_w.shape)
    print(adder2_w[0, 0, 0, :], adder1_w[0, 0, 1, :])
    in_ch = pool1.get_output(0).shape[0]
    in_h = pool1.get_output(0).shape[1]
    in_w = pool1.get_output(0).shape[2]
    adder2_plugin = get_adder2d_plugin(weights=adder2_w, nbWeights=adder2_w.size, nbInCh=in_ch, nInH=in_h, nInW=in_w,
                                       filterSize=5, nbFilters=50, stride=1, padding=0)
    adder2 = network.add_plugin_v2(inputs=[pool1.get_output(0)], plugin=adder2_plugin)
    # ********************************************

    pool2 = network.add_pooling(input=adder2.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool2.stride = (2, 2)
    print("pool2 shape:", pool2.get_output(0).shape, "pool2 formats:{0:b}".format(pool2.get_output(0).allowed_formats))
    pool2.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
    print("pool2 shape after:", pool2.get_output(0).shape, "pool2 formats:{0:b}".format(pool2.get_output(0).allowed_formats))

    fc1_w = weights['fc1.weight'].numpy()
    fc1_b = weights['fc1.bias'].numpy()
    fc1 = network.add_fully_connected(input=pool2.get_output(0), num_outputs=500, kernel=fc1_w, bias=fc1_b)

    relu1 = network.add_activation(input=fc1.get_output(0), type=trt.ActivationType.TANH)

    fc2_w = weights['fc2.weight'].numpy()
    fc2_b = weights['fc2.bias'].numpy()
    fc2 = network.add_fully_connected(input=relu1.get_output(0), num_outputs=ModelData.OUTPUT_SIZE, kernel=fc2_w, bias=fc2_b)

    softmax_1 = network.add_softmax(input=fc2.get_output(0))

    softmax_1.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=softmax_1.get_output(0))
    # network.mark_output(tensor=adder1.get_output(0))  # for debugging


def build_engine(weights):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_workspace_size = common.GiB(1)
        # Populate the network using weights from the PyTorch model.
        populate_network(network, weights)
        # setting the tensor format for network
        # network.get_input(0).allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        # network.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        # Build and return an engine.
        return builder.build_cuda_engine(network)


# Loads a random test case from pytorch's DataLoader
def load_random_test_case(model, pagelocked_buffer):
    # Select an image at random to be the test case.
    img, expected_output = model.get_random_testcase()
    # Copy to the pagelocked input buffer
    np.copyto(pagelocked_buffer, img)
    return expected_output


def load_single_test_case(pagelocked_buffer):
    img = np.load('test_img.npy')
    label = np.load('test_label.npy')
    print('img shape:', img.shape, 'label shape:', label.shape)
    # Copy to the pagelocked input buffer
    np.copyto(pagelocked_buffer, img)
    return label


def print_feature_map(feature_array, c, h, w):
    for i in range(c):
        print('[', end = '')
        for j in range(h):
            print('[', end='')
            for k in range(w):
                print(feature_array[i*h*w + j*w + k], end=',')
            print(']')
        print(']')


def single_image_inference():
    mnist_model = MnistModel()
    # mnist_model.learn()
    mnist_model.network.load_state_dict(torch.load('adder_mnist_v2.pth'))
    weights = mnist_model.get_weights()

    # Do inference with TensorRT.
    with build_engine(weights) as engine:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            case_num = load_single_test_case(pagelocked_buffer=inputs[0].host)
            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            pred = np.argmax(output)
            # log_softmax = F.log_softmax(torch.from_numpy(output), dim=0)
            print("Actual label: " + str(case_num))
            print("Pred label : " + str(pred))
            print("Pred Vec:", output)
            # print("Pred log softmax:", log_softmax)


def random_image_inference():
    mnist_model = MnistModel()
    # mnist_model.learn()
    mnist_model.network.load_state_dict(torch.load('adder_mnist_v2.pth'))
    weights = mnist_model.get_weights()

    # Do inference with TensorRT.
    with build_engine(weights) as engine:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            for i in range(20):
                case_num = load_random_test_case(mnist_model, pagelocked_buffer=inputs[0].host)
                # For more information on performing inference, refer to the introductory samples.
                # The common.do_inference function will return a list of outputs - we only have one in this case.
                [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                pred = np.argmax(output)
                print("Actual label: " + str(case_num))
                print("Pred label : " + str(pred))


def main():
    # random_image_inference()
    single_image_inference()


if __name__ == '__main__':
    main()
