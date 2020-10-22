import resnet20
from PIL import Image
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

import torch
import tensorrt as trt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (3, 32, 32)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32


def populate_BasicBlock(network, weights, input_tensor, layer_id, block_id, inplanes, planes, stride=1, downsample = False):

    conv1_w = weights['layer' + str(layer_id) + '.' + str(block_id) + 'conv1.weight'].numpy()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=planes, kernel_shape=(3, 3), kernel=conv1_w)
    conv1.stride = (1, 1)

    network.add_activation(input=conv1.get_output(0), type=trt.ActivationType.RELU)
    output._trt = layer.get_output(0)


def populate_network(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    conv1_w = weights['conv1.weight'].numpy()
    conv1_b = weights['conv1.bias'].numpy()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=16, kernel_shape=(3, 3), kernel=conv1_w, bias=conv1_b)
    conv1.stride = (1, 1)

    pool1 = network.add_pooling(input=conv1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool1.stride = (2, 2)

    conv2_w = weights['conv2.weight'].numpy()
    conv2_b = weights['conv2.bias'].numpy()
    conv2 = network.add_convolution(pool1.get_output(0), 50, (5, 5), conv2_w, conv2_b)
    conv2.stride = (1, 1)

    pool2 = network.add_pooling(conv2.get_output(0), trt.PoolingType.MAX, (2, 2))
    pool2.stride = (2, 2)

    fc1_w = weights['fc1.weight'].numpy()
    fc1_b = weights['fc1.bias'].numpy()
    fc1 = network.add_fully_connected(input=pool2.get_output(0), num_outputs=500, kernel=fc1_w, bias=fc1_b)

    relu1 = network.add_activation(input=fc1.get_output(0), type=trt.ActivationType.RELU)

    fc2_w = weights['fc2.weight'].numpy()
    fc2_b = weights['fc2.bias'].numpy()
    fc2 = network.add_fully_connected(relu1.get_output(0), ModelData.OUTPUT_SIZE, fc2_w, fc2_b)

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
    model = resnet20.resnet20()
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('saved_models/ResNet20-AdderNet.pth'))
    weights = model.network.state_dict()
    print(weights)


    # mnist_model = model.MnistModel()
    # mnist_model.learn()
    # weights = mnist_model.get_weights()
    # # Do inference with TensorRT.
    # with build_engine(weights) as engine:
    #     # Build an engine, allocate buffers and create a stream.
    #     # For more information on buffer allocation, refer to the introductory samples.
    #     inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    #     with engine.create_execution_context() as context:
    #         case_num = load_random_test_case(mnist_model, pagelocked_buffer=inputs[0].host)
    #         # For more information on performing inference, refer to the introductory samples.
    #         # The common.do_inference function will return a list of outputs - we only have one in this case.
    #         [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    #         pred = np.argmax(output)
    #         print("Test Case: " + str(case_num))
    #         print("Prediction: " + str(pred))

if __name__ == '__main__':
    main()
