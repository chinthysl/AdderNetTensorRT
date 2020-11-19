import pycuda.driver as cuda
import pycuda.autoinit

import torch
import tensorrt as trt

import numpy as np
import matplotlib.pyplot as plt
import sys

import common
from addernet_mnist import MnistModel

sys.path.append("./plugin/build/")
from adder2dpytrt import Adder2dPlugin

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list


def get_adder2d_plugin(weights, nbWeights, nbInCh, nInH, nInW, filterSize, nbFilters, stride, padding):
    plugin = None
    plugin_name = "Adder2d_TRT"
    for plugin_creator in PLUGIN_CREATORS:
        # print(plugin_creator.name)
        if plugin_creator.name == plugin_name:
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
    INPUT_NAME = "input_img"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "output_prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32


def populate_network(network, weights):
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    # ************AdderLayer1**********************
    adder1_w = weights['adder1.adder'].numpy()
    adder1_w = adder1_w.astype('float32')

    in_ch = input_tensor.shape[0]
    in_h = input_tensor.shape[1]
    in_w = input_tensor.shape[2]
    adder1_plugin = get_adder2d_plugin(weights=adder1_w, nbWeights=adder1_w.size, nbInCh=in_ch, nInH=in_h, nInW=in_w,
                                       filterSize=5, nbFilters=20, stride=1, padding=0)
    adder1 = network.add_plugin_v2(inputs=[input_tensor], plugin=adder1_plugin)


    # **********BatchNormLayer1**********************
    weight = weights['bn1.weight'].numpy()
    bias = weights['bn1.bias'].numpy()
    running_mean = weights['bn1.running_mean'].numpy()
    running_var = weights['bn1.running_var'].numpy()
    eps = 1e-5

    scale = weight / np.sqrt(running_var + eps)
    bias = bias - running_mean * scale
    power = np.ones_like(scale)
    bn1 = network.add_scale(adder1.get_output(0), trt.ScaleMode.CHANNEL, bias, scale, power)


    # **********MaxPoolLayer1**********************
    pool1 = network.add_pooling(input=bn1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool1.stride = (2, 2)


    # ************AdderLayer2**********************
    adder2_w = weights['adder2.adder'].numpy()
    adder2_w = adder2_w.astype('float32')

    in_ch = pool1.get_output(0).shape[0]
    in_h = pool1.get_output(0).shape[1]
    in_w = pool1.get_output(0).shape[2]
    adder2_plugin = get_adder2d_plugin(weights=adder2_w, nbWeights=adder2_w.size, nbInCh=in_ch, nInH=in_h, nInW=in_w,
                                       filterSize=5, nbFilters=50, stride=1, padding=0)
    adder2 = network.add_plugin_v2(inputs=[pool1.get_output(0)], plugin=adder2_plugin)


    # **********BatchNormLayer2**********************
    weight = weights['bn2.weight'].numpy()
    bias = weights['bn2.bias'].numpy()
    running_mean = weights['bn2.running_mean'].numpy()
    running_var = weights['bn2.running_var'].numpy()
    eps = 1e-5

    scale = weight / np.sqrt(running_var + eps)
    bias = bias - running_mean * scale
    power = np.ones_like(scale)
    bn2 = network.add_scale(adder2.get_output(0), trt.ScaleMode.CHANNEL, bias, scale, power)


    # **********MaxPoolLayer2**********************
    pool2 = network.add_pooling(input=bn2.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool2.stride = (2, 2)


    # **********FCLayer1***************************
    fc1_w = weights['fc1.weight'].numpy()
    fc1_b = weights['fc1.bias'].numpy()

    fc1 = network.add_fully_connected(input=pool2.get_output(0), num_outputs=500, kernel=fc1_w, bias=fc1_b)


    # **********ReluLayer1***************************
    relu1 = network.add_activation(input=fc1.get_output(0), type=trt.ActivationType.TANH)


    # **********FCLayer2***************************
    fc2_w = weights['fc2.weight'].numpy()
    fc2_b = weights['fc2.bias'].numpy()
    fc2 = network.add_fully_connected(input=relu1.get_output(0), num_outputs=ModelData.OUTPUT_SIZE, kernel=fc2_w, bias=fc2_b)


    # **********SoftMaxLayer1***************************
    softmax_1 = network.add_softmax(input=fc2.get_output(0))


    softmax_1.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=softmax_1.get_output(0))
    # network.mark_output(tensor=adder1.get_output(0))  # for debugging


def build_engine(weights, max_batch_size):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_workspace_size = common.GiB(1)
        builder.max_batch_size = max_batch_size
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


def load_single_test_case(pagelocked_buffer):
    img = np.load('./rand_iamge/test_img.npy')
    label = np.load('./rand_iamge/test_label.npy')
    print('img shape:', img.shape, 'label shape:', label.shape)
    # Copy to the pagelocked input buffer
    np.copyto(pagelocked_buffer, img)
    return label


def single_image_inference():
    mnist_model = MnistModel()
    mnist_model.network.load_state_dict(torch.load('./saved_models/addernet_mnist.pth'))
    weights = mnist_model.get_weights()

    # Do inference with TensorRT.
    with build_engine(weights, 1) as engine:
        # Build an engine, allocate buffers and create a stream.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            case_num = load_single_test_case(pagelocked_buffer=inputs[0].host)
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            pred = np.argmax(output)
            print("Actual label: " + str(case_num))
            print("Pred label : " + str(pred))
            print("Pred Vec:", output)


def random_image_inference():
    mnist_model = MnistModel()
    mnist_model.network.load_state_dict(torch.load('addernet_mnist.pth'))
    weights = mnist_model.get_weights()

    # Do inference with TensorRT.
    with build_engine(weights, 1) as engine:
        # Build an engine, allocate buffers and create a stream.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            for i in range(100):
                case_num = load_random_test_case(mnist_model, pagelocked_buffer=inputs[0].host)
                # The common.do_inference function will return a list of outputs - we only have one in this case.
                [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                pred = np.argmax(output)
                print("Actual label: " + str(case_num))
                print("Pred label : " + str(pred))


def calc_accuracy():
    mnist_model = MnistModel()
    mnist_model.network.load_state_dict(torch.load('./saved_models/addernet_mnist.pth'))
    weights = mnist_model.get_weights()

    with build_engine(weights, 1) as engine:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            correct = 0
            for data, target in mnist_model.test_loader:
                for case_num in range(len(data)):
                    test_img = data.numpy()[case_num].ravel().astype(np.float32)
                    true_label = target.numpy()[case_num]

                    np.copyto(inputs[0].host, test_img)
                    [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                    pred_label = np.argmax(output)
                    if true_label==pred_label:
                        correct += 1
            print('Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(mnist_model.test_loader.dataset),
                                                       100. * correct / len(mnist_model.test_loader.dataset)))


def calc_latency():
    # **********Create the TRT Model and do Infernce********************
    mnist_model = MnistModel()
    mnist_model.network.load_state_dict(torch.load('./saved_models/addernet_mnist.pth'))
    weights = mnist_model.get_weights()

    with build_engine(weights, 128) as engine:
        max_batch_size = 64
        mean_latency_list = []
        std_latency_list = []
        batch_size_list = []

        for batch_size in range(1, max_batch_size+1):
            batch_size_list.append(batch_size)
            dummy_input = np.random.rand(batch_size, 1, 28, 28).ravel().astype(np.float32)
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            inputs[0].host = dummy_input

            with engine.create_execution_context() as context:
                # GPU-WARM-UP
                for _ in range(100):
                    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
                    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
                    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
                # MEASURE PERFORMANCE
                repetitions = 1000
                timings = np.zeros((repetitions, 1))
                for rep in range(repetitions):
                    start = cuda.Event()
                    end = cuda.Event()

                    start.record()
                    # start.synchronize()
                    # ************bottleneck*************
                    # context.profiler = trt.Profiler()
                    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
                    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
                    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
                    # **********bottleneck***************
                    end.record()
                    end.synchronize()
                    stream.synchronize()

                    mili_secs = start.time_till(end)
                    timings[rep] = mili_secs

                mean_syn = np.sum(timings) / repetitions
                std_syn = np.std(timings)
                mean_latency_list.append(mean_syn)
                std_latency_list.append(std_syn)

            print("Latency calculated for batch size: {}".format(batch_size), end="")
            print('\r', end='')

    plt.figure(figsize=(10, 5))
    plt.plot(mean_latency_list, label='mean')
    # plt.plot(std_latency_list, label='std')
    plt.xlabel('Input Batch Size')
    plt.ylabel('Mean Latency (ms)')
    plt.title('Inference Latency - AdderNetMnist TensorRT Model')
    plt.savefig('figures/tensorrt_latency.jpg')

    print('Inference Latency graph saved for multiple batch sizes')
    print('Single Image inference Latency: {}ms'.format(mean_latency_list[0]))


def main():
    # single_image_inference()
    # random_image_inference()
    calc_accuracy()
    calc_latency()


if __name__ == '__main__':
    main()
