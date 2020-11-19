import sys
sys.path.append("../plugin/build")
from adder2dpytrt import Adder2dPlugin

import tensorrt as trt
import numpy as np


TRT_LOGGER = trt.Logger()

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list


def get_adder2d_plugin():
    plugin = None
    plugin_name = "Adder2d_TRT"
    for plugin_creator in PLUGIN_CREATORS:
        print(plugin_creator.name )
        if plugin_creator.name == plugin_name:
            weight_field = trt.PluginField("weights", np.ones((16*3*3*16,), dtype=np.float32), trt.PluginFieldType.FLOAT32)
            nbWeights_field = trt.PluginField("nbWeights", np.array([16*3*3*16], dtype=np.int32), trt.PluginFieldType.INT32)
            nbInputChannels_field = trt.PluginField("nbInputChannels", np.array([16], dtype=np.int32), trt.PluginFieldType.INT32)
            nInputHeight_field = trt.PluginField("nInputHeight", np.array([32], dtype=np.int32), trt.PluginFieldType.INT32)
            nInputWidth_field = trt.PluginField("nInputWidth", np.array([32], dtype=np.int32), trt.PluginFieldType.INT32)
            filterSize_field = trt.PluginField("filterSize", np.array([3], dtype=np.int32), trt.PluginFieldType.INT32)
            nbFilters_field = trt.PluginField("nbfilter", np.array([16], dtype=np.int32), trt.PluginFieldType.INT32)
            stride_field = trt.PluginField("stride", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32)
            padding_field = trt.PluginField("padding", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32)

            field_collection = trt.PluginFieldCollection([weight_field, nbWeights_field, filterSize_field,
                                                          nbInputChannels_field, nInputHeight_field, nInputWidth_field,
                                                          nbFilters_field, stride_field, padding_field])
            plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
    return plugin


def test_plugin():
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_workspace_size = 2**20
        input_tensor = network.add_input(name="input_data", dtype=trt.float32, shape=(3, 32, 32))
        adder_plugin = get_adder2d_plugin()
        print(adder_plugin)
        adder_layer = network.add_plugin_v2(inputs=[input_tensor], plugin=adder_plugin)
        adder_layer.get_output(0).name = "outputs"
        network.mark_output(adder_layer.get_output(0))


if __name__ == "__main__":
    test_plugin()
