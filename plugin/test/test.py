import sys
sys.path.append("../build")

import tensorrt as trt
import numpy as np

TRT_LOGGER = trt.Logger()

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list


def get_adder2d_plugin():
    plugin = None
    plugin_name = "Adder2d_TRT"
    for plugin_creator in PLUGIN_CREATORS:
        if plugin_creator.name == plugin_name:
            weight_field = trt.PluginField("weights", np.ones((128,), dtype=np.float32), trt.PluginFieldType.kFLOAT32)
            nbWeights_field = trt.PluginField("nbWeights", 128, trt.PluginFieldType.kINT32)
            filterSize_field = trt.PluginField("filterSize", 3, trt.PluginFieldType.kINT32)
            nbFilters_field = trt.PluginField("nbFilters", 64, trt.PluginFieldType.kINT32)
            stride_field = trt.PluginField("stride", 1, trt.PluginFieldType.kINT32)
            padding_field = trt.PluginField("padding", 0, trt.PluginFieldType.kINT32)

            field_collection = trt.PluginFieldCollection([weight_field, nbWeights_field, filterSize_field,
                                                          nbFilters_field, stride_field, padding_field])
            plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
    return plugin


def test_plugin():
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_workspace_size = 2**20
        input_tensor = network.add_input(name="input_data", dtype=trt.float32, shape=(3, 32, 32))
        adder_layer = network.add_plugin_v2(inputs=input_tensor, plugin=get_adder2d_plugin())
        adder_layer.get_output(0).name = "outputs"
        network.mark_output(adder_layer.get_output(0))


def has_adder2d_plugin():
    try:
        from adder2dpytrt import Adder2dPlugin
        print('Adder2dPlugin loaded from dynamic lib')
        return True
    except Exception as e:
        print(e)
        return False


def get_adder2d_plugin_v2(size, mode, align_corners):
    if(has_adder2d_plugin()):
        PLUGIN_NAME = 'Adder2d_TRT'
        registry = trt.get_plugin_registry()
        creator = [c for c in registry.plugin_creator_list if c.name == PLUGIN_NAME and c.plugin_namespace == '_TRT'][0]
        adder2d_plugin = Adder2dPlugin(size=size, mode=mode, align_corners=align_corners)
        return creator.deserialize_plugin(PLUGIN_NAME, torch2trt_plugin.serializeToString())


if __name__ == "__main__":
    test_plugin()


