import torch
import onnx

from addernet_mnist import MnistModel
OPSET = 12


model = MnistModel()
model.network.load_state_dict(torch.load('./saved_models/addernet_mnist.pth'))
model.network.to('cuda')

dummy_input = torch.randn(1, 1, 28, 28, device='cuda')

# convert pytorch model to onnx format
torch.onnx.export(model.network, dummy_input, "./saved_models/addernet_mnist.onnx", verbose=True, opset_version=OPSET)

# Load the ONNX model
model = onnx.load("./saved_models/addernet_mnist.onnx")
# Check that the IR is well formed
onnx.checker.check_model(model)
# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)
