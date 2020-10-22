import torch
from torchsummary import summary

import onnx
import resnet20
import resnet50
OPSET = 12


model = resnet20.resnet20()
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load('saved_models/ResNet20-AdderNet.pth'))
print(model)
# summary(model, (3, 224, 224))

dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
torch.onnx.export(model.module, dummy_input, "resnet50.onnx", verbose=True, opset_version=OPSET)


# Load the ONNX model
model = onnx.load("resnet50.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)
