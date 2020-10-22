import torch
import resnet20
import resnet50
from tqdm import tqdm
import time

# cuDnn configurations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

model = resnet20.resnet20()
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load('ResNet20-AdderNet.pth'))

print("AdderNet-ResNet20 Speed testing... ...")
random_input = torch.randn(1, 3, 32, 32, device='cuda')
model.eval()

time_list = []
for i in tqdm(range(10001)):
    torch.cuda.synchronize()
    tic = time.time()
    model(random_input)
    torch.cuda.synchronize()
    # the first iteration time cost much higher, so exclude the first iteration
    # print(time.time()-tic)
    time_list.append(time.time() - tic)
time_list = time_list[1:]
print("     + Done 10000 iterations inference !")
print("     + Total time cost: {}s".format(sum(time_list)))
print("     + Average time cost: {}s".format(sum(time_list) / 10000))
print("     + Frame Per Second: {:.2f}".format(1 / (sum(time_list) / 10000)))


model = resnet50.resnet50()
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load('ResNet50-AdderNet.pth'))

print("AdderNet-ResNet50 Speed testing... ...")
random_input = torch.randn(1, 3, 224, 224, device='cuda')
model.eval()

time_list = []
for i in tqdm(range(10001)):
    torch.cuda.synchronize()
    tic = time.time()
    model(random_input)
    torch.cuda.synchronize()
    # the first iteration time cost much higher, so exclude the first iteration
    # print(time.time()-tic)
    time_list.append(time.time() - tic)
time_list = time_list[1:]
print("     + Done 10000 iterations inference !")
print("     + Total time cost: {}s".format(sum(time_list)))
print("     + Average time cost: {}s".format(sum(time_list) / 10000))
print("     + Frame Per Second: {:.2f}".format(1 / (sum(time_list) / 10000)))
