import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
from random import randint

import adder


# Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.adder1 = adder.adder2d(1, 20, kernel_size=5, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(20)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adder2 = adder.adder2d(20, 50, kernel_size=5, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(800, 500)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.adder1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.adder2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = x.view(-1, 800)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class MnistModel(object):
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 100
        self.learning_rate = 0.0025
        self.sgd_momentum = 0.9
        self.log_interval = 100
        # Fetch MNIST data set.
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data/mnist', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=self.batch_size,
            shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data/mnist', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=self.test_batch_size,
            shuffle=True)
        self.network = Net()
        # print(self.network)
        summary(self.network, (1, 28, 28), device='cpu')

    # Train the network for one or more epochs, validating after each epoch.
    def learn(self, num_epochs=2):
        # Train the network for a single epoch
        def train(epoch):
            self.network.train()
            optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum=self.sgd_momentum)
            for batch, (data, target) in enumerate(self.train_loader):
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = self.network(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch * len(data), len(self.train_loader.dataset), 100. * batch / len(self.train_loader), loss.data.item()))

        # Test the network
        def test(epoch):
            self.network.eval()
            test_loss = 0
            correct = 0
            for data, target in self.test_loader:
                with torch.no_grad():
                    data, target = Variable(data), Variable(target)
                output = self.network(data)
                test_loss += F.nll_loss(output, target).data.item()
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).cpu().sum()
            test_loss /= len(self.test_loader)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(self.test_loader.dataset), 100. * correct / len(self.test_loader.dataset)))

        for e in range(num_epochs):
            train(e + 1)
            test(e + 1)

    def get_weights(self):
        return self.network.state_dict()

    def get_random_testcase(self):
        data, target = next(iter(self.test_loader))
        case_num = randint(0, len(data) - 1)
        test_case = data.numpy()[case_num].ravel().astype(np.float32)
        test_name = target.numpy()[case_num]
        return test_case, test_name


def train_n_save_weights():
    mnist_model = MnistModel()
    mnist_model.learn()
    torch.save(mnist_model.get_weights(), './saved_models/addernet_mnist.pth')
    weights = mnist_model.get_weights()
    print(weights)

    # ********save single test image********
    data, target = next(iter(mnist_model.test_loader))
    case_num = randint(0, len(data) - 1)

    torch.save(data[case_num].unsqueeze(0), './rand_image/test_img.pt')
    torch.save(target[case_num], './rand_image/test_label.pt')

    np.save('./rand_image/test_img.npy', data.numpy()[case_num].ravel().astype(np.float32))
    np.save('./rand_image/test_label.npy', target.numpy()[case_num])


def single_image_inference():
    mnist_model = MnistModel()
    mnist_model.network.load_state_dict(torch.load('./saved_models/addernet_mnist.pth'))

    input_tensor = torch.load('test_img.pt')
    input_label = torch.load('test_label.pt')
    input_label = input_label.numpy()

    output = mnist_model.network(input_tensor)
    output = output.detach().numpy()
    pred = np.argmax(output)
    print("Actual label: " + str(input_label))
    print("Pred label: " + str(pred))
    print("Pred label:", output)


def calc_accuracy():
    mnist_model = MnistModel()
    mnist_model.network.load_state_dict(torch.load('./saved_models/addernet_mnist.pth'))

    correct = 0
    for data, target in mnist_model.test_loader:
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = mnist_model.network(data)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    print('Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(mnist_model.test_loader.dataset),
                                               100. * correct / len(mnist_model.test_loader.dataset)))


def calc_latency():

    def run_test(device_name):
        model = MnistModel()
        model.network.load_state_dict(torch.load('./saved_models/addernet_mnist.pth'))
        device = torch.device(device_name)
        model.network.to(device)

        max_batch_size = 64
        mean_latency_list = []
        std_latency_list = []
        batch_size_list = []
        for batch_size in range(1, max_batch_size+1):
            batch_size_list.append(batch_size)
            dummy_input = torch.randn(batch_size, 1, 28, 28, dtype=torch.float).to(device)
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            repetitions = 100
            timings = np.zeros((repetitions, 1))
            # GPU-WARM-UP
            for _ in range(10):
                _ = model.network(dummy_input)
            # MEASURE PERFORMANCE
            with torch.no_grad():
                for rep in range(repetitions):
                    starter.record()
                    _ = model.network(dummy_input)
                    ender.record()
                    # WAIT FOR GPU SYNC
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    timings[rep] = curr_time
            mean_syn = np.sum(timings) / repetitions
            std_syn = np.std(timings)
            mean_latency_list.append(mean_syn*1000.0)
            std_latency_list.append(std_syn*1000.0)
            print("Latency calculated for batch size: {}".format(batch_size), end="")
            print('\r', end='')

        print('Execution finished on:', device_name)
        return mean_latency_list

    # cpu_time = run_test('cpu')
    gpu_time = run_test('cuda')

    plt.figure(figsize=(10, 5))
    # plt.plot(cpu_time, label='CPU')
    plt.plot(gpu_time, label='CUDA')
    plt.legend(loc="upper left")
    plt.xlabel('Input Batch Size')
    plt.ylabel('Mean Latency (ms)')
    plt.title('Inference Latency - AdderNetMnist Pytorch Model')
    plt.savefig('figures/pytorch_latency.jpg')

    print('Inference latency graph saved for multiple batch sizes')
    # print('Single image inference latency cpu(ms):', cpu_time[0])
    print('Single image inference latency gpu(ms):', gpu_time[0])


def main():
    # train_n_save_weights()
    # single_image_inference()
    calc_accuracy()
    calc_latency()


if __name__ == '__main__':
    main()