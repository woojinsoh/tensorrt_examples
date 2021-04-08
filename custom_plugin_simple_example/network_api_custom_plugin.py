import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import os

from random import randint
import tensorrt as trt
import common

import ctypes

WORKING_DIR = os.environ.get("TRT_WORKING_DIR") or os.path.dirname(os.path.realpath(__file__))
CLIP_PLUGIN_LIBRARY = os.path.join(WORKING_DIR, 'build/libclipplugin.so')

# Load the shared object file containing the Clip plugin implementation.
ctypes.CDLL(CLIP_PLUGIN_LIBRARY)

# Get TRT Plugin repo
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list


class Net(nn.Module):
    """ sample network module"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = x.view(-1, 800)

        # ReLU6 is not supported from TensorRT, which will be substituted with the custom layer plugin
        x = F.relu6(self.fc1(x))
        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



class MnistModel(object):
    """ Get MNIST data and weights via model training"""
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 100
        self.learning_rate = 0.0025
        self.sgd_momentum = 0.9
        self.log_interval = 100
        # Fetch MNIST data set.
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/workspace/tensorrt/yolov5/data/', train=True, download=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            timeout=600)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/workspace/tensorrt/yolov5/data/', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=self.test_batch_size,
            shuffle=True,
            num_workers=1,
            timeout=600)
        self.network = Net()

    # Train the network for one or more epochs, validating after each epoch.
    def learn(self, num_epochs=1):
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



def get_trt_custom_plugin(plugin_name):
    """get tensorRT custom plugin"""
    plugin = None
    for plugin_creator in PLUGIN_CREATORS:        
        if plugin_creator.name == plugin_name:
            clipMin_field = trt.PluginField("clipMin", np.array([0.0], dtype=np.float32), trt.PluginFieldType.FLOAT32)
            clipMax_field = trt.PluginField("clipMax", np.array([6.0], dtype=np.float32), trt.PluginFieldType.FLOAT32)                        
            field_collection = trt.PluginFieldCollection([clipMin_field, clipMax_field])
            plugin = plugin_creator.create_plugin(name=plugin_name, field_collection = field_collection)
    return plugin



def populate_network(network, weights):
    """ Build model using tensorRT network API """
    input_tensor = network.add_input(name="Input", dtype=trt.float32, shape=(1, 28, 28))

    conv1_w = weights['conv1.weight'].numpy()
    conv1_b = weights['conv1.bias'].numpy()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=20, kernel_shape=(5, 5), kernel=conv1_w, bias=conv1_b)
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

    # custom plugin is implemented. The name of the plugin should be referred to CLIP_PLUGIN_NAME in plugin/customClipPlugin.cpp"
    relu1 = network.add_plugin_v2(inputs=[fc1.get_output(0)], plugin=get_trt_custom_plugin("CustomClipPlugin"))    

    fc2_w = weights['fc2.weight'].numpy()
    fc2_b = weights['fc2.bias'].numpy()
    fc2 = network.add_fully_connected(relu1.get_output(0), 10, fc2_w, fc2_b)

    fc2.get_output(0).name = "Output"
    network.mark_output(tensor=fc2.get_output(0))


def build_engine(weights):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, builder.create_builder_config() as config:
        config.max_workspace_size = common.GiB(1)
        # Populate the network using weights from the PyTorch model.
        populate_network(network, weights)
        # Build and return an engine.
        return builder.build_engine(network, config)


def load_random_test_case(model, pagelocked_buffer):
    # Select an image at random to be the test case.
    img, expected_output = model.get_random_testcase()
    # Copy to the pagelocked input buffer
    np.copyto(pagelocked_buffer, img)
    return expected_output



def main():        
    # By doing this, you will also register the Clip plugin with the TensorRT
    # PluginRegistry through use of the macro REGISTER_TENSORRT_PLUGIN present
    # in the plugin implementation. Refer to plugin/clipPlugin.cpp for more details.
    if not os.path.isfile(CLIP_PLUGIN_LIBRARY):
        raise IOError("\n{}\n{}\n{}\n".format(
            "Failed to load library ({}).".format(CLIP_PLUGIN_LIBRARY),
            "Please build the Clip sample plugin.",
            "For more information, see the included README.md"
        ))

    # Train MNIST data and get weights.
    mnist_model = MnistModel()
    mnist_model.learn()
    weights = mnist_model.get_weights()
    
    # Do inference with TensorRT.
    with build_engine(weights) as engine:        
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            case_num = load_random_test_case(mnist_model, pagelocked_buffer=inputs[0].host)            
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            pred = np.argmax(output)
            print("Test Case: " + str(case_num))
            print("Prediction: " + str(pred))


if __name__ == '__main__':
    main()