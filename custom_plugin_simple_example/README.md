# Custom plugin implementation for TensorRT

## Adding a custom layer to Tensorflow 2.0 network with TensorRT in Python (Sample #1)

This sample, `uff_custom_plugin`, demonstrates how to use plugins written in C++ with the TensorRT Python bindings and UFF Parser. This sample uses the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).
In addition, this sample is modifed from the [TensorRT official repository](https://github.com/NVIDIA/TensorRT/tree/master/samples/python/uff_custom_plugin) for TF 2.0 compatibility.
Since TF 2.0, Frozen graph has been a problem because they have removed `tf.Session` and some functionality. Generating a frozen model in TF2.0 is based on [Frozen Graph Tensorflow 2.x](https://github.com/leimao/Frozen-Graph-TensorFlow/tree/master/TensorFlow_v2).


## Adding a custom layer to PyTorch network implementing TensorRT Network definition API (Sample #2)
This sample, `network_api_custom_plugin`, demonstrates how to use plugins with the simple CNN model written in [Python tensorRT network definition API](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#network_python). This sample also uses MNIST dataset.

## How they work

These two samples implement a clip layer (as a CUDA kernel), wraps the implementation in a TensorRT plugin (with a corresponding plugin creator) and then generates a shared library module containing its code. The user then dynamically loads this library in Python, which causes the plugin to be registered in TensorRT's PluginRegistry and makes it available to the UFF parser as well.

This sample includes:
`plugin/`
This directory contains files for the Clip layer plugin.

`clipKernel.cu`
A CUDA kernel that clips input.

`clipKernel.h`
The header exposing the CUDA kernel to C++ code.

`customClipPlugin.cpp`
A custom TensorRT plugin implementation, which uses the CUDA kernel internally.

`customClipPlugin.h`
The ClipPlugin headers.

`lenet5.py`
This script trains an MNIST network that uses ReLU6 activation using the clip plugin.

`sample.py`
This script transforms the trained model into UFF (delegating ReLU6 activations to ClipPlugin instances) and runs inference in TensorRT.

`requirements.txt`
This file specifies all the Python packages required to run this Python sample.

## Prerequisites

1. Run TensorRT container.
    ```bash
    docker run --rm -it --gpus all -v `pwd`:/workspace nvcr.io/nvidia/tensorrt:21.02-py3 /bin/bash
    ```

2. [Install CMake](https://cmake.org/download/).

3. Switch back to test container (if applicable) and install the dependencies for Python.
   ```bash
   python3 -m pip install -r requirements.txt
   ```

  NOTE
  - On PowerPC systems, you will need to manually install TensorFlow using IBM's [PowerAI](https://www.ibm.com/support/knowledgecenter/SS5SF7_1.6.0/navigation/pai_install.htm).
  - On Jetson boards, you will need to manually install TensorFlow by following the documentation for [Xavier](https://docs.nvidia.com/deeplearning/dgx/install-tf-xavier/index.html) or [TX2](https://docs.nvidia.com/deeplearning/dgx/install-tf-jetsontx2/index.html).

4. Install the UFF toolkit and graph surgeon; depending on your TensorRT installation method, to install the toolkit and graph surgeon, choose the method you used to install TensorRT for instructions (see [TensorRT Installation Guide: Installing TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing)).

5. Build the plugin and its corresponding Python bindings.
    ```bash
    mkdir build && pushd build
    cmake .. && make -j
    popd
    ```
    **NOTE:** If any of the dependencies are not installed in their default locations, you can manually specify them. For example:
        ```
        cmake .. -DPYBIND11_DIR=/path/to/pybind11/
                -DCMAKE_CUDA_COMPILER=/usr/local/cuda-x.x/bin/nvcc  (Or adding /path/to/nvcc into $PATH)
                -DCUDA_INC_DIR=/usr/local/cuda-x.x/include/  (Or adding /path/to/cuda/include into $CPLUS_INCLUDE_PATH)
                -DPYTHON3_INC_DIR=/usr/include/python3.6/
                -DTRT_LIB=/path/to/tensorrt/lib/
                -DTRT_INCLUDE=/path/to/tensorrt/include/
        ```

        `cmake ..` displays a complete list of configurable variables. If a variable is set to `VARIABLE_NAME-NOTFOUND`, then you’ll need to specify it manually or set the variable it is derived from correctly.


## Running Sample #1
The custom plugin is added using `GraphSurgeon` API when TF2.0 is converted to UFF.

1. Build and train model using keras dataset and persist it as `pb` file with frozen graphs.
    ```bash
    python3 lenet5.py
    ```

2.  Run inference using TensorRT with the custom clip plugin implementation:
    ```bash
    python3 uff_custom_plugin.py
    ```

3.  Verify that the sample ran successfully. If the sample runs successfully you should see a match between the test case and the prediction.
    ```
    === Testing ===
    Loading Test Case: 6
    Prediction: 6
    ```

## Running Sample #2
The custom plugin is added to define The network layer, which is not originally supported by TensorRT network definition API.
1. Build and train model using keras dataset and its weights are directly fed into tensorRT Engine.
    ```bash
    python3 network_api_custom_plugin.py
    ```

2.  Verify that the sample ran successfully. If the sample runs successfully you should see a match between the test case and the prediction.
    ```
    === Testing ===
    Test Case: 4
    Prediction: 4
    ```