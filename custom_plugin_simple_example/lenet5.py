#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This file contains functions for training a TensorFlow model

import tensorflow as tf
import numpy as np
import os
import tensorrt as trt

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(WORKING_DIR, 'models')


def maybe_mkdir(dir_path):
    """ Create a directory """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_data():
    """ Import MNIST data from keras internal dataset """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    x_train = np.reshape(x_train, (-1, 1, 28, 28))
    x_test = np.reshape(x_test, (-1, 1, 28, 28))
    return x_train, y_train, x_test, y_test


def build_model():
    """ Create lenet5 using Keras Sequential API """
    # Create the keras model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[1, 28, 28], name="InputLayer"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    
    # ReLU6 is not supported from TensorRT, which will be substituted with the custom layer plugin
    model.add(tf.keras.layers.Activation(activation=tf.nn.relu6, name="ReLU6"))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="OutputLayer"))
    return model


def train_model():    
    """ Train and evaluate model """
    # Create Keras model
    model = build_model()

    # Print model architecture
    model.summary()

    # Compile model with optimizer
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Load data
    x_train, y_train, x_test, y_test = load_data()
    
    # Train model on train data
    model.fit(
        x_train, y_train,
        epochs = 5,
        verbose = 1)
        
    # Evaluate model on test data
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test loss: {}\nTest accuracy: {}".format(test_loss, test_acc))

    return model


def save_model(model):    
    """ Save your model as a frozen graph with TF2.0 compatibility """
    # Make directory to save model in if it doesn't exist already
    maybe_mkdir(MODEL_DIR)

    # Save model to SavedModel format
    tf.saved_model.save(model, MODEL_DIR)
    
    # Convert Keras model to ConcretFunction
    full_model = tf.function(lambda x:model(x))
    full_model = full_model.get_concrete_function(
        x = tf.TensorSpec(model.inputs[0].shape, model.input[0].dtype))
    
    # Get frozen concreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs ")
    print(frozen_func.outputs)

    # Persist frozen graph from frozen ConcretFunction
    tf.io.write_graph(graph_or_graph_def = frozen_func.graph,
                      logdir = MODEL_DIR,
                      name = "lenet5_frozen_graph.pb",
                      as_text = False)


if __name__ == "__main__":    
    model = train_model()        
    save_model(model)

    
