import onnx
import onnx_graphsurgeon as gs
import os
import numpy as np
import torch

@gs.Graph.register()
def transpose(self, *args):
    return self.layer(op="Transpose", inputs=args, outputs=["transpose_output"], attrs={"perm":[0,2,1]})[0]

@gs.Graph.register()
def global_average_pooling(self, *args):
    return self.layer(op="GlobalAveragePool", inputs=args, outputs=["global_avg_pooling_output"])[0]

@gs.Graph.register()
def squeeze(self, *args):
    return self.layer(op="Squeeze", inputs=args, outputs=["squeeze_output"])[0]

def onnx_add_mean_pooling(graph, inp):
    '''add max pooling operations to the last layer of the onnx graph'''
    pooling_inp = graph.transpose(inp)
    pooling_out = graph.global_average_pooling(pooling_inp)
    final_out = graph.squeeze(pooling_out, [np.int32(2)])

    graph.outputs = [final_out]
    graph.outputs[0].to_variable(dtype=np.float32)
    graph.cleanup().toposort()  
    return graph

def onnx_conversion(filename, model, input_tensor):
    '''convert network model to onnx graph'''
    onnx_output = filename
    onnx_path = os.path.realpath(onnx_output)

    print("converting model to ONNX")
    torch.onnx.export(model,              
                  input_tensor, # Model inputs
                  onnx_output,                   # Save Path
                  export_params=True,         # Save Weights in the model
                  opset_version=13,           # ONNX Version
                  do_constant_folding=True,   # when Optimizing
                  input_names = ['input'],    # define input name
                  output_names = ['output'],  # define output name
                #   dynamic_axes={'input' : {0 : 'batch_size'},    # dynamic length
                #                 'output' : {0 : 'batch_size'}})
    )
    print("Persisting onnx file: {}".format(onnx_path))
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)  # insert intermediate layer shape

def onnx_trt_perf_comparison(onnx_path):
    '''simple latency and accuracy comparison between onnx and tensorRT'''
    from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath
    from polygraphy.backend.trt import TrtRunner
    from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
    from polygraphy.comparator import Comparator

    build_engine = EngineFromNetwork(NetworkFromOnnxPath(onnx_path))
    build_onnxrt_session = SessionFromOnnx(onnx_path)

    runners = [
        TrtRunner(build_engine),
        OnnxrtRunner(build_onnxrt_session),
    ]

    results = Comparator.run(runners)

    success =True
    success &= bool(Comparator.compare_accuracy(results))