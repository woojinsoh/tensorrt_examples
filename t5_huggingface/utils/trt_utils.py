import torch
import ctypes
import tensorrt as trt
import numpy as np
import os
from cuda import cuda, cudart
from typing import Dict, List
import onnx
import onnx_graphsurgeon as gs


def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: np.dtype):
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, arr: np.ndarray):
        if arr.size > self.host.size:
            raise ValueError(
                f"Tried to fit an array of size {arr.size} into host memory of size {self.host.size}"
            )
        np.copyto(self.host[:arr.size], arr.flat, casting='safe')

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))


class TRTRunner:
    """
    Common features to do inference on TensorRT engine 
    """
    def __init__(self, trt_engine_path, model_config):
        self.logger = trt.Logger(trt.Logger.ERROR)        
        self.engine_path = trt_engine_path
        if os.path.exists(trt_engine_path):
            self.engine = self._load_engine(self.engine_path)
        else:
            raise ValueError("trt engine does not exist at {}. Build trt engine first using build_engine(...)".format(trt_engine_path))
        self.context = self.engine.create_execution_context()

        self.stream = cuda_call(cudart.cudaStreamCreate())
        
        # by default, the optimization profile is 0
        self.profile_idx = 0
        
        # i/o infos
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        
        self.model_config = model_config
    
        self.inputs = []
        self.outputs = []
        self.bindings = []

    @staticmethod
    def build_engine(trt_engine_output_path, onnx_file_input_path, input_shape_tensors, fp16):
        """
        shape_tensors should be defined as a Dict type like:
            shape_tensors = {
                "input_ids": {
                    "min": (1, 1),
                    "opt": (1, 384),
                    "max": (1, 768)
                },
                "attention_mask": {
                    "min": (1, 1),
                    "opt": (1, 384),
                    "max": (1, 768)
                }
            }
        """
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()

        config.set_flag(trt.BuilderFlag.FP16) if fp16 else None
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        config.set_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, enable=True)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2048 * 1024 * 1024)
        for name, shape in input_shape_tensors.items():
            profile.set_shape(name, shape['min'], shape['opt'], shape['max'])
        config.add_optimization_profile(profile)        
        
        parser = trt.OnnxParser(network, logger)
        with open(onnx_file_input_path, "rb") as f:
            if not parser.parse(f.read()):
                raise ValueError("Parse failed")
        
        if fp16:
            network = add_extra_fp32([None, network])[1]
        print(">>> Building TRT engine")
        serialized_engine = builder.build_serialized_network(network, config)
        
        with open(trt_engine_output_path, 'wb') as f:
            f.write(serialized_engine)
        print(">>> TRT engine has been successfully bulit and persisted at {}".format(trt_engine_output_path))

    def _load_engine(self, trt_engine_path):
        with open(trt_engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            return engine
            
    def get_optimization_profile(self, batch_size, seq_len):
        selected_profile_idx = None
        for idx in range(self.engine.num_optimization_profiles):            
            for name in self.tensor_names:
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                   profile_shape =  self.engine.get_tensor_profile_shape(name, idx)
                   # check maximum batch_size and seqeunce lengths for the input
                   if profile_shape[0][0] <= batch_size and profile_shape[2][0] >= batch_size \
                        and profile_shape[0][1] <=  seq_len and profile_shape[2][1] >= seq_len:
                        print("Selected profile: {}".format(profile_shape))
                        selected_profile_idx = idx
                        break

        if selected_profile_idx == -1:
            raise RuntimeError("Couldn't find any profiles that match batch_size={}, seq_len={}".format(batch_size, seq_len))
        
        return selected_profile_idx

    def allocate_buffers(self, profile_idx, max_batch_size, output_dims):
        #Setup I/O bindings
        for name in self.tensor_names:
            # Define Shapes
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                # Get the maximum input seq length from the profile for the dynamic shape input_ids
                shape = self.engine.get_tensor_shape(name) if profile_idx is None else self.engine.get_tensor_profile_shape(name, profile_idx)[-1]
            else:
                # Define output shape
                shape = trt.Dims(output_dims)
            shape_valid = np.all([s >= 0 for s in shape])
            if not shape_valid and profile_idx is None:
                raise ValueError(f"Binding {name} has dynamic shape, but no profile was specified.")
            
            # Define Sizes
            size = trt.volume(shape)
            if self.engine.has_implicit_batch_dimension:
                size *= max_batch_size

            # Define types
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))

            # Allocate host and device buffers
            binding_memory = HostDeviceMem(size, dtype)
            
            # Append the device buffer to device bindings
            self.bindings.append(int(binding_memory.device))

            # Append to the appropriate list
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(binding_memory)
            else:
                self.outputs.append(binding_memory)
    
    def init_bindings(self):
        self.bindings = [0] * self.engine.num_bindings
                        
    def infer(self, data):
        for input_name in data:
            self.inputs[self.engine[input_name]].host = data[input_name]
            self.context.set_input_shape(input_name, trt.Dims(data[input_name].shape))
 
        [cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)) for inp in self.inputs]
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream)
        [cuda_call(cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)) for out in self.outputs]
        cuda_call(cudart.cudaStreamSynchronize(self.stream))

        return [out.host for out in self.outputs]
    
    def torch_infer(self, data: Dict, batch_size: int):
        """
        Using Torch Tensors directly. It doesn't need to allocate the device buffer in advance.
        """
        output_dims = (batch_size, self.model_config.d_model, self.model_config.d_model)
        output_tensor = torch.tensor(0)

        for name in self.tensor_names:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = trt.Dims(data[name].shape)
                self.bindings[self.engine.get_binding_index(name)] = data[name].data_ptr()
                self.context.set_input_shape(name, shape)
            else:
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))                
                shape = trt.Dims(output_dims)
                output_tensor = torch.tensor(np.zeros(shape).astype(dtype)).cuda()
                self.bindings[self.engine.get_binding_index(name)] = output_tensor.data_ptr()
        
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream)
        cuda_call(cudart.cudaStreamSynchronize(self.stream))
        
        return output_tensor


###################################################################################################################################################
#####                             Function for ONNX and tensorRT graph surgeons in order to adapt the accuracy.                               #####
###################################################################################################################################################

def add_extra_fp32(network_definition):
    """
    Force operations involved in layer norm to run in FP32 precision.
    """
    pow_ops = {}
    for layer_index, layer in enumerate(network_definition[1]):
        if layer.type == trt.LayerType.IDENTITY:
            all_fp32 = all([layer.output_type_is_set(o) and layer.get_output_type(o) == trt.float32 for o in range(layer.num_outputs)])
            if all_fp32:
                if layer.get_input(0).dtype == trt.float32:
                    layer.precision = trt.float32

        if layer.type == trt.LayerType.ELEMENTWISE:
            layer.__class__ = getattr(trt, "IElementWiseLayer")
            if layer.op == trt.ElementWiseOperation.POW:
                pow_ops[layer] = layer_index
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)

    for _, index in pow_ops.items():
        # Iterate from few layers before pow to include residual add and cast op.
        # Iterate till 10 layers after pow op to include all operations included in layer norm.
        START_OFFSET = 4
        END_OFFSET = 12
        for i in range(index-START_OFFSET, index+END_OFFSET):
            l = network_definition[1].get_layer(i)
            if l.type == trt.LayerType.REDUCE:
                l.precision = trt.float32
                l.set_output_type(0, trt.float32)

            if l.type == trt.LayerType.ELEMENTWISE:
                l.__class__ = getattr(trt, "IElementWiseLayer")
                if l.op == trt.ElementWiseOperation.SUM:
                    l.precision = trt.float32
                    l.set_output_type(0, trt.float32)

            if l.type == trt.LayerType.UNARY:
                l.__class__ = getattr(trt, "IUnaryLayer")
                if l.op == trt.UnaryOperation.SQRT:
                    l.precision = trt.float32
                    l.set_output_type(0, trt.float32)

            if l.type == trt.LayerType.ELEMENTWISE:
                l.__class__ = getattr(trt, "IElementWiseLayer")
                if l.op == trt.ElementWiseOperation.DIV:
                    l.precision = trt.float32
                    l.set_output_type(0, trt.float32)

            if l.type == trt.LayerType.ELEMENTWISE:
                l.__class__ = getattr(trt, "IElementWiseLayer")
                if l.op == trt.ElementWiseOperation.PROD:
                    l.precision = trt.float32
                    l.set_output_type(0, trt.float32)

    return network_definition


def clamp_weights_onnx(onnx_input_fpath: str, onnx_output_fpath: str, min: float, max: float, ignore_nodes: List = None):
    """
    Clamps given onnx model to targeted upper and lower bounds.
    """

    graph = gs.import_onnx(onnx.load(onnx_input_fpath))
    if ignore_nodes is None:
        ignore_nodes = {}
    else:
        ignore_nodes = {k: True for k in ignore_nodes}

    for tensor in graph.tensors().values():
        if tensor.name in ignore_nodes or isinstance(tensor, gs.ir.tensor.Variable):
            continue

        np.clip(tensor.values, min, max, out=tensor.values)

    for tensor in graph.nodes:
        node_attr = tensor.attrs.get("value", None)
        if tensor.name in ignore_nodes:
            continue

        if node_attr is not None:
            np.clip(node_attr.values, min, max, out=node_attr.values)

    model = gs.export_onnx(graph)
    # onnx.save(model, onnx_output_fpath, save_as_external_data=True)
    onnx.save(model, onnx_output_fpath)

def clamp_weights_onnx_to_fp16_bounds(onnx_input_fpath: str, onnx_output_fpath: str, ignore_nodes: List = None):
    upper_bound = 65504
    return clamp_weights_onnx(onnx_input_fpath, onnx_output_fpath, -upper_bound, upper_bound, ignore_nodes)

def move_t5_cast_op(onnx_input_fpath: str, onnx_output_fpath: str):
    """
    T5 encoder and decoder have cast ops after residual add operation.
    Moving the cast operation before add helps with FP16 accuracy as addition operation
    can cause overflow in FP16.
    """

    graph = gs.import_onnx(onnx.load(onnx_input_fpath))
    cast_nodes = [node for node in graph.nodes if node.op == "Cast"]
    # Version check for backward compatibility
    torch_version_major = int(torch.__version__.split('.')[0])
    torch_version_minor = int(torch.__version__.split('.')[1])
    version_check = torch_version_major == 1 and torch_version_minor > 12
    for n in cast_nodes:
        # Cast appears at the output of add and feeds into a Pow op.
        if n.i().op == "Add":
            found_pow = False
            for o in n.outputs:
                for o1 in o.outputs:
                    if o1.op == "Pow":
                        found_pow = True

            if found_pow:
                if version_check:
                    # Using Clip would be the simplest way, but unfortunately TRT refuses to put "Clip" on Myelin. The WAR
                    # is to insert a Max followed by a Min instead.
                    # Replace the Cast with Max + Min
                    n.op = "Max"
                    n.name = n.name.replace("Cast", "Max")
                    n.attrs = {}
                    lower_bound = gs.Constant(n.name + "/lower_bound", np.array(-64000.0, dtype=np.float32))
                    n.inputs = [n.inputs[0], lower_bound]

                    max_node_output = n.outputs[0]
                    # Max has already exist, avoid tensors with same names
                    max_node_output.name = max_node_output.name.replace("Cast", "ClipMax")

                    upper_bound = gs.Constant(n.name + "/upper_bound", np.array(64000.0, dtype=np.float32))
                    min_node_inputs = [max_node_output, upper_bound]

                    min_node_output = gs.Variable(max_node_output.name.replace("ClipMax", "ClipMin"), dtype = np.float32)
                    min_node = gs.Node(op="Min", inputs = min_node_inputs, outputs = [min_node_output], attrs = {})
                    graph.nodes.append(min_node)

                    for o in max_node_output.outputs:
                        # To avoid loop in graph
                        if o.op != "Min":
                            o.inputs = [min_node_output if i == max_node_output else i for i in o.inputs]
                else:
                    n.i().outputs = n.outputs
                    n.outputs.clear()

    graph.cleanup().toposort()

    add_nodes = [node for node in graph.nodes if node.op == "Add"]
    for n in add_nodes:
        if (version_check and (n.o().o().o().op == "Pow")) or ((not version_check) and (n.o().op == "Pow")):
            add_inputs = n.inputs
            outs = []
            for i in add_inputs:
                identity_out = gs.Variable("identity_out" + i.name, dtype=np.float32)
                new_cast = gs.Node(op="Cast", inputs=[i], outputs=[identity_out], attrs={"to": 1})
                outs.append(identity_out)
                graph.nodes.append(new_cast)
            n.inputs = outs

    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    onnx.save(model, onnx_output_fpath, save_as_external_data=False)