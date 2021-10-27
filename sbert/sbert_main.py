
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import onnx
import onnx_graphsurgeon as gs
import time
import argparse

from onnx_impl import onnx_add_mean_pooling, onnx_conversion, onnx_trt_perf_comparison
from trt_impl import EngineBuilder, TRTInfer

parser = argparse.ArgumentParser(description="Arguments for sbert trt converting")
parser.add_argument("--onnx-file", default='sbert.onnx', type=str, help='onnx file name')
parser.add_argument("--modified-onnx-file", default='modified.onnx', type=str, help='modified onnx graph with onnx-graphsurgeon')
parser.add_argument("--onnx-conversion", action='store_true', help="check if it needs onnx conversion")
parser.add_argument("--graph-surgeon", action='store_true', help="check if it needs to apply onnx-graphsurgeon")
parser.add_argument("--build-trt-engine", action='store_true', help="check if it needs to build trt engine")
parser.add_argument("--precision", required=True, help='precision when building trt engine', choices=['fp16', 'fp32'])
parser.add_argument("--trt-inference", action='store_true', help="do inference with trt. it should have trt engine in advance")
args = parser.parse_args()

ONNX_FILE = args.onnx_file
MODIFIED_ONNX_FILE = args.modified_onnx_file
ONNX_CONVERSION = args.onnx_conversion
GRAPH_SURGEON = args.graph_surgeon
COMPARE_PERF = False
BUILD_TRT_ENGINE = args.build_trt_engine
PRECISION = args.precision
TRT_INFERENCE = args.trt_inference


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def from_huggingface():
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1')
    model = AutoModel.from_pretrained('sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1')
    return tokenizer, model


def main():
    tokenizer, model = from_huggingface()
    # Sample sentences we want sentence embeddings for
    sentences = ['This is an example sentence', "Each sentence is converted"]

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Warm-up
    with torch.no_grad():
        model_output = model(torch.randint_like(encoded_input['input_ids'], low=0, high=1000))
    
    # Compute token embeddings
    start_time = time.time()
    with torch.no_grad():
        model_output = model(encoded_input['input_ids'])
    
    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    elapsed_time = time.time() - start_time

    # Huggingface inference Output
    print("Huggingface sentence transformer output:")
    print(sentence_embeddings)


    # TRT Implementation from here
    input_tensor = encoded_input["input_ids"]

    if(ONNX_CONVERSION):
        '''convert model to onnx file and persist it'''
        onnx_conversion(ONNX_FILE, model, input_tensor)

    if(GRAPH_SURGEON):
        '''add max pooling to the model'''
        graph = gs.import_onnx(onnx.load(ONNX_FILE))
        inp = graph.outputs[0]
        onnx_output = MODIFIED_ONNX_FILE
        modified_graph = onnx_add_mean_pooling(graph, inp)
        onnx.save(gs.export_onnx(modified_graph), onnx_output)

    if(COMPARE_PERF):
        onnx_trt_perf_comparison(MODIFIED_ONNX_FILE)

    if(BUILD_TRT_ENGINE):
        builder = EngineBuilder()
        builder.create_network(MODIFIED_ONNX_FILE)  
        builder.create_engine("sbert.trt", PRECISION)

    if(TRT_INFERENCE):
        trt_infer = TRTInfer("sbert.trt")
        print("input shape: {}".format(input_tensor.shape))
        input_batch = to_numpy(input_tensor).reshape(-1).astype(np.int32)
        output, trt_elapsed_time = trt_infer.infer(input_batch)

        print("output shape: {}".format(output.shape))

        if (PRECISION == 'fp32'):
            np.testing.assert_allclose(to_numpy(sentence_embeddings), output, rtol=1e-05, atol=1e-05)
            print("[Passed] Accuracy difference from original model inference is within the tolerance")
        print("[original inference elapsed time] {} secs".format(elapsed_time))
        print("[trt inference elapsed time] {} secs".format(trt_elapsed_time))


if __name__ == "__main__":
    main()