
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import onnx
import onnx_graphsurgeon as gs
import time

from onnx_impl import onnx_add_mean_pooling, onnx_conversion, onnx_trt_perf_comparison
from trt_impl import EngineBuilder, TRTInfer

ONNX_FILE = "sbert.onnx"
MODIFIED_ONNX_FILE = "modified.onnx"
ONNX_CONVERSION = True
GRAPH_SURGEON = True
COMPARE_PERF = False
BUILD_TRT_ENGINE = True
TRT_INFERENCE = True


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

    if (COMPARE_PERF):
        onnx_trt_perf_comparison(MODIFIED_ONNX_FILE)

    if (BUILD_TRT_ENGINE):
        builder = EngineBuilder()
        builder.create_network(MODIFIED_ONNX_FILE)  
        builder.create_engine("sbert.trt", "FP32")

    if (TRT_INFERENCE):
        trt_infer = TRTInfer("sbert.trt")
        print("input shape: {}".format(input_tensor.shape))
        input_batch = to_numpy(input_tensor).reshape(-1).astype(np.int32)
        output, trt_elapsed_time = trt_infer.infer(input_batch)

        print("output shape: {}".format(output.shape))
        np.testing.assert_allclose(to_numpy(sentence_embeddings), output, rtol=1e-05, atol=1e-05)
        print("[Passed] Accuracy difference from original model inference is within the tolerance")
        print("[original inference elapsed time] {} secs".format(elapsed_time))
        print("[trt inference elapsed time] {} secs".format(trt_elapsed_time))


if __name__ == "__main__":
    main()