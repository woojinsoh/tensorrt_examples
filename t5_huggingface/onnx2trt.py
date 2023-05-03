import os
from utils.trt_utils import TRTRunner
from transformers import T5Config

T5_VARIANT = "t5-base"
TRT_ENGINE_DIR = './models/{}/TRT'.format(T5_VARIANT)

ONNX_ENCODER_FILE_PATH = "models/{}/ONNX/{}-encoder.onnx".format(T5_VARIANT, T5_VARIANT)
ONNX_DECODER_FILE_PATH = "models/{}/ONNX/{}-fp16-decoder.onnx".format(T5_VARIANT, T5_VARIANT)

TRT_ENCODER_FILE_PATH = "models/{}/TRT/{}-trt-encoder.engine".format(T5_VARIANT, T5_VARIANT)
TRT_DECODER_FILE_PATH = "models/{}/TRT/{}-trt-fp16-decoder.engine".format(T5_VARIANT, T5_VARIANT)

os.popen('mkdir -p {}'.format(TRT_ENGINE_DIR))
config = T5Config.from_pretrained(T5_VARIANT, use_cache = False)
max_batch_size = 1

print("[ TRT engine for T5 Encoder ]")
encoder_input_shape_tensors = {
    "input_ids": {
        "min": (max_batch_size, 1),
        "opt": (max_batch_size, 384),
        "max": (max_batch_size, 768)
    },
    "attention_mask": {
        "min": (max_batch_size, 1),
        "opt": (max_batch_size, 384),
        "max": (max_batch_size, 768)
    },
}
# T5-Encoder larger than 't5-small' is highly recommended to use FP32 for the accuracy.
TRTRunner.build_engine(TRT_ENCODER_FILE_PATH, ONNX_ENCODER_FILE_PATH, encoder_input_shape_tensors, fp16=False)



print("\n[ TRT engine for T5 Decoder ]")
# For beam search without cache, decoder engine's inputs are expanded "num_beams" times. i.e., the first dimension should be max_batch_size * num_beams.
decoder_input_shape_tensors = {
    "input_ids": {
        "min": (max_batch_size, 1),    
        "opt": (max_batch_size, 384),
        "max": (max_batch_size, 768)
    },
    "encoder_hidden_states":{
        "min": (max_batch_size, 1, config.d_model),
        "opt": (max_batch_size, 384, config.d_model),
        "max": (max_batch_size, 768, config.d_model)
    },
    "encoder_attention_mask": {
        "min": (max_batch_size, 1),
        "opt": (max_batch_size, 384),
        "max": (max_batch_size, 768)
    },
}
TRTRunner.build_engine(TRT_DECODER_FILE_PATH, ONNX_DECODER_FILE_PATH, decoder_input_shape_tensors, fp16=True)
