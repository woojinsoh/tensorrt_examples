import torch
import numpy as np
import copy
import time
from transformers import T5Config, AutoTokenizer, T5ForConditionalGeneration
from utils.module_wrappers import T5TRTEncoderModuleWithTorchTensor, T5TRTDecoderModuleWithTorchTensor, T5TRTConditionalGenerationWrapper

def print_diff_percentile(diff_tensor:torch.tensor, name: str):
    print("{}: max: {:.4f}, avr: {:.4f}".format(name, diff_tensor.max(), diff_tensor.mean()))
    res = np.sort(diff_tensor.reshape([-1]).numpy(), axis=0)
    for i in [0, 10, 25, 50, 75, 90, 100]:
        print("{}%: {:.4f}".format(i, np.percentile(res, i)), end="\t")
    print()

# Set global values
T5_VARIANT = "t5-base"
TH_MODEL_PATH = "T5/wd-t5-base/T5/t5-base/T5-base/pytorch_model"
ONNX_ENCODER_FILE_PATH = "models/{}/ONNX/{}-encoder.onnx".format(T5_VARIANT, T5_VARIANT)
ONNX_DECODER_FILE_PATH = "models/{}/ONNX/{}-fp16-decoder.onnx".format(T5_VARIANT, T5_VARIANT)
TRT_ENCODER_FILE_PATH = "models/{}/TRT/{}-trt-encoder.engine".format(T5_VARIANT, T5_VARIANT)
TRT_DECODER_FILE_PATH = "models/{}/TRT/{}-trt-fp16-decoder.engine".format(T5_VARIANT, T5_VARIANT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
  # Load PyTorch Model with LM head, Tokenizer, and Model config from Huggingface
  th_model = T5ForConditionalGeneration.from_pretrained(T5_VARIANT).eval().to(device)
  lm_head = copy.deepcopy(th_model.lm_head)
  config = T5Config.from_pretrained(T5_VARIANT, use_cache = False)
  config.tie_word_embeddings = True
  tokenizer = AutoTokenizer.from_pretrained(T5_VARIANT)

  # Sample Input
  batch_size = 1
  input_text = "premise: If I fall asleep then I am going to wake up in 8 hours. hypothesis: I fell asleep but did not wake up in 8 hours."
  print("[Info] Sample Input Text >>> {}".format(input_text))
  input = tokenizer([input_text] * batch_size,
                    padding = "longest",
                    max_length=256,
                    pad_to_multiple_of=8,
                    truncation=True,
                    return_tensors="pt")
  input_ids = input.input_ids.to(device)
  attention_mask = input.attention_mask.to(device)
  dec_input_ids = torch.ones((batch_size, 1), dtype=torch.int32, device=device) * config.decoder_start_token_id

  with torch.no_grad():
      # PyTorch Inference
      th_enc_out = th_model.encoder(input_ids = input_ids, attention_mask = attention_mask) 
      th_dec_out = th_model.decoder(input_ids = dec_input_ids, encoder_hidden_states = th_enc_out.last_hidden_state, encoder_attention_mask = attention_mask)
      # th_head_out = th_model.lm_head(th_dec_out.last_hidden_state * config.d_model ** -0.5)
      stime = time.time()
      th_generate_out = th_model.generate(input_ids = input_ids)  
      th_generate_text = tokenizer.batch_decode(th_generate_out, skip_special_tokens=True)
      print("[PyTorch] Generated Output >>> {}".format(th_generate_text))
      print("elapsed time: " + str(time.time()-stime))
      # TRT Inference
      trt_encoder = T5TRTEncoderModuleWithTorchTensor(TRT_ENCODER_FILE_PATH, config)
      trt_decoder = T5TRTDecoderModuleWithTorchTensor(TRT_DECODER_FILE_PATH, config)
      trt_model = T5TRTConditionalGenerationWrapper(trt_encoder, trt_decoder, lm_head, config)

      trt_enc_out = trt_encoder(input_ids, attention_mask)
      trt_dec_out = trt_decoder(dec_input_ids, trt_enc_out.last_hidden_state, attention_mask)
      # trt_head_out = trt_model.lm_head(trt_dec_out.last_hidden_state * config.d_model ** -0.5)
      stime = time.time()
      trt_generate_out = trt_model.generate(input_ids = input_ids)
      trt_generate_text = tokenizer.decode(trt_generate_out[0])
      print("[TensorRT] Generated Output >>> {}".format(trt_generate_text))
      print("elapsed time: " + str(time.time()-stime))

  # Check the accuracy difference
  print("[The value difference between PyTorch and TRT inference]")
  enc_diff = abs(trt_enc_out.last_hidden_state.cpu() - th_enc_out.last_hidden_state.cpu())
  dec_diff = abs(trt_dec_out.last_hidden_state.cpu() - th_dec_out.last_hidden_state.cpu())
  print_diff_percentile(enc_diff, "encoder_diff")
  print_diff_percentile(dec_diff, "decoder_diff")

if __name__ =="__main__":
  main()