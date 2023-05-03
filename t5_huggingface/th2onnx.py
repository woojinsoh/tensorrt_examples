# Some function using ONNX graphsurgeon to adapt the the accuracy of T5 models.
# refer to: https://github.com/NVIDIA/TensorRT/blob/release/8.6/demo/HuggingFace/NNDF/tensorrt_utils.py#L116

import torch
import os
from transformers import T5ForConditionalGeneration
from utils.module_wrappers import T5EncoderModule, T5DecoderModule
from utils.trt_utils import clamp_weights_onnx_to_fp16_bounds, move_t5_cast_op

T5_VARIANT = "t5-base"
PYTORCH_MODEL_DIR = './models/{}/pytorch'.format(T5_VARIANT)
ONNX_MODEL_DIR = './models/{}/ONNX'.format(T5_VARIANT)

def export_encoder_to_onnx(onnx_output_fpath, t5_model, FP16=False):
    """
    Export T5 Torch encoder from HF to ONNX
    """
    print("[ Exporting Encoder to ONNX format ]")
    if FP16:
        print(">>> FP16 enabled.")
        onnx_output_fpath = os.path.join(ONNX_MODEL_DIR, T5_VARIANT + "-fp16-encoder.onnx")
    
    print(">>> Converting...")
    # Load T5 encoder
    simplified_encoder = T5EncoderModule(t5_model.encoder).eval()

    # Set dummy input
    input_ids = torch.tensor([[42] * 10])
    attention_mask = torch.tensor([[1] * 8 + [0] * 2])
    dummy_input = (input_ids, attention_mask)

    # Export to ONNX    
    opt_args={}
    version_major = int((torch.__version__).split('.')[0])
    version_minor = int((torch.__version__).split('.')[1])
    if version_major < 1 or (version_major == 1 and version_minor < 11):
        opt_args['use_external_data_format'] = True

    torch.onnx.export(
        simplified_encoder,
        dummy_input,
        onnx_output_fpath,
        do_constant_folding=True,
        opset_version=13,
        input_names=('input_ids', 'attention_mask'),
        output_names=('hidden_states',),
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'attention_mask': {0: 'batch', 1: 'sequence'},
            'hidden_states': {0: 'batch', 1: 'sequence'},
        },
        training=torch.onnx.TrainingMode.EVAL,
        **opt_args
    )
    print(">>> ONNX encoder model saved as {}".format(onnx_output_fpath))

    if FP16:
        # from NNDF.tensorrt_utils import clamp_weights_onnx_to_fp16_bounds, move_t5_cast_op at tensorRT huggingface demo github
        move_t5_cast_op(onnx_output_fpath, onnx_output_fpath)
        clamp_weights_onnx_to_fp16_bounds(onnx_output_fpath, onnx_output_fpath)
        print(">>> Clamping FP16 weights for T5 encoder done.")


def export_decoder_to_onnx(onnx_output_fpath, t5_model, FP16=False):
    """
    Export T5 Torch decoder from HF to ONNX
    """
    print("\n[ Exporting Decoder to ONNX format ]")
    if FP16:
        print(">>> FP16 enabled.")
        onnx_output_fpath = os.path.join(ONNX_MODEL_DIR, T5_VARIANT + "-fp16-decoder.onnx")
   
    print(">>> Converting...")   
    # Load T5 encoder
    simplified_encoder = T5EncoderModule(t5_model.encoder)

    # Load T5 decoder
    decoder_with_lm_head = T5DecoderModule(t5_model.decoder, t5_model.lm_head, t5_model.config, is_trt=True)
    
    # This code allows for huggingface compatible torch class to use onnx exporter
    old_forward = decoder_with_lm_head.forward_without_lm
    def _export_forward_without_lm_head(input_ids, encoder_hidden_states, encoder_attention_mask, **kwargs):
        result = old_forward(input_ids, encoder_hidden_states, encoder_attention_mask, use_cache=False, **kwargs)
        return result[0]    
    decoder_with_lm_head.forward = _export_forward_without_lm_head

    # Set dummy input
    input_ids = torch.tensor([[42] * 10])
    attention_mask = torch.tensor([[1] * 8 + [0] * 2])

    
    encoder_hidden_states = simplified_encoder(input_ids)
    dummy_input = (input_ids, encoder_hidden_states, attention_mask )
    
    # Export to ONNX
    opt_args={}
    version_major = int((torch.__version__).split('.')[0])
    version_minor = int((torch.__version__).split('.')[1])
    if version_major < 1 or (version_major == 1 and version_minor < 11):
        opt_args['use_external_data_format'] = True

    torch.onnx.export(
        decoder_with_lm_head,
        dummy_input,
        onnx_output_fpath,
        do_constant_folding=True,
        export_params=True,
        opset_version=13,
        input_names=('input_ids', 'encoder_hidden_states', 'encoder_attention_mask'), 
        output_names=('hidden_states',),
        dynamic_axes={     
                'input_ids': {0: 'batch', 1: 'sequence'},
                'encoder_hidden_states': {0:'batch', 1: 'sequence_encoder_hidden_length'},
                'encoder_attention_mask': {0: 'batch', 1: 'sequence_encoder_input_length'},
                'hidden_states': {0: 'batch', 1: 'sequence'}
        },
        training=torch.onnx.TrainingMode.EVAL,
        **opt_args
    )
    print(">>> ONNX decoder model saved as {}".format(onnx_output_fpath))

    if FP16:
        # from NNDF.tensorrt_utils import clamp_weights_onnx_to_fp16_bounds, move_t5_cast_op at tensorRT huggingface demo github
        move_t5_cast_op(onnx_output_fpath, onnx_output_fpath)
        clamp_weights_onnx_to_fp16_bounds(onnx_output_fpath, onnx_output_fpath)
        print(">>> Clamping FP16 weights for T5 decoder done.")


def main():
    # Load T5 Torch model from Huggingface
    t5_model = T5ForConditionalGeneration.from_pretrained(T5_VARIANT)
    # tokenizer = T5Tokenizer.from_pretrained(T5_VARIANT)
    # config = T5Config.from_pretrained(T5_VARIANT, use_cache = False)

    # Save T5 Torch model locally
    os.popen('mkdir -p {}'.format(PYTORCH_MODEL_DIR))
    t5_model.save_pretrained(PYTORCH_MODEL_DIR)

    # Export T5 torch encoder model to ONNX
    os.popen('mkdir -p {}'.format(ONNX_MODEL_DIR))
    encoder_onnx_model_fpath = os.path.join(ONNX_MODEL_DIR, T5_VARIANT + "-encoder.onnx")
    export_encoder_to_onnx(encoder_onnx_model_fpath, t5_model, FP16=False)

    # Export T5 torch decoder model to ONNX
    decoder_onnx_model_fpath = os.path.join(ONNX_MODEL_DIR, T5_VARIANT + "-decoder.onnx")
    export_decoder_to_onnx(decoder_onnx_model_fpath, t5_model, FP16=True)


if __name__ == "__main__":
    main()