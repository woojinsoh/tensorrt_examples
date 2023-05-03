import torch
import numpy as np
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.generation_utils import GenerationMixin
from utils.trt_utils import TRTRunner

class T5TRTConditionalGenerationWrapper(T5ForConditionalGeneration):
    """
    A Wrapper to use TensorRT engine with Huggingface ConditionalGeneration Modules.
    """
    def __init__(self, t5_trt_encoder, t5_trt_decoder, lm_head, hf_config,  **kwargs):
        super().__init__(hf_config)
        self.encoder = t5_trt_encoder
        self.decoder = t5_trt_decoder  # this should be the decoder without lm-head.
        self.lm_head = lm_head  # this is on GPU


class T5TRTEncoderModuleWithTorchTensor(TRTRunner, torch.nn.Module):
    """
    Huggingface T5 Encoder wrapper with PyTorch tensors directly.
    Assuming that the TRT engine is built on Encoder from Huggingface T5 ConditionalGeneration.
    """
    def __init__(self, engine_path, model_config, *args, **kwargs):
        torch.nn.Module.__init__(self)
        TRTRunner.__init__(self, engine_path, model_config)
        self.main_input_name = "input_ids"
        self.init_bindings()
        
    def forward(self, input_ids, attention_mask, *args, **kwargs):
        data = {
            "input_ids": input_ids.type(torch.int32),
            "attention_mask": attention_mask.type(torch.int32)
        }
        batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]

        out = super().torch_infer(data, batch_size)
        last_hidden_state = out[:,:seq_len,:]
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=last_hidden_state,)
        
class T5TRTDecoderModuleWithTorchTensor(TRTRunner, torch.nn.Module):
    """
    Huggingface T5 Decoder wrapper for TensorRT with Pyorch tensors directly.
    Assuming that the TRT engine is built on Decoder from Huggingface T5 ConditionalGeneration.
    """
    def __init__(self, engine_path, model_config, *args, **kwargs):
        torch.nn.Module.__init__(self)
        TRTRunner.__init__(self, engine_path, model_config)
        self.main_input_name = "input_ids"
        self.init_bindings()
    
    def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask, *args, **kwargs):
        data = {
            "input_ids": input_ids.type(torch.int32),
            "encoder_hidden_states": encoder_hidden_states,     
            "encoder_attention_mask": encoder_attention_mask.type(torch.int32)
        }
        batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]        
  
        out = super().torch_infer(data, batch_size)
        last_hidden_state = out[:,:seq_len,:]
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=last_hidden_state,)

class T5TRTEncoderModule(TRTRunner, torch.nn.Module):
    """
    T5 Encoder Wrapper for TensorRT.
    Assuming that the TRT engine is built on Encoder from Huggingface T5 ConditionalGeneration.
    """
    def __init__(self, engine_path, model_config, max_batch_size=1, *args, **kwargs):
        torch.nn.Module.__init__(self)
        TRTRunner.__init__(self, engine_path, model_config)
        
        self.model_config = model_config
        self.d_model = model_config.d_model   
        self.max_batch_size = max_batch_size
        self.output_dims = (self.max_batch_size, self.d_model, self.d_model)
        self.profile_idx = 0 # self.get_optimization_profile(self.batch_size, self.max_input_seq_len)

        self.allocate_buffers(profile_idx = self.profile_idx, max_batch_size = self.max_batch_size, output_dims = self.output_dims)
            
    def forward(self, input_ids, attention_mask, *args, **kwargs):
        data = {
            "input_ids": input_ids.astype(np.int32),
            "attention_mask": attention_mask.astype(np.int32)
        }
        batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]

        out = super().infer(data)
        last_hidden_state = out[0][:batch_size * seq_len * self.d_model].reshape(batch_size, seq_len, self.d_model)        
        return last_hidden_state


class T5TRTDecoderModule(TRTRunner, torch.nn.Module):
    """
    T5 Decoder Wrapper for TensorRT.
    Assuming that the TRT engine is built on Decoder from Huggingface T5 ConditionalGeneration.
    """
    def __init__(self, engine_path, model_config, max_batch_size=1, max_output_seq=256, *args, **kwargs):
        torch.nn.Module.__init__(self)
        TRTRunner.__init__(self, engine_path, model_config)

        self.max_batch_size = max_batch_size
        self.max_output_seq_len = max_output_seq
        self.d_model = model_config.d_model
        self.output_dims = (self.max_batch_size, self.max_output_seq_len, self.d_model)
        self.profile_idx = 0 # self.get_optimization_profile(self.max_batch_size, self.max_output_seq_len)

        self.allocate_buffers(profile_idx = self.profile_idx, max_batch_size = self.max_batch_size, output_dims = self.output_dims)
        
           
    def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask, *args, **kwargs):
        data = {
            "input_ids": input_ids.astype(np.int32),
            "encoder_hidden_states": encoder_hidden_states,     
            "encoder_attention_mask": encoder_attention_mask.astype(np.int32)
        }
        batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]

        out = super().infer(data)
        last_hidden_state = out[0][:batch_size * seq_len * self.d_model].reshape(batch_size, seq_len, self.d_model)        
        return last_hidden_state


class T5EncoderModule(torch.nn.Module, GenerationMixin):
    """
    A simplified definition of T5 encoder from Huggingface T5 ConditionalGeneration.
    It outputs only the last hidden state. It is used to convert torch to ONNX.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # Use hardcoded value to extend compatibility with older HF versions.
        self.main_input_name = "input_ids"

    def forward(self, *input, **kwargs):
        return self.encoder(*input, **kwargs)[0]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

class T5DecoderModule(torch.nn.Module, GenerationMixin):
    """
    A simplied definition of T5 Decoder without support for loss from Huggingface T5 ConditionalGeneration.
    Decoder with lm-head attached. It is used to convert torch to ONNX.
    """

    def __init__(self, decoder, lm_head, config, is_trt = False):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.config = config
        self.device = "cuda" # HuggingFace's beam search requires to set self.device. Set it to avoid application crash
        # Use hardcoded value to extend compatibility with older HF versions.
        self.main_input_name = "input_ids"
        # trt uses cached and precomputed cross attention vs. framework uses the entire kv cache as output. Need to treat them differently.
        self.is_trt = is_trt

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        use_cache=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "encoder_hidden_states": kwargs["encoder_outputs"].last_hidden_state,
            "use_cache": use_cache,
            "past_key_values": past
        }

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        use_cache = None,
        past_key_values = None,
        return_dict = None,
        **kwargs,
    ):
        # self.decoder is the HuggingFace t5 decoder
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs
        )

        # self.config.d_model ** -0.5 for rescaling output on vocab.
        # as seen in https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5ForConditionalGeneration
        sequence_output = decoder_outputs[0] * self.config.d_model ** -0.5
        logits = self.lm_head(sequence_output)
        if use_cache:
            if self.is_trt:
                past_key_values = ()
                past_key_values_output = decoder_outputs[1]
                for layer_past_states in past_key_values_output:
                    past_key_values = past_key_values + (layer_past_states[:2],)
            else:
                past_key_values = decoder_outputs[1]

        if not return_dict:
            return (logits, past_key_values)

        return Seq2SeqLMOutput(
            logits=logits,
            past_key_values=past_key_values
        )

    def forward_without_lm(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_attention_mask,
        use_cache = None,
        past_key_values = None,
        return_dict = None,
        **kwargs,
    ):
        # self.decoder is the HuggingFace t5 decoder
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs
        )

        # self.config.d_model ** -0.5 for rescaling output on vocab.
        # as seen in https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5ForConditionalGeneration
        last_hidden_state = decoder_outputs[0]
        if use_cache:
            if self.is_trt:
                past_key_values = ()
                past_key_values_output = decoder_outputs[1]
                for layer_past_states in past_key_values_output:
                    past_key_values = past_key_values + (layer_past_states[:2],)
            else:
                past_key_values = decoder_outputs[1]

        if not return_dict:
            return (last_hidden_state, past_key_values)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_state,
            past_key_values=past_key_values
        )


# TODO for kv-cache implementation
class T5DecoderCrossAttentionKVGenerator(torch.nn.Module):
    def __init__(self, decoder, device = "cpu"):
        super().__init__()
        self.decoder = decoder
        self.device = device

    def forward(self, encoder_hidden_states):
        '''
        Use same but simplified code as HF modeling_t5.py to generate cross attention kv cache from provided encoder_hidden_states
        '''
        present_key_values = ()
        for layer_module in self.decoder.block:
            # hidden_states and position_bias are required for the forward call, but irrelevant of cross attention kv cache calculation, so generate dummy variables
            dummy_hidden_states = torch.zeros(1,1).to(self.device)
            dummy_position_bias = torch.zeros(1, layer_module.layer[1].EncDecAttention.n_heads, 1, encoder_hidden_states.shape[1]).to(self.device)
            cross_attention_outputs = layer_module.layer[1](
                hidden_states=dummy_hidden_states,
                key_value_states=encoder_hidden_states,
                use_cache=True,
                past_key_value=None,
                position_bias=dummy_position_bias
            )
            present_key_values = present_key_values + cross_attention_outputs[1]

        return present_key_values

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)