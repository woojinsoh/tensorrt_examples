# Tacotron2 inference acceleration test with TensorRT

This repo tries implementing TensorRT acceleration to 

1. Tacotron2 + WaveGlow
2. Tacotron2 + ParallelWaveGAN

Most of the codelines are referring to the Tacotron2 section of [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2) repository based on the **PyTorch:22.07-py3 NGC container** where **TensorRT 8.4.3.1** is installed manually. Some parts of the network model architectures and their pipelines must be additionally modified for practical usages and better performance as it is just grounded on the opensource pre-trained model codelines.

## Pre-trained model
Below are checkpoints exploited.
- Tacotron2: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/tacotron2__pyt_ckpt
- WaveGlow: https://ngc.nvidia.com/models/nvidia:waveglow256pyt_fp16
- ParallelWaveGAN: https://github.com/kan-bayashi/ParallelWaveGAN#results (ljspeech_parallel_wavegan.v1.long)

## Getting Started
Explore the following notebooks:
- [Pre-requisite](./1_Pre-requisite.ipynb)
- [TRT Conversion](./2_TensorRT_Conversion.ipynb)
- [Inference Test](./3_Inference_Test.ipynb)

## Test Results
Results are obtained on A100 with PyTorch:22.07-py3 container where TensorRT version is 8.4.3.1.
### Tacotron2 + Waveglow
| Framework | Batch size | Input length | Precision | Num requests | Avg latency(s) | Latency std(s) |
|-----------|------------|--------------|-----------|--------------|----------------|----------------|
|   Torch   |      1     |      155     |   FP16    |      10      |      1.213     |     0.034      |               
| TensorRT  |      1     |      155     |   FP16    |      10      |      0.810     |     0.036      |

### Tacotron2 + ParallelWaveGan
| Framework | Batch size | Input length | Precision | Num requests | Avg latency(s) | Latency std(s) |
|-----------|------------|--------------|-----------|--------------|----------------|----------------|
|   Torch   |      1     |      155     |   FP16    |      10      |      1.152     |     0.032      |               
| TensorRT  |      1     |      155     |   FP16    |      10      |      0.787     |     0.037      |
