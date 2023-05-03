# T5 from Huggingface inference acceleration with TensorRT

This work is based on tensorRT container(nvcr.io/nvidia/pytorch:22.12-py3) from NGC with NVIDIA A100. 

## Quick start
### Pre-requisite
1. Install related libraries.
    ```bash
    pip install -r requirements.txt
    ```

2. Export to ONNX. Modify file paths if needed.
    ```bash
    python th2onnx.py
    ```

3. Convert T5 ONNX models into TensorRT engines. Modify the file paths if needed.
    ```bash
    python onnx2trt.py
    ```

### Inference
- Simple TRT inference
    ```bash
    python trt_infer.py
    ```

- Implementing Huggingface `generate` funtion with PyTorch tensors.
    ```bash
    python trt_hf_infer.py
    ```

- Polygraphy test for ONNX and TRT 
    ```bash
    python polygraphy_test.py
    ```
