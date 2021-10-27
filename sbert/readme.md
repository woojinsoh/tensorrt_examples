# SBERT inference acceleration with TensorRT

This work is based on tensorRT container(nvcr.io/nvidia/tensorrt:21.09-py3) from NGC on NVIDIA V100 GPU cards. In this version, input shape including batch size is fixed according to the sample tensor shape.

## Quick start
1. Install related libraries.
    ```bash
    pip install -r requirements.txt
    ```

2. Run `sbert_main.py`.
    ```bash
    python sbert_main.py --onnx-conversion --graph-surgeon --build-trt-engine --precision fp16 --trt-inference
    ```

3. Once onnx files and trt engine are created, run `sbert_main.py` like
    ```bash
    python sbert_main.py --precision fp16 --trt-inference
    ```