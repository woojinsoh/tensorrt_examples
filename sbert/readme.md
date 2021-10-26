# SBERT inference acceleration with TensorRT

This work is based on tensorRT container(nvcr.io/nvidia/tensorrt:21.09-py3) from NGC on NVIDIA V100 GPU cards. In this version, input shape including batch size is fixed according to the sample tensor shape.

## Quick start
1. Install related libraries.
    ```bash
    pip install -r requirements.txt
    ```
2. Set flag variables inside `sbert_main.py`. Once ONNX files and TRT Engine are created, please modify the corresponding flag values to False.

3. Run `sbert_main.py`.
    ```bash
    python sbert_main.py
    ```
