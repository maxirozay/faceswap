## Installation

```bash
# git clone this repository
git clone https://github.com/haofanwang/inswapper.git
cd inswapper

# create a Python venv
python3 -m venv venv

# activate the venv
source venv/bin/activate

# install required packages
pip install -r requirements.txt
```

You have to install ``onnxruntime-gpu`` manually to enable GPU inference, install ``onnxruntime`` by default to use CPU only inference.

## Download Checkpoints

First, you need to download [face swap model](https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx) and save it under `./checkpoints`. To obtain better result, it is highly recommended to improve image quality with face restoration model. Here, we use [CodeFormer](https://github.com/sczhou/CodeFormer). You can finish all as following, required models will be downloaded automatically when you first run the inference.

```bash
mkdir checkpoints
wget -O ./checkpoints/inswapper_128.onnx https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx

cd ..
git lfs install
git clone https://huggingface.co/spaces/sczhou/CodeFormer
```
