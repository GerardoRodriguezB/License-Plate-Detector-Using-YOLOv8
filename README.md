# License-Plates-Detector-Using-YOLOv8

This repository provides a step-by-step guide to train YOLOv8 on a custom dataset for detecting license plates, which are not included in the default pretrained classes. It also explains how to set up and load an Anaconda environment with the required package versions using Jupyter Notebook.

![Ejemplo de imagen](im/Test.JPG)

##Environment Setup

In the Anaconda prompt create a new environment with python 3.10

```bash
conda create -n env_name python=3.10
```

Then, activate the environment

```bash
conda activate env_name
```
Navigate to the root folder of the project and install the required packages to avoid compatibility issues:

```bash
pip install -r requirements.txt
```

##Install Pytorch
If your machine has a GPU compatible with CUDA, install the GPU versions:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

Otherwise, install the CPU versions:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

##Using the Environment in Jupyter
Register the Anaconda environment `env_name` as a Jupyter kernel

```bash
python -m ipykernel install --user --name=env_name --display-name "Kernel_Name"
```

`--name` is the internal identifier

`--display-name` is the name you will see in jupyter

Once you open your Jupyter Notebook, select your kernel from the top-right corner:

![Ejemplo de imagen](im/jupyter_kernel.JPG)

##Dataset

The dataset used to train the model is available in Kaggle 
[Car Plate Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection), 

It contains 433 images with annotations in PASCAL VOC XML format. This repository includes the YOLO format (TXT) files with normalized coordinates. 
Organize the dataset files in the following folder structure:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

At the end of the notebook it is included a cell in which the detector is tested in a random image of the folder dataset/test/images.

