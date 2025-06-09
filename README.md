# RSMambaReg
RSMambaReg: Continual Learning on Remote Sensing Image Classification with State Space Model and Elastic Weight Consolidation

# Abstract
Will be posted after paper published.

# Proposed Architecture
This repository is adopted from paper <a href='https://github.com/KyanChen/RSMamba'>RSMamba: Remote Sensing Image Classification with State Space Model</a>, which is based on the <a href='https://github.com/open-mmlab/mmpretrain'>MMPretrain project</a>.


The process framework for continual learning adopted in this study demonstrates the step-by-step methodology applied to sequential datasets.

![image](https://github.com/user-attachments/assets/45aa9671-e7c4-4993-8305-517f33a13c29)

Fig. 1. Continual Learning Model Design.

The proposed method architecture, RSMambaReg, adopted from https://github.com/KyanChen/RSMamba with addition of a regularization-based method with Elastic Weight Consolidation (EWC).

![image](https://github.com/user-attachments/assets/238b18ba-32c1-4c33-846a-2aae87ac7a99)

Fig. 2. RSMambaReg Architecture

The current branch has been tested on Linux system, PyTorch 2.x and CUDA 12.1, supports Python 3.8+, and is compatible with most CUDA versions.

# Requirements
<ol>
  <li>Linux system, Windows is not tested, depending on whether causal-conv1d and mamba-ssm can be installed</li>
  <li>Python 3.8+, recommended 3.11</li>
  <li>PyTorch 2.0 or higher, recommended 2.2</li>
  <li>CUDA 11.7 or higher, recommended 12.1</li>
  <li>MMCV 2.0 or higher, recommended 2.1</li>
</ol>

#Environment Installation
It is recommended to use Miniconda for installation. The following commands will create a virtual environment named rsmamba and install PyTorch and MMCV. In the following installation steps, the default installed CUDA version is 12.1. If your CUDA version is not 12.1, please modify it according to the actual situation.

Note: If you are experienced with PyTorch and have already installed it, you can skip to the next section. Otherwise, you can follow the steps below.

### Step 0 : Install <a href='https://docs.conda.io/projects/miniconda/en/latest/index.html'>Miniconda</a>

### Step 1: Create a virtual environment named `rsmamba` and activate it.

```bash
conda create -n rsmamba python=3.11 -y
conda activate rsmamba
````

### Step 2: Install [PyTorch2.2.x.](https://pytorch.org/get-started/locally/)

Linux/Windows:

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 -y
````

or

```bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
````


### Step 3: Install [MMCV2.1.x.](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)


```bash
pip install -U openmim
mim install mmcv==2.1.0
# or
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
````

### Step 4: Install other dependencies.


```bash
pip install -U mat4py ipdb modelindex
pip install transformers==4.39.2
pip install causal-conv1d==1.2.0.post2
pip install mamba-ssm==1.2.0.post1
````


