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

Requirements
Linux system, Windows is not tested, depending on whether causal-conv1d and mamba-ssm can be installed
Python 3.8+, recommended 3.11
PyTorch 2.0 or higher, recommended 2.2
CUDA 11.7 or higher, recommended 12.1
MMCV 2.0 or higher, recommended 2.1
Environment Installation
It is recommended to use Miniconda for installation. The following commands will create a virtual environment named rsmamba and install PyTorch and MMCV. In the following installation steps, the default installed CUDA version is 12.1. If your CUDA version is not 12.1, please modify it according to the actual situation.

Note: If you are experienced with PyTorch and have already installed it, you can skip to the next section. Otherwise, you can follow the steps below.
