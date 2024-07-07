# Installation

This repository is built in PyTorch 2.3.0 and tested on Debian 11 environment (Python3.9, CUDA12.1).
Follow these intructions

1. Clone our repository
```
git clone https://github.com/AlphacatPlus/VmambaIR.git
cd VmambaIR
```

2. Make conda environment
```
conda create -n vmambair python=3.9
conda activate vmambair
```

3. Install mamba dependencies
```
cd mamba
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
```

4. Install other dependencies
```
cd SRGAN #cd RealSR/Deraining
bash pip.sh

```
