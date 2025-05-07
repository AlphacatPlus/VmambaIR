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

3. Install dependencies
```
cd VmambaIR
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

```

4. Install mamba
```
cd Mamba
cd kernels/selective_scan && pip install .
```

