<div align="center">
<p align="center"> <img src="figs/logo.png" width="200px"> </p>
</div>

# VmambaIR: Visual State Space Model for Image Restoration

[Yuan Shi](https://github.com/shiyuan7), [Bin Xia](https://github.com/Zj-BinXia), [Xiaoyu Jin](https://github.com/xyjin01), Xing Wang, Tianyu Zhao, Xin Xia, Xuefeng Xiao, and [Wenming Yang](https://scholar.google.com/citations?user=vsE4nKcAAAAJ&hl=zh-CN), "VmambaIR: Visual State Space Model for Image Restoration", arXiv, 2024

[[arXiv](https://arxiv.org/abs/2403.11423)] [[supplementary material]()] [[visual results]()] [pretrained models]

#### 🔥🔥🔥 News

- **2024-03-18:** This repo is released.

---

> **Abstract:** Image restoration is a critical task in low-level computer vision, aiming to restore high-quality images from degraded inputs. Various models, such as convolutional neural networks (CNNs), generative adversarial networks (GANs), transformers, and diffusion models (DMs), have been employed to address this problem with significant impact. However, CNNs have limitations in capturing long-range dependencies. DMs require large prior models and computationally intensive denoising steps. Transformers have powerful modeling capabilities but face challenges due to quadratic complexity with input image size. To address these challenges, we propose VmambaIR, which introduces State Space Models (SSMs) with linear complexity into comprehensive image restoration tasks. We utilize a Unet architecture to stack our proposed Omni Selective Scan (OSS) blocks, consisting of an OSS module and an Efficient Feed-Forward Network (EFFN). Our proposed omni selective scan mechanism overcomes the unidirectional modeling limitation of SSMs by efficiently modeling image information flows in all six directions. Furthermore, we conducted a comprehensive evaluation of our VmambaIR across multiple image restoration tasks, including image deraining, single image super-resolution, and real-world image super-resolution. Extensive experimental results demonstrate that our proposed VmambaIR achieves state-of-the-art (SOTA) performance with much fewer computational resources and parameters. Our research highlights the potential of state space models as promising alternatives to the transformer and CNN architectures in serving as foundational frameworks for next-generation low-level visual tasks.

![](figs/Snipaste_2024-03-18_21-18-39.png)

---
Single Image Super-Resolution

[<img src="figs/Snipaste_2024-03-18_22-28-23.png" height="216"/>](https://imgsli.com/MjQ4MjI5) [<img src="figs/Snipaste_2024-03-18_22-22-34.png" height="216"/>](https://imgsli.com/MjQ4MjI2) [<img src="figs/Snipaste_2024-03-18_22-25-10.png" height="216"/>](https://imgsli.com/MjQ4MjI3) [<img src="figs/Snipaste_2024-03-18_22-26-41.png" height="216"/>](https://imgsli.com/MjQ4MjI4)


Real-World Image Super-Resolution

[<img src="figs/Snipaste_2024-03-18_22-29-05.png" height="216"/>](https://imgsli.com/MjQ4MjMw) [<img src="figs/Snipaste_2024-03-18_22-34-00.png" height="216"/>](https://imgsli.com/MjQ4MjMx) [<img src="figs/Snipaste_2024-03-18_22-34-25.png" height="216"/>](https://imgsli.com/MjQ4MjMy) [<img src="figs/Snipaste_2024-03-18_22-36-36.png" height="216"/>](https://imgsli.com/MjQ4MjM0)


Image Deraining

[<img src="figs/Snipaste_2024-03-18_22-37-57.png" height="216"/>](https://imgsli.com/MjQ4MjM3) [<img src="figs/Snipaste_2024-03-18_22-40-59.png" height="216"/>](https://imgsli.com/MjQ4MjM5) [<img src="figs/Snipaste_2024-03-18_22-42-11.png" height="216"/>](https://imgsli.com/MjQ4MjQz) [<img src="figs/Snipaste_2024-03-18_22-44-20.png" height="216"/>](https://imgsli.com/MjQ4MjQ1)

---


## 🔗 Contents

1. Datasets
1. [Models](#Model)
1. [Installation](#Installation)
1. [Results](#results)
1. [Citation](#citation)
1. [Acknowledgements](#acknowledgements)

## <a name="results"></a>🔎 Results

We achieved state-of-the-art performance on multiple image restoration tasks. Detailed results can be found in the paper.

<details>
<summary>Evaluation on Single Image Super-Resolution (click to expand)</summary>


- quantitative comparisons in Table 1 of the main paper

<p align="center">
  <img width="900" src="figs/Snipaste_2024-03-18_21-51-46.png">
</p>



- visual comparison in Figure 5 of the main paper

<p align="center">
  <img width="900" src="figs/Snipaste_2024-03-18_21-53-07.png">
</p>
</details>



<details>
<summary>Evaluation on Real-World Image Super-Resolution (click to expand)</summary>



- quantitative comparisons in Table 2 of the main paper

<p align="center">
  <img width="900" src="figs/Snipaste_2024-03-18_21-54-35.png">
</p>


- visual comparison in Figure 6 of the main paper

<p align="center">
  <img width="900" src="figs/Snipaste_2024-03-18_21-55-06.png">
</p>

</details>


<details>
<summary>Evaluation on Image Deraining (click to expand)</summary>



- quantitative comparisons in Table 2 of the main paper

<p align="center">
  <img width="900" src="figs/Snipaste_2024-03-18_21-56-16.png">
</p>


- visual comparison in Figure 6 of the main paper

<p align="center">
  <img width="900" src="figs/Snipaste_2024-03-18_21-56-24.png">
</p>

</details>


## <a name="Installation"></a> 🔧 Installation

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


## <a name="Model"></a>📃 Model
Our models are available here [Google Drive](https://drive.google.com/drive/folders/10Pogbp2hkCadGPcuCrlxaZXHt68JAQ3d?usp=sharing). 



## <a name="citation"></a>📎 Citation

If you find the code helpful in your resarch or work, please cite the following paper(s).

```
@article{shi2024vmambair,
  title={VmambaIR: Visual State Space Model for Image Restoration},
  author={Shi, Yuan and Xia, Bin and Jin, Xiaoyu and Wang, Xing and Zhao, Tianyu and Xia, Xin and Xiao, Xuefeng and Yang, Wenming},
  journal={arXiv preprint arXiv:2403.11423},
  year={2024}
}
```

## <a name="acknowledgements"></a>💡 Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR), [Vmamba](https://github.com/MzeroMiko/VMamba), [Restormer](https://github.com/swz30/Restormer) and [DiffIR](https://github.com/Zj-BinXia/DiffIR). We thank them all for their excellent work.

