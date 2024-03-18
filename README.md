<div align="center">
<p align="center"> <img src="figs/WechatIMG2431.jpg" width="200px"> </p>
</div>

# VmambaIR: Visual State Space Model for Image Restoration

[Yuan Shi](https://github.com/shiyuan7), [Bin Xia](https://github.com/Zj-BinXia), [Xiaoyu Jin](), [Xing Wang], [Tianyu Zhao], [Xin Xia], [Xuefeng Xiao], and [Wenming Yang](https://scholar.google.com/citations?user=vsE4nKcAAAAJ&hl=zh-CN), "VmambaIR: Visual State Space Model for Image Restoration", arXiv, 2024

[[arXiv]()] [[supplementary material]()] [[visual results]()] [pretrained models]

#### 🔥🔥🔥 News

- **2024-03-18:** This repo is released.

---

> **Abstract:** Image restoration is a critical task in low-level computer vision, aiming to restore high-quality images from degraded inputs. Various models, such as convolutional neural networks (CNNs), generative adversarial networks (GANs), transformers, and diffusion models (DMs), have been employed to address this problem with significant impact. However, CNNs have limitations in capturing long-range dependencies. DMs require large prior models and computationally intensive denoising steps. Transformers have powerful modeling capabilities but face challenges due to quadratic complexity with input image size. To address these challenges, we propose VmambaIR, which introduces State Space Models (SSMs) with linear complexity into comprehensive image restoration tasks. We utilize a Unet architecture to stack our proposed Omni Selective Scan (OSS) blocks, consisting of an OSS module and an Efficient Feed-Forward Network (EFFN). Our proposed omni selective scan mechanism overcomes the unidirectional modeling limitation of SSMs by efficiently modeling image information flows in all six directions. Furthermore, we conducted a comprehensive evaluation of our VmambaIR across multiple image restoration tasks, including image deraining, single image super-resolution, and real-world image super-resolution. Extensive experimental results demonstrate that our proposed VmambaIR achieves state-of-the-art (SOTA) performance with much fewer computational resources and parameters. Our research highlights the potential of state space models as promising alternatives to the transformer and CNN architectures in serving as foundational frameworks for next-generation low-level visual tasks.

![](figs/Snipaste_2024-03-18_21-18-39.png)

---
Single Imgae Super-Resolution

[<img src="figs/img_005_mamba.png" height="216"/>](https://imgsli.com/MjQ4MjE2) [<img src="figs/img_065_mamba.png" height="216"/>](https://imgsli.com/MjQ4MjE5) [<img src="figs/img_061_mamba.png" height="216"/>](https://imgsli.com/MjQ4MjIw) [<img src="figs/img_098_mamba.png" height="216"/>](https://imgsli.com/MjQ4MjIx)


Real-World Imgae Super-Resolution

[<img src="figs/0802_mamba.png" height="216"/>](https://imgsli.com/MjQ4MjA4) [<img src="figs/0849_mamba.png" height="216"/>](https://imgsli.com/MjQ4MjEw) [<img src="figs/0891_mamba.png" height="216"/>](https://imgsli.com/MjQ4MjEx) [<img src="figs/0893_mamba.png" height="216"/>](https://imgsli.com/MjIyMzA4)



---


## ⚒️ TODO

* [ ] Release code and pretrained models

## 🔗 Contents

1. Datasets
1. Models
1. Training
1. Testing
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

## <a name="citation"></a>📎 Citation

If you find the code helpful in your resarch or work, please cite the following paper(s).

```
@article{
}
```

## <a name="acknowledgements"></a>💡 Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR), [Vmamba](https://github.com/MzeroMiko/VMamba).

