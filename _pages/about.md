---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

Hi, this is Yukang Chen (陈玉康)’s website!   
I am a Research Scientist in NVIDIA, working with [Prof. Song Han](https://hanlab.mit.edu/songhan).  
I got my Ph.D. degree in CUHK, supervised by [Prof. Jiaya Jia](https://jiaya.me).  
During my Ph.D. study, I worked closely with [Prof. Xiaojuan Qi](https://scholar.google.com/citations?user=bGn0uacAAAAJ&hl=en) and [Dr. Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&hl=en).

I focus on Efficient and **Long AI** - **Boosting AI's Long ability while keeping Efficiency**.  
This direction covers, but is not limited to, the following topics:
- 🚗 **Long-range AD**: Long-range 3D perception via **sparse convolution**.
- 🚀 **Long-context LLMs**: Efficient fine-tuning of long-context LLMs via **sparse attention**.
- 🎥 **Long-video VLMs**: Scaling VLMs to long videos via **sequence parallelism**.
- 🧠 **Long-sequence Reasoning** Long-sequence RL for LLMs/VLMs via **sequence parallelism**.
- 🎬 **Long-video Generation**: Short→Long AR with efficient fine-tuning via **sparse attention**.

If you are interested in **Long AI** and seeking collaboration, please feel free to contact me via [Email](chenyukang2020@gmail.com).


# 🔥 News
- *2025.09*: &nbsp;🎉🎉 **[Long-RL](https://github.com/NVlabs/Long-RL)** is accepted by **Neurips'25**!
- *2025.01*: &nbsp;🎉🎉 **[LongVILA](https://arxiv.org/pdf/2408.10188)** is accepted by **ICLR'25**!
- *2024.09*: &nbsp;🎉🎉 **[RL-GPT](https://proceedings.neurips.cc/paper_files/paper/2024/file/31f119089f702e48ecfd138c1bc82c4a-Paper-Conference.pdf)** is accepted by **Neurips'24** as **Oral**!
- *2024.02*: &nbsp;🎉🎉 **[LISA](https://github.com/dvlab-research/LISA)** is accepted by **CVPR'24** as **Oral**!
- *2024.01*: &nbsp;🎉🎉 **[LongLoRA](https://github.com/dvlab-research/LongLoRA)** is accepted by **ICLR'24** as **Oral**!
- *2023.04*: &nbsp;🎉🎉 **[3D-Box-Segment-Anything](https://github.com/dvlab-research/3D-Box-Segment-Anything)** is released, a combination of **[VoxelNeXt](https://github.com/dvlab-research/VoxelNeXt)** and **[SAM](https://arxiv.org/abs/2304.02643)**.
- *2023.04*: &nbsp;🎉🎉 **[VoxelNeXt](https://github.com/dvlab-research/VoxelNeXt)** is accepted by **CVPR'23**!
- *2022.03*: &nbsp;🎉🎉 **[Focal Sparse Conv](https://github.com/dvlab-research/FocalsConv)** is accepted by **CVPR'22** as **Oral**!
- *2022.03*: &nbsp;🎉🎉 **[Scale-aware AutoAug](https://ieeexplore.ieee.org/document/9756374)** is accepted by **T-PAMI**!



# 💬 Invited Talks and Report

- *2025.07*: **[Long-RL](https://github.com/NVlabs/Long-RL)** was reported by **机器之心** (see **[Link](https://www.jiqizhixin.com/articles/2025-07-14-2)**).
- *2023.10*: **[LongLoRA](https://github.com/dvlab-research/LongLoRA)** was reported by **新智源** (see [Link](https://mp.weixin.qq.com/s/8QoKHgwjxv7fG_CCqouU8w)).
- *2023.08*: **[LISA](https://github.com/dvlab-research/LISA)** was reported by **量子位** (see **[Link](https://mp.weixin.qq.com/s/ia7_55hfI-cs2wWalmk8yA)**).
- *2023.06*: Invited Talk by CVRP 2023 ScanNet Workshop (see **[Link](http://www.scan-net.org/cvpr2023workshop/)**).
- *2023.06*: Invited Talk by VALSE 2023 Perception Workshop for **[VoxelNeXt](https://github.com/dvlab-research/VoxelNeXt)**.
- *2023.04*: Invited Talk and reported by **将门创投** for **[VoxelNeXt](https://github.com/dvlab-research/VoxelNeXt)** (see **[Link](https://mp.weixin.qq.com/s/ijj9Zy81_645mqCaRbRFAg)**).
- *2022.06*: Invited Talk by **深蓝学院** for **[Focal Sparse Conv](https://github.com/dvlab-research/FocalsConv)**.


# 📝 Representative Publications ([Full List](https://scholar.google.com/citations?user=6p0ygKUAAAAJ))
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2024 Oral</div><img src='https://github.com/yukang2017/yukang2017.github.io/raw/main/images/LongLora.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models**](https://arxiv.org/abs/2309.12307) 
<div style="display: inline">
    <a href="https://arxiv.org/abs/2309.12307"> <strong>[Paper]</strong></a>
    <a href="https://github.com/dvlab-research/LongLoRA"> <strong>[Code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[Abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> We present LongLoRA, an efficient fine-tuning approach that extends the context sizes of pre-trained large language models (LLMs), with limited computation cost. Typically, training LLMs with long context sizes is computationally expensive, requiring extensive training hours and GPU resources. For example, training on the context length of 8192 needs 16x computational costs in self-attention layers as that of 2048. In this paper, we speed up the context extension of LLMs in two aspects. On the one hand, although dense global attention is needed during inference, fine-tuning the model can be effectively and efficiently done by sparse local attention. The proposed shifted sparse attention effectively enables context extension, leading to non-trivial computation saving with similar performance to fine-tuning with vanilla attention. Particularly, it can be implemented with only two lines of code in training, while being optional in inference. On the other hand, we revisit the parameter-efficient fine-tuning regime for context expansion. Notably, we find that LoRA for context extension works well under the premise of trainable embedding and normalization. LongLoRA combines this improved LoRA with S^2-Attn. LongLoRA demonstrates strong empirical results on various tasks on Llama2 models from 7B/13B to 70B. LongLoRA extends Llama2 7B from 4k context to 100k, or Llama2 70B to 32k on a single 8x A100 machine. LongLoRA extends models' context while retaining their original architectures, and is compatible with most existing techniques, like Flash-Attention2. In addition, we further conduct supervised fine-tuning with LongLoRA and our long instruction-following LongAlpaca dataset. </p>
    </div>
<img src='https://img.shields.io/github/stars/dvlab-research/LongLoRA.svg?style=social&label=Star' alt="sym" height="100%">
</div>

**Yukang Chen**, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, Jiaya Jia

- **Efficient fine-tuning** - 100k context on a single 8x A100 with 1.8x speed up. 
- **Easy implementation** - Shifted sparse attention compatible with Flash-Attn.
- **LongAlpaca** - The first open-sourced long instruction-following dataset.

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Arxiv 2025</div><img src='https://github.com/yukang2017/yukang2017.github.io/raw/main/images/LongLive.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**LongLive: Real-time Interactive Long Video Generation**](https://arxiv.org/abs/2509.22622) 
<div style="display: inline">
    <a href="https://arxiv.org/abs/2509.22622"> <strong>[Paper]</strong></a>
    <a href="https://github.com/NVlabs/LongLive"> <strong>[Code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[Abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> We present LongLive, a frame-level autoregressive (AR) framework for real-time and interactive long video generation. Long video generation presents challenges in both efficiency and quality. Diffusion and Diffusion-Forcing models can produce high-quality videos but suffer from low efficiency due to bidirectional attention. Causal attention AR models support KV caching for faster inference, but often degrade in quality on long videos due to memory challenges during long-video training. In addition, beyond static prompt-based generation, interactive capabilities, such as streaming prompt inputs, are critical for dynamic content creation, enabling users to guide narratives in real time. This interactive requirement significantly increases complexity, especially in ensuring visual consistency and semantic coherence during prompt transitions. To address these challenges, LongLive adopts a causal, frame-level AR design that integrates a KV-recache mechanism that refreshes cached states with new prompts for smooth, adherent switches; streaming long tuning to enable long video training and to align training and inference (train-long-test-long); and short window attention paired with a frame-level attention sink, shorten as frame sink, preserving long-range consistency while enabling faster generation. With these key designs, LongLive fine-tunes a 1.3B-parameter short-clip model to minute-long generation in just 32 GPU-days. At inference, LongLive sustains 20.7 FPS on a single NVIDIA H100, achieves strong performance on VBench in both short and long videos. LongLive supports up to 240-second videos on a single H100 GPU. LongLive further supports INT8-quantized inference with only marginal quality loss. </p>
    </div>
<img src='https://img.shields.io/github/stars/NVlabs/LongLive.svg?style=social&label=Star' alt="sym" height="100%">
</div>

Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, Song Han, **Yukang Chen**

- **Efficient fine-tuning** - 100k context on a single 8x A100 with 1.8x speed up. 
- **Easy implementation** - Shifted sparse attention compatible with Flash-Attn.
- **LongAlpaca** - The first open-sourced long instruction-following dataset.

</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">NeurIPS 2025</div><img src='https://github.com/yukang2017/yukang2017.github.io/raw/main/images/Long-RL.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**Long-RL: Scaling RL to Long Sequences**](https://arxiv.org/abs/2507.07966) 
<div style="display: inline">
    <a href="https://arxiv.org/abs/2507.07966"> <strong>[Paper]</strong></a>
    <a href="https://github.com/NVlabs/Long-RL"> <strong>[Code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[Abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> We introduce a full-stack framework that scales up reasoning in vision-language models (VLMs) to long videos, leveraging reinforcement learning. We address the unique challenges of long video reasoning by integrating three critical components: (1) a large-scale dataset, LongVideo-Reason, comprising 104K long video QA pairs with high-quality reasoning annotations across diverse domains such as sports, games, and vlogs; (2) a two-stage training pipeline that extends VLMs with chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL); and (3) a training infrastructure for long video RL, named Multi-modal Reinforcement Sequence Parallelism (MR-SP), which incorporates sequence parallelism and a vLLM-based engine tailored for long video, using cached video embeddings for efficient rollout and prefilling. In our experiments, LongVILA-R1-7B achieves strong performance on video benchmarks, reaching 65.1% and 71.1% accuracy on VideoMME without and with subtitles, respectively, and consistently outperforming LongVILA-7B across multiple benchmarks. Moreover, LongVILA-R1-7B supports processing up to 8,192 video frames per video, and configurable FPS settings. Notably, our MR-SP system achieves up to 2.1x speedup on long video RL training. In addition, we release our training system for public availability that supports RL training on various modalities (video, text, and audio), various models (VILA and Qwen series), and even image and video generation models. On a single A100 node (8 GPUs), it supports RL training on hour-long videos (e.g., 3,600 frames). </p>
    </div>
<img src='https://img.shields.io/github/stars/NVlabs/Long-RL.svg?style=social&label=Star' alt="sym" height="100%">
</div>

Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, Song Han, **Yukang Chen**

- **Efficient fine-tuning** - 100k context on a single 8x A100 with 1.8x speed up. 
- **Easy implementation** - Shifted sparse attention compatible with Flash-Attn.
- **LongAlpaca** - The first open-sourced long instruction-following dataset.

</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2025</div><img src='https://github.com/yukang2017/yukang2017.github.io/raw/main/images/LongVILA.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**LongVILA: Scaling Long-Context Visual Language Models for Long Videos**](https://arxiv.org/abs/2408.10188) 
<div style="display: inline">
    <a href="https://arxiv.org/abs/2408.10188"> <strong>[Paper]</strong></a>
    <a href="https://github.com/NVlabs/VILA/tree/main/longvila"> <strong>[Code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[Abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> Long-context capability is critical for multi-modal foundation models, especially for long video understanding. We introduce LongVILA, a full-stack solution for long-context visual-language models by co-designing the algorithm and system. For model training, we upgrade existing VLMs to support long video understanding by incorporating two additional stages, i.e., long context extension and long video supervised fine-tuning. However, training on long video is computationally and memory intensive. We introduce the long-context Multi-Modal Sequence Parallelism (MM-SP) system that efficiently parallelizes long video training and inference, enabling 2M context length training on 256 GPUs without any gradient checkpointing. LongVILA efficiently extends the number of video frames of VILA from 8 to 2048, achieving 99.8% accuracy in 6,000-frame (more than 1 million tokens) video needle-in-a-haystack. LongVILA-7B demonstrates strong accuracy on 9 popular video benchmarks, e.g., 65.1% VideoMME with subtitle. Besides, MM-SP is 2.1x - 5.7x faster than ring style sequence parallelism and 1.1x - 1.4x faster than Megatron with a hybrid context and tensor parallelism. Moreover, it seamlessly integrates with Hugging Face Transformers. </p>
    </div>
<img src='https://img.shields.io/github/stars/NVlabs/VILA.svg?style=social&label=Star' alt="sym" height="100%">
</div>

**Yukang Chen**, Fuzhao Xue, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, Ethan He, Hongxu Yin, Pavlo Molchanov, Jan Kautz, Linxi Fan, Yuke Zhu, Yao Lu, Song Han

- **Efficient fine-tuning** - 100k context on a single 8x A100 with 1.8x speed up. 
- **Easy implementation** - Shifted sparse attention compatible with Flash-Attn.
- **LongAlpaca** - The first open-sourced long instruction-following dataset.

</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">CVPR 2023</div><img src='https://github.com/yukang2017/yukang2017.github.io/raw/main/images/VoxelNeXt.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking**](https://arxiv.org/abs/2303.11301) 
<div style="display: inline">
    <a href="https://arxiv.org/abs/2303.11301"> <strong>[Paper]</strong></a>
    <a href="https://github.com/dvlab-research/VoxelNeXt"> <strong>[Code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[Abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> 3D object detectors usually rely on hand-crafted proxies, e.g., anchors or centers, and translate well-studied 2D frameworks to 3D. Thus, sparse voxel features need to be densified and processed by dense prediction heads, which inevitably costs extra computation. In this paper, we instead propose VoxelNext for fully sparse 3D object detection. Our core insight is to predict objects directly based on sparse voxel features, without relying on hand-crafted proxies. Our strong sparse convolutional network VoxelNeXt detects and tracks 3D objects through voxel features entirely. It is an elegant and efficient framework, with no need for sparse-to-dense conversion or NMS post-processing. Our method achieves a better speed-accuracy trade-off than other mainframe detectors on the nuScenes dataset. For the first time, we show that a fully sparse voxel-based representation works decently for LIDAR 3D object detection and tracking. Extensive experiments on nuScenes, Waymo, and Argoverse2 benchmarks validate the effectiveness of our approach. Without bells and whistles, our model outperforms all existing LIDAR methods on the nuScenes tracking test benchmark. </p>
    </div>
<img src='https://img.shields.io/github/stars/dvlab-research/VoxelNeXt.svg?style=social&label=Star' alt="sym" height="100%">
</div>

**Yukang Chen**, Jianhui Liu, Xiangyu Zhang, Xiaojuan Qi, Jiaya Jia

- **Efficient fine-tuning** - 100k context on a single 8x A100 with 1.8x speed up. 
- **Easy implementation** - Shifted sparse attention compatible with Flash-Attn.
- **LongAlpaca** - The first open-sourced long instruction-following dataset.

</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">CVPR 2022 Oral</div><img src='https://github.com/yukang2017/yukang2017.github.io/raw/main/images/FocalsConv.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**Focal Sparse Convolutional Networks for 3D Object Detection**](https://arxiv.org/abs/2204.12463) 
<div style="display: inline">
    <a href="https://arxiv.org/abs/2204.12463"> <strong>[Paper]</strong></a>
    <a href="https://github.com/dvlab-research/FocalsConv"> <strong>[Code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[Abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> Non-uniformed 3D sparse data, e.g., point clouds or voxels in different spatial positions, make contribution to the task of 3D object detection in different ways. Existing basic components in sparse convolutional networks (Sparse CNNs) process all sparse data, regardless of regular or submanifold sparse convolution. In this paper, we introduce two new modules to enhance the capability of Sparse CNNs, both are based on making feature sparsity learnable with position-wise importance prediction. They are focal sparse convolution (Focals Conv) and its multi-modal variant of focal sparse convolution with fusion, or Focals Conv-F for short. The new modules can readily substitute their plain counterparts in existing Sparse CNNs and be jointly trained in an end-to-end fashion. For the first time, we show that spatially learnable sparsity in sparse convolution is essential for sophisticated 3D object detection. Extensive experiments on the KITTI, nuScenes and Waymo benchmarks validate the effectiveness of our approach. Without bells and whistles, our results outperform all existing single-model entries on the nuScenes test benchmark at the paper submission time. </p>
    </div>
<img src='https://img.shields.io/github/stars/dvlab-research/FocalsConv.svg?style=social&label=Star' alt="sym" height="100%">
</div>

**Yukang Chen**, Yanwei Li, Xiangyu Zhang, Jian Sun, Jiaya Jia

- **Efficient fine-tuning** - 100k context on a single 8x A100 with 1.8x speed up. 
- **Easy implementation** - Shifted sparse attention compatible with Flash-Attn.
- **LongAlpaca** - The first open-sourced long instruction-following dataset.

</div>
</div>


# 🗒️ Academic Services

- Conference Reviewer: Neurips, ICLR, ICML, CVPR, ICCV, ECCV, and AAAI.
- Journal Reviewer: T-PAMI and T-TIP. 
- Area Chair for AAAI 2026.


# 🎖 Honors and Awards 

- 2025 World's Top 2% Scientists.
- 2023 Final-list candidate of ByteDance Scholarship.
- 2023 Winner of ScanNet Indoor Scene Understanding (CVPR 2023 ScanNet Workshop).
- 2022 1st of nuScenes Lidar Multi-object Tracking Leaderboard.
- 2019 Winner of COCO Detection Challenge (ICCV 2019 COCO Workshop).
