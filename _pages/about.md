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

I focus on **Long AI** - **Efficiently scaling AI to long horizons**.  
This direction covers, but is not limited to, the following topics:
- 📚 **Long-context LLMs**: Efficient long-context LLMs via **sparse attention**.
- 🎥 **Long-video VLMs**: Scaling VLMs to long videos via **sequence parallelism**.
- 🧠 **Long-sequence Reasoning**: Long-sequence RL for LLMs/VLMs via **sequence parallelism**.
- 🎬 **Long-video Generation**: Short→Long AR with efficient fine-tuning via **sparse attention**.
- 🚗 **Long-range Autonomous Driving**: Long-range 3D perception in AD via **sparse convolution**.

If you are interested in **Long AI** and seeking collaboration, please feel free to contact me via [Email](chenyukang2020@gmail.com).
<p align="center" style="border-radius: 10px">
  <img src="https://github.com/yukang2017/yukang2017.github.io/raw/main/images/LongAI.png" width="100%" alt="LongAI"/>
</p>


# 🔥 News
- *2026.04*: &nbsp;🎉🎉 **[TriAttention](https://github.com/WeianMao/triattention)** is accepted by **ICML'26**!
- *2026.01*: &nbsp;🎉🎉 **[LongLive](https://github.com/NVlabs/LongLive)** and **[QeRL](https://github.com/NVlabs/QeRL)** are accepted by **ICLR'26**!
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
- *2026.05*: **[TriAttention](https://github.com/WeianMao/triattention)** was reported by **新智元** (see **[link](https://mp.weixin.qq.com/s/iUQyNUnphYiJAacVKwViAg)**).
- *2025.10*: Invited Talk by ICCV 2025 HiGen Workshop (see **[link](https://higen-2025.github.io)**).
- *2025.10*: **[LongLive](https://github.com/NVlabs/LongLive)** was reported by **新智元** (see **[link](https://mp.weixin.qq.com/s/318DMk2thfpoSFT1oOzBXg)**).
- *2025.07*: **[Long-RL](https://github.com/NVlabs/Long-RL)** was reported by **机器之心** (see **[link](https://www.jiqizhixin.com/articles/2025-07-14-2)**).
- *2023.10*: **[LongLoRA](https://github.com/dvlab-research/LongLoRA)** was reported by **新智元** (see **[link](https://mp.weixin.qq.com/s/8QoKHgwjxv7fG_CCqouU8w)**).
- *2023.08*: **[LISA](https://github.com/dvlab-research/LISA)** was reported by **量子位** (see **[link](https://mp.weixin.qq.com/s/ia7_55hfI-cs2wWalmk8yA)**).
- *2023.06*: Invited Talk by CVRP 2023 ScanNet Workshop (see **[link](http://www.scan-net.org/cvpr2023workshop/)**).
- *2023.06*: Invited Talk by VALSE 2023 Perception Workshop for **[VoxelNeXt](https://github.com/dvlab-research/VoxelNeXt)**.
- *2023.04*: Invited Talk and reported by **将门创投** for **[VoxelNeXt](https://github.com/dvlab-research/VoxelNeXt)** (see **[link](https://mp.weixin.qq.com/s/ijj9Zy81_645mqCaRbRFAg)**).
- *2022.06*: Invited Talk by **深蓝学院** for **[Focal Sparse Conv](https://github.com/dvlab-research/FocalsConv)**.


# 📝 Representative Publications ([Full List](https://scholar.google.com/citations?user=6p0ygKUAAAAJ))

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Arxiv 2026</div><img src='https://github.com/yukang2017/yukang2017.github.io/raw/main/images/LongLive2.0-logo.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**LongLive 2.0: An NVFP4 Parallel Infrastructure for Long Video Generation**](https://arxiv.org/abs/2509.22622) 
<div style="display: inline">
    <a href="https://arxiv.org/abs/2605.18739"> <strong>[Paper]</strong></a>
    <a href="https://github.com/NVlabs/LongLive"> <strong>[Code]</strong></a>
    <a href="https://nvlabs.github.io/LongLive/LongLive2"> <strong>[Demo]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[Abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> We present LongLive-2.0, an NVFP4-based parallel infrastructure throughout the full training and inference workflow of long video generation, addressing speed and memory bottlenecks. For training, we introduce sequence-parallel autoregressive (AR) training, instantiated as Balanced SP, which co-designs the efficient teacher-forcing layout with SP execution by pairing clean-history and noisy-target temporal chunks on each rank, enabling a natural teacher-forcing mask with SP-aware chunked VAE encoding. Combined with NVFP4 precision, it reduces GPU memory cost and accelerates GEMM computation during training, the proportion of which increases as video length grows. Moreover, we show that a high-quality infrastructure and dataset enable a remarkably clean training pipeline. Unlike existing Self-Forcing series methods that rely on ODE initialization and subsequent distribution matching distillation (DMD), LongLive-2.0 directly tunes a diffusion model into a long, multi-shot, interactive auto-regressive (AR) diffusion model. It can be further converted to real-time generation (4 to 2 denoising steps) with standalone LoRA weights. For inference on Blackwell GPUs, we enable W4A4 NVFP4 inference, quantize KV cache into NVFP4 for memory savings, and boost end-to-end throughput with asynchronous streaming VAE decoding. On non-Blackwell GPU architectures, we deploy SP inference to match the speed on Blackwell GPUs, while the quantized KV cache can lower inter-GPU communication of SP. Experiments show up to 2.15x speedup in training, and 1.84x in inference. LongLive-2.0-5B achieves 45.7 FPS inference while attaining strong performance on benchmarks. To our knowledge, LongLive-2.0 is the first NVFP4 training and inference system for long video generation. </p>
    </div>
<img src='https://img.shields.io/github/stars/NVlabs/LongLive.svg?style=social&label=Star' alt="LongLive" height="100%">
</div>

**Yukang Chen** *, Luozhou Wang*, Wei Huang*, Shuai Yang*, Bohan Zhang, Yicheng Xiao, Ruihang Chu, Weian Mao, Qixin Hu, Shaoteng Liu, Yuyang Zhao, Huizi Mao, Ying-Cong Chen, Enze Xie, Xiaojuan Qi, Song Han

- **The first open-source FP4 Infra for Long Video Gen**.
- **Real-time Inference** - 45.7 FPS on 5B model.
- **Support real-video training, few-step distillation, multi-shot, sequence-parallel, NVFP4 KV cache, and async VAE decoding**. 


</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2026</div><img src='https://github.com/yukang2017/yukang2017.github.io/raw/main/images/longlive-logo.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**LongLive: Real-time Interactive Long Video Generation**](https://arxiv.org/abs/2509.22622) 
<div style="display: inline">
    <a href="https://arxiv.org/abs/2509.22622"> <strong>[Paper]</strong></a>
    <a href="https://github.com/NVlabs/LongLive"> <strong>[Code]</strong></a>
    <a href="https://nvlabs.github.io/LongLive/"> <strong>[Demo]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[Abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> We present LongLive, a frame-level autoregressive (AR) framework for real-time and interactive long video generation. Long video generation presents challenges in both efficiency and quality. Diffusion and Diffusion-Forcing models can produce high-quality videos but suffer from low efficiency due to bidirectional attention. Causal attention AR models support KV caching for faster inference, but often degrade in quality on long videos due to memory challenges during long-video training. In addition, beyond static prompt-based generation, interactive capabilities, such as streaming prompt inputs, are critical for dynamic content creation, enabling users to guide narratives in real time. This interactive requirement significantly increases complexity, especially in ensuring visual consistency and semantic coherence during prompt transitions. To address these challenges, LongLive adopts a causal, frame-level AR design that integrates a KV-recache mechanism that refreshes cached states with new prompts for smooth, adherent switches; streaming long tuning to enable long video training and to align training and inference (train-long-test-long); and short window attention paired with a frame-level attention sink, shorten as frame sink, preserving long-range consistency while enabling faster generation. With these key designs, LongLive fine-tunes a 1.3B-parameter short-clip model to minute-long generation in just 32 GPU-days. At inference, LongLive sustains 20.7 FPS on a single NVIDIA H100, achieves strong performance on VBench in both short and long videos. LongLive supports up to 240-second videos on a single H100 GPU. LongLive further supports INT8-quantized inference with only marginal quality loss. </p>
    </div>
<img src='https://img.shields.io/github/stars/NVlabs/LongLive.svg?style=social&label=Star' alt="LongLive" height="100%">
</div>

Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, Song Han, **Yukang Chen**

- **Real-time Inference** - **20.7 FPS** generation on a single H100 GPU.
- **Long Video Gen** - Up to **240-second** generation with interactive prompts. 
- **Efficient Fine-tuning** - Extend Wan to minute-long in 32 H100 GPU-days.

</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICML 2026</div><img src='https://github.com/yukang2017/yukang2017.github.io/raw/main/images/OpenClaw_demo.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**TriAttention: Efficient Long Reasoning with Trigonometric KV Compression**](https://arxiv.org/abs/2510.11696) 
<div style="display: inline">
    <a href="https://arxiv.org/abs/2604.04921"> <strong>[Paper]</strong></a>
    <a href="https://github.com/WeianMao/triattention"> <strong>[Code]</strong></a>
    <a href="https://weianmao.github.io/tri-attention-project-page/#demo"> <strong>[Demo]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[Abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> Extended reasoning in large language models (LLMs) creates severe KV cache memory bottlenecks. Leading KV cache compression methods estimate KV importance using attention scores from recent post-RoPE queries. However, queries rotate with position during RoPE, making representative queries very few, leading to poor top-key selection and unstable reasoning. To avoid this issue, we turn to the pre-RoPE space, where we observe that Q and K vectors are highly concentrated around fixed non-zero centers and remain stable across positions -- Q/K concentration. We show that this concentration causes queries to preferentially attend to keys at specific distances (e.g., nearest keys), with the centers determining which distances are preferred via a trigonometric series. Based on this, we propose TriAttention to estimate key importance by leveraging these centers. Via the trigonometric series, we use the distance preference characterized by these centers to score keys according to their positions, and also leverage Q/K norms as an additional signal for importance estimation. On AIME25 with 32K-token generation, TriAttention matches Full Attention reasoning accuracy while achieving 2.5x higher throughput or 10.7x KV memory reduction, whereas leading baselines achieve only about half the accuracy at the same efficiency. TriAttention enables OpenClaw deployment on a single consumer GPU, where long context would otherwise cause out-of-memory with Full Attention. </p>
    </div>
<img src='https://img.shields.io/github/stars/NVlabs/QeRL.svg?style=social&label=Star' alt="QeRL" height="100%">
</div>

Weian Mao, Xi Lin, Wei Huang, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, **Yukang Chen**

- **High Efficiency** - 2.5x higher FPS and 10.7x KV memory reduction in LLMs.
- **OpenClaw** - 32B LLM on a 24GB GPU. 
- **Long Video Gen** - Reducing 50% KV Cache in AR Long Video Generation.

</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">NeurIPS 2025</div><img src='https://github.com/yukang2017/yukang2017.github.io/raw/main/images/long-rl_demo.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**Long-RL: Scaling RL to Long Sequences**](https://arxiv.org/abs/2507.07966) 
<div style="display: inline">
    <a href="https://arxiv.org/abs/2507.07966"> <strong>[Paper]</strong></a>
    <a href="https://github.com/NVlabs/Long-RL"> <strong>[Code]</strong></a>
    <a href="https://www.youtube.com/watch?v=ykbblK2jiEg"> <strong>[Demo]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[Abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> We introduce a full-stack framework that scales up reasoning in vision-language models (VLMs) to long videos, leveraging reinforcement learning. We address the unique challenges of long video reasoning by integrating three critical components: (1) a large-scale dataset, LongVideo-Reason, comprising 104K long video QA pairs with high-quality reasoning annotations across diverse domains such as sports, games, and vlogs; (2) a two-stage training pipeline that extends VLMs with chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL); and (3) a training infrastructure for long video RL, named Multi-modal Reinforcement Sequence Parallelism (MR-SP), which incorporates sequence parallelism and a vLLM-based engine tailored for long video, using cached video embeddings for efficient rollout and prefilling. In our experiments, LongVILA-R1-7B achieves strong performance on video benchmarks, reaching 65.1% and 71.1% accuracy on VideoMME without and with subtitles, respectively, and consistently outperforming LongVILA-7B across multiple benchmarks. Moreover, LongVILA-R1-7B supports processing up to 8,192 video frames per video, and configurable FPS settings. Notably, our MR-SP system achieves up to 2.1x speedup on long video RL training. In addition, we release our training system for public availability that supports RL training on various modalities (video, text, and audio), various models (VILA and Qwen series), and even image and video generation models. On a single A100 node (8 GPUs), it supports RL training on hour-long videos (e.g., 3,600 frames). </p>
    </div>
<img src='https://img.shields.io/github/stars/NVlabs/Long-RL.svg?style=social&label=Star' alt="Long-RL" height="100%">
</div>

**Yukang Chen**, Wei Huang, Baifeng Shi, Qinghao Hu, Hanrong Ye, Ligeng Zhu, Zhijian Liu, Pavlo Molchanov, Jan Kautz, Xiaojuan Qi, Sifei Liu, Hongxu Yin, Yao Lu, Song Han

- **MR-SP System** - RL on hour-long videos (3,600 frames), up to **2.1x** speedup. 
- **LongVILA-R1-7B** - **8,192 frames**/video and **71.1%** on VideoMME with sub.
- **LongVideo-Reason Dataset** - **104K** long-video QA-reasoning pairs.

</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2025</div><img src='https://github.com/yukang2017/yukang2017.github.io/raw/main/images/longvila-logo.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**LongVILA: Scaling Long-Context Visual Language Models for Long Videos**](https://arxiv.org/abs/2408.10188) 
<div style="display: inline">
    <a href="https://arxiv.org/abs/2408.10188"> <strong>[Paper]</strong></a>
    <a href="https://github.com/NVlabs/VILA/tree/main/longvila"> <strong>[Code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[Abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> Long-context capability is critical for multi-modal foundation models, especially for long video understanding. We introduce LongVILA, a full-stack solution for long-context visual-language models by co-designing the algorithm and system. For model training, we upgrade existing VLMs to support long video understanding by incorporating two additional stages, i.e., long context extension and long video supervised fine-tuning. However, training on long video is computationally and memory intensive. We introduce the long-context Multi-Modal Sequence Parallelism (MM-SP) system that efficiently parallelizes long video training and inference, enabling 2M context length training on 256 GPUs without any gradient checkpointing. LongVILA efficiently extends the number of video frames of VILA from 8 to 2048, achieving 99.8% accuracy in 6,000-frame (more than 1 million tokens) video needle-in-a-haystack. LongVILA-7B demonstrates strong accuracy on 9 popular video benchmarks, e.g., 65.1% VideoMME with subtitle. Besides, MM-SP is 2.1x - 5.7x faster than ring style sequence parallelism and 1.1x - 1.4x faster than Megatron with a hybrid context and tensor parallelism. Moreover, it seamlessly integrates with Hugging Face Transformers. </p>
    </div>
<img src='https://img.shields.io/github/stars/NVlabs/VILA.svg?style=social&label=Star' alt="LongVILA" height="100%">
</div>

**Yukang Chen**, Fuzhao Xue, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, Ethan He, Hongxu Yin, Pavlo Molchanov, Jan Kautz, Linxi Fan, Yuke Zhu, Yao Lu, Song Han

- **MM-SP System** - **2M-tokens** training on 256 GPUs, **1.4x** faster than Megatron. 
- **LongVILA-7B** - **99.8%** on 6,000-frame (>1M tokens) needle-in-a-haystack.
- **LongVILA-SFT Dataset** - **54K** high-quality long video QA pairs.

</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2024 Oral</div><img src='https://github.com/yukang2017/yukang2017.github.io/raw/main/images/longlora-logo.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models**](https://arxiv.org/abs/2309.12307) 
<div style="display: inline">
    <a href="https://arxiv.org/abs/2309.12307"> <strong>[Paper]</strong></a>
    <a href="https://github.com/dvlab-research/LongLoRA"> <strong>[Code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[Abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> We present LongLoRA, an efficient fine-tuning approach that extends the context sizes of pre-trained large language models (LLMs), with limited computation cost. Typically, training LLMs with long context sizes is computationally expensive, requiring extensive training hours and GPU resources. For example, training on the context length of 8192 needs 16x computational costs in self-attention layers as that of 2048. In this paper, we speed up the context extension of LLMs in two aspects. On the one hand, although dense global attention is needed during inference, fine-tuning the model can be effectively and efficiently done by sparse local attention. The proposed shifted sparse attention effectively enables context extension, leading to non-trivial computation saving with similar performance to fine-tuning with vanilla attention. Particularly, it can be implemented with only two lines of code in training, while being optional in inference. On the other hand, we revisit the parameter-efficient fine-tuning regime for context expansion. Notably, we find that LoRA for context extension works well under the premise of trainable embedding and normalization. LongLoRA combines this improved LoRA with S^2-Attn. LongLoRA demonstrates strong empirical results on various tasks on Llama2 models from 7B/13B to 70B. LongLoRA extends Llama2 7B from 4k context to 100k, or Llama2 70B to 32k on a single 8x A100 machine. LongLoRA extends models' context while retaining their original architectures, and is compatible with most existing techniques, like Flash-Attention2. In addition, we further conduct supervised fine-tuning with LongLoRA and our long instruction-following LongAlpaca dataset. </p>
    </div>
<img src='https://img.shields.io/github/stars/dvlab-research/LongLoRA.svg?style=social&label=Star' alt="LongLoRA" height="100%">
</div>

**Yukang Chen**, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, Jiaya Jia

- **Efficient Fine-tuning** - 100k context on a single 8x A100, **1.8x** speed up. 
- **Easy Implementation** - Shifted sparse attention, compatible with Flash-Attn.
- **LongAlpaca** - The first open-source long instruction-following dataset.

</div>
</div>

# 📋 Academic Services

- Area Chair for AAAI 2026.
- Journal Reviewer: T-PAMI and T-TIP.
- Conference Reviewer: Neurips, ICLR, ICML, CVPR, ICCV, ECCV, and AAAI.


# 🎖 Honors and Awards 

- 2025 World's Top 2% Scientists.
- 2023 Final-list candidate of ByteDance Scholarship.
- 2022 1st of nuScenes LiDAR Multi-Object Tracking leaderboard.
- 2019 Winner of COCO Detection Challenge (ICCV 2019 COCO Workshop).
- 2023 Winner of ScanNet Indoor Scene Understanding (CVPR 2023 ScanNet Workshop).
