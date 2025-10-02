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
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2024 Oral</div><img src='https://github.com/yukang2017/yukang2017.github.io/blob/main/images/LongLora.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models**](https://arxiv.org/abs/2309.12307) <img src='https://img.shields.io/github/stars/dvlab-research/LongLoRA.svg?style=social&label=Star' alt="sym" height="100%">
<div style="display: inline">
    <a href="https://arxiv.org/abs/2309.12307"> <strong>[Paper]</strong></a>
    <a href="https://github.com/dvlab-research/LongLoRA"> <strong>[Code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[Abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> We present LongLoRA, an efficient fine-tuning approach that extends the context sizes of pre-trained large language models (LLMs), with limited computation cost. Typically, training LLMs with long context sizes is computationally expensive, requiring extensive training hours and GPU resources. For example, training on the context length of 8192 needs 16x computational costs in self-attention layers as that of 2048. In this paper, we speed up the context extension of LLMs in two aspects. On the one hand, although dense global attention is needed during inference, fine-tuning the model can be effectively and efficiently done by sparse local attention. The proposed shifted sparse attention effectively enables context extension, leading to non-trivial computation saving with similar performance to fine-tuning with vanilla attention. Particularly, it can be implemented with only two lines of code in training, while being optional in inference. On the other hand, we revisit the parameter-efficient fine-tuning regime for context expansion. Notably, we find that LoRA for context extension works well under the premise of trainable embedding and normalization. LongLoRA combines this improved LoRA with S^2-Attn. LongLoRA demonstrates strong empirical results on various tasks on Llama2 models from 7B/13B to 70B. LongLoRA extends Llama2 7B from 4k context to 100k, or Llama2 70B to 32k on a single 8x A100 machine. LongLoRA extends models' context while retaining their original architectures, and is compatible with most existing techniques, like Flash-Attention2. In addition, we further conduct supervised fine-tuning with LongLoRA and our long instruction-following LongAlpaca dataset. </p>
    </div>
</div>

**Yukang Chen**, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, Jiaya Jia

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
