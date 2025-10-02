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

Hi, this is Wei Huang(黄炜)’s website! I am currently a Ph.D advised by [Prof.Xiaojuan Qi](https://xjqi.github.io/) and [Prof.Shiming Zhang](https://wise.hku.hk/people/). I am also co-supervised by  [Prof.Zhongrui Wang](https://scholar.google.com.sg/citations?user=Ofl3nUsAAAAJ&hl=en). Previously I obtained my bachelor's degree in computer science (Jun 2023) from Beihang University where I was advised by [Prof.Si Liu](https://colalab.net/people) and also worked with [Prof.Xianglong Liu](https://xlliu-beihang.github.io/). 

I’m currently conducting some research in efficient/tiny deep learning and application, including:

🚀 **Efficient AI**: The efficiency of the Large Language/Vision-Language Model and Diffusion Model (e.g. model quantization/binarization).
⌚ **Wearable AI**: AI for wearable and digital health, AI for sensitive organic electrochemical transistor (OECT).
🔥 **Brain-mimic Computing**: Neuromorphic computing and hardware acceleration (e.g. spiking neural network-SNN).




# 🔥 News
- *2025.09*: &nbsp;🎉🎉 Two papers are accepted by **Neurips'25**! One for scaling long-video reasoning (*Long-RL*: Scaling RL to Long Videos) and one for unified reasoning model (*Mindomni*: Unleashing reasoning generation in vision language models with rgpo). All the codes are opensourced now!
- *2025.05*: &nbsp;🎉🎉 One paper for long-term glucose prediction (*Uncertainty-Estimate ProbSparse-Transformer*) is accepted by **Advanced Intelligent Systems**
- *2025.05*: &nbsp;🎉🎉 One paper for structural mixed-precision low-bit quantization for LLMs (*SliM-LLM*) is accepted by **ICML'25**! All the codes are opensourced now!
- *2025.02*: &nbsp;🎉🎉 One paper for efficient fine-grained chain-of-thought video understanding framework (VideoEspresso) is accepted by **CVPR'25**, <span style="color:red">**Oral Paper 0.73%**</span>! All the codes are opensourced now!
- *2025.01*: &nbsp;🎉🎉 Three papers are accepted by **ICLR'25**! One for MoE-LLM compression (*MC-MoE*: MoE-LLM compression) and two papers (*InfoMax*: data pruning; *From-Layers-to-States*: dynamic neural network layer) for data efficiency and dynamic neural networks. All the codes are opensourced now!
- *2024.12*: &nbsp;🎉🎉 One *Technical Report* is accepted by **Visual Intelligence**
- *2024.12*: &nbsp;🎉🎉 One *Review* on AI in wearable diabetes management is accepted by **Advanced Intelligent Systems**
- *2024.05*: &nbsp;🎉🎉 One paper for snn security on rram is accepted by **ICCAD'24**! All the codes are opensourced now!
- *2024.04*: &nbsp;🎉🎉 One paper for post-training binary quantization of LLMs is accepted by **ICML'24**! All the codes are opensourced now!

# 💬 Invited Talks and Report

- *2025.07*: Our *Scaling RL to Long Videos*  was reported by **机器之心**. Please see the [link](https://www.jiqizhixin.com/articles/2025-07-14-2).
- *2025.06*: **AI-Time** online talk on *VideoEspresso*. Please see the [video](https://www.bilibili.com/video/BV1Yr7Hz1EKi?spm_id_from=333.1387.homepage.video_card.click).
- *2024.05*: *BiLLM* was reported by **IEEE Spectrum**. Thanks to [Matthew](https://www.newyorker.com/contributors/matthew-hutson) for the interview and report. Please see the [link](https://spectrum.ieee.org/1-bit-llm). 
- *2024.05*: **AI-Time** online talk on *BiLLM*. Please see the [video](https://www.bilibili.com/video/BV1XM4m1z7RU/?share_source=copy_web&vd_source=c680cccdae8e0fd2e453769e2e789b78). 
- *2024.04*: Our emperical study *How Good Are Low-bit Quantized LLaMA3 Models? An Empirical Study* (new version: *[An Empirical Study of LLaMA3 Quantization: From LLMs to MLLMs](https://arxiv.org/abs/2404.14047)*) was reported by **QbitAI (量子位)**. Please see the [link](https://m.thepaper.cn/newsDetail_forward_27189727).
- *2024.03*: Our *BiLLM: Pushing the Limit of Post-Training Quantization for LLMs*  was reported by **QbitAI (量子位)**. Please see the [link](https://www.qbitai.com/2024/06/152191.html).
  
# 📝 Publications

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICML 2025</div><img src='https://github.com/Aaronhuang-778/SliM-LLM/raw/main/imgs/WX20240527-155305%402x.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**SliM-LLM: Salience-Driven Mixed-Precision Quantization for Large Language Models**](https://arxiv.org/abs/2405.14917) <img src='https://img.shields.io/github/stars/Aaronhuang-778/SliM-LLM.svg?style=social&label=Star' alt="sym" height="100%">

**Wei Huang**, Haotong Qin, Yangdong Liu, Yawei Li, Qinshuo Liu, Xianglong Liu, Luca Benini, Michele Magno, Shiming Zhang, Xiaojuan Qi

- A novel scheme that observes and proves the structure-clustering of salient elements in LLMs weight matrix.
- The first group-wise mixed-precision quantization framework for LLMs.
- Serve as a plug-and-play approach to GPTQ/Omniquant/..., improving the inference-friendly method under low-bit quantization.

<div style="display: inline">
    <a href="https://arxiv.org/abs/2405.14917"> <strong>[paper]</strong></a>
    <a href="https://github.com/Aaronhuang-778/SliM-LLM"> <strong>[code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> Large language models (LLMs) achieve remarkable performance in natural language understanding but require substantial computation and memory resources. Post-training quantization (PTQ) is a powerful compression technique extensively investigated in LLMs. However, existing PTQ methods are still not ideal in terms of accuracy and efficiency, especially with below 4 bit-widths. Standard PTQ methods using group-wise quantization suffer difficulties in quantizing LLMs accurately to such low-bit, but advanced methods remaining high-precision weights element-wisely are hard to realize their theoretical hardware efficiency. This paper presents a Salience-Driven Mixed-Precision Quantization scheme for LLMs, namely SliM-LLM. The scheme exploits the salience distribution of weights to determine optimal bit-width and quantizers for accurate LLM quantization, while aligning bit-width partition to groups for compact memory usage and fast integer inference. Specifically, the proposed SliM-LLM mainly relies on two novel techniques: (1) Salience-Determined Bit Allocation utilizes the clustering characteristics of salience distribution to allocate the bit-widths of each group, increasing the accuracy of quantized LLMs and maintaining the inference efficiency; (2) Salience-Weighted Quantizer Calibration optimizes the parameters of the quantizer by considering the element-wise salience within the group, balancing the maintenance of salient information and minimization of errors. Comprehensive experiments show that SliM-LLM significantly improves the accuracy of LLMs at ultra-low bits, e.g., 2-bit LLaMA-7B achieves a 5.5-times memory-saving than original model on NVIDIA A800 GPUs, and 48% decrease of perplexity compared to the state-of-the-art gradient-free PTQ method. Moreover, SliM-LLM+, which is integrated from the extension of SliM-LLM with gradient-based quantizers, further reduces perplexity by 35.1%. </p>
    </div>
</div>

</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">CVPR 2025 <span style="color:red">Oral</span></div><img src='https://i.postimg.cc/LXzVcgFP/Wechat-IMG197.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection**](https://arxiv.org/pdf/2411.14794) <img src='https://img.shields.io/github/stars/hshjerry/VideoEspresso.svg?style=social&label=Star' alt="sym" height="100%">

Songhao Han, **Wei Huang**, Hairong Shi, Le Zhuo, Xiu Su, Shifeng Zhang, Xu Zhou, Xiaojuan Qi, Yue Liao, Si Liu

- A novel dataset designed to enhance video reasoning by addressing the limitations of existing datasets in terms of scale and granularity.
- We proposed a  Hybrid LVLMs Collaboration framework achieving cost-effective and accurate video reasoning, outperforming baseline models on the majority of tasks across our proposed benchmark.
- VideoEspresso sets a new starting point in video reasoning, offering rich annotations that facilitate advanced multimodal understanding.

<div style="display: inline">
    <a href="https://arxiv.org/pdf/2411.14794"> <strong>[paper]</strong></a>
    <a href="https://github.com/hshjerry/VideoEspresso"> <strong>[code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
    </div>
</div>

</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2025</div><img src='https://github.com/Aaronhuang-778/MC-MoE/raw/main/imgs/WX20241009-191322%402x.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More**](https://arxiv.org/abs/2410.06270) <img src='https://img.shields.io/github/stars/Aaronhuang-778/MC-MoE.svg?style=social&label=Star' alt="sym" height="100%">

**Wei Huang**, Yue Liao, Jianhui Liu, Ruifei He, Haoru Tan, Shiming Zhang, Hongsheng Li, Si Liu, Xiaojuan Qi

- MC-MoE for accurate weight-only quantization (Weight=1.5～2.5bit).
- MC-MoE for efficient online dynamic pruning (additional compression ratio > 10%)
- MC-MoE integrates static quantization and dynamic pruning to collaboratively achieve extreme compression for MoE-LLMs with less accuracy loss, ensuring an optimal trade-off between performance and efficiency.
- For instance, at 2.54 bits, MC-MoE compresses 76.6% of the model, with only a 3.8% average accuracy loss. During dynamic inference, we further reduce activated parameters by 15%, with a performance drop of less than 0.6%.

<div style="display: inline">
    <a href="https://arxiv.org/abs/2410.06270"> <strong>[paper]</strong></a>
    <a href="https://github.com/Aaronhuang-778/MC-MoE"> <strong>[code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> Mixture-of-Experts large language models (MoE-LLMs) marks a significant step forward of language models, however, they encounter two critical challenges in practice: 1) expert parameters lead to considerable memory consumption and loading latency; and 2) the current activated experts are redundant, as many tokens may only require a single expert. Motivated by these issues, we investigate the MoE-LLMs and make two key observations: a) different experts exhibit varying behaviors on activation reconstruction error, routing scores, and activated frequencies, highlighting their differing importance, and b) not all tokens are equally important -- only a small subset is critical. Building on these insights, we propose MC-MoE, a training-free Mixture-Compressor for MoE-LLMs, which leverages the significance of both experts and tokens to achieve an extreme compression. First, to mitigate storage and loading overheads, we introduce Pre-Loading Mixed-Precision Quantization, which formulates the adaptive bit-width allocation as a Linear Programming problem, where the objective function balances multi-factors reflecting the importance of each expert. Additionally, we develop Online Dynamic Pruning, which identifies important tokens to retain and dynamically select activated experts for other tokens during inference to optimize efficiency while maintaining performance. Our MC-MoE integrates static quantization and dynamic pruning to collaboratively achieve extreme compression for MoE-LLMs with less accuracy loss, ensuring an optimal trade-off between performance and efficiency. Extensive experiments confirm the effectiveness of our approach. For instance, at 2.54 bits, MC-MoE compresses 76.6% of the model, with only a 3.8% average accuracy loss. During dynamic inference, we further reduce activated parameters by 15%, with a performance drop of less than 0.6%. </p>
    </div>
</div>

</div>
</div>



<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2025</div><img src='https://github.com/Aaronhuang-778/AaronHuang-778.github.io/raw/main/images/informax.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**Data Pruning by Information Maximization**](https://openreview.net/forum?id=93XT0lKOct)

Haoru Tan, Sitong Wu, **Wei Huang**, Shizhen Zhao, Xiaojuan Qi

- A new coreset algorithm designed to maximize overall information by accounting for each sample’s individual contribution while reducing information overlap, with a simultaneous focus on maintaining diversity and importance.
- An efficient gradient-based solver enhanced by sparsification techniques and dataset partitioning strategies to make InfoMax scale to large-scale datasets.
- InfoMax exhibits the best performance and consistently outperforms the state-of-the-art schemes in a series of tasks, including image classification, an vision-language pre-training, large language model supervised fine-tuning experiments.

<div style="display: inline">
    <a href="https://openreview.net/forum?id=93XT0lKOct"> <strong>[paper]</strong></a>
    <a href=""> <strong>[code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
    </div>
</div>

</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2025</div><img src='https://github.com/Aaronhuang-778/AaronHuang-778.github.io/raw/main/images/layers.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**From Layers to States: A State Space Model Perspective to Deep Neural Network Layer Dynamics**](https://arxiv.org/pdf/2502.10463)

Qinshuo Liu, Weiqin Zhao, **Wei Huang**, Yanwen Fang, Lequan Yu, Guodong Li

- For a deep neural network, we treat the outputs from layers as states of a continuous process and attempt to leverage the SSM to design the aggregation of layers. To our best knowledge, this is the first time such a perspective has been presented. 
- This leads to a proposed lightweight module, the Selective State Space Model Layer Aggregation (S6LA) module, and it conceptualizes a neural network as a selective state space model(S6), hence solving the layer interactions by the long sequence modelling selective mechanism. 
- Compared with other SOTA convolutional and transformer-based layer aggregation models, S6LA demonstrates superior performance in classification, detection, and instance segmentation tasks.

<div style="display: inline">
    <a href="https://arxiv.org/pdf/2502.10463"> <strong>[paper]</strong></a>
    <a href=""> <strong>[code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
    </div>
</div>

</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICCAD 2024</div><img src='https://github.com/u3556440/SNNGX_qSNN_encryption/raw/main/_img_src/SNNGX_cover.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**SNNGX: Securing Spiking Neural Networks with Genetic XOR Encryption on RRAM-based Neuromorphic Accelerator**](https://arxiv.org/abs/2407.15152) <img src='https://img.shields.io/github/stars/u3556440/SNNGX_qSNN_encryption.svg?style=social&label=Star' alt="sym" height="100%">

Kwunhang Wong*, Songqi Wang*, **Wei Huang**, Xinyuan Zhang, Yangu He, Karl M.H. Lai, Yuzhong Jiao, Ning Lin, Xiaojuan Qi, Xiaoming Chen, Zhongrui Wang

- The first IP protection scheme specifically for SNNs, leveraging a genetic algorithm combined with classic XOR encryption to secure the networks against unauthorized access and tampering.
- A flexible solution for securing SNNs across various applications, especially in critical domains like biomedical applications where model security is paramount..

<div style="display: inline">
    <a href="https://arxiv.org/abs/2407.15152"> <strong>[paper]</strong></a>
    <a href="https://github.com/u3556440/SNNGX_qSNN_encryption"> <strong>[code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> Biologically plausible Spiking Neural Networks (SNNs), characterized by spike sparsity, are growing tremendous attention over intellectual edge devices and critical bio-medical applications as compared to artificial neural networks (ANNs). However, there is a considerable risk from malicious attempts to extract white-box information (i.e., weights) from SNNs, as attackers could exploit well-trained SNNs for profit and white-box adversarial concerns. There is a dire need for intellectual property (IP) protective measures. In this paper, we present a novel secure software-hardware co-designed RRAM-based neuromorphic accelerator for protecting the IP of SNNs. Software-wise, we design a tailored genetic algorithm with classic XOR encryption to target the least number of weights that need encryption. From a hardware perspective, we develop a low-energy decryption module, meticulously designed to provide zero decryption latency. Extensive results from various datasets, including NMNIST, DVSGesture, EEGMMIDB, Braille Letter, and SHD, demonstrate that our proposed method effectively secures SNNs by encrypting a minimal fraction of stealthy weights, only 0.00005% to 0.016% weight bits. Additionally, it achieves a substantial reduction in energy consumption, ranging from x59 to x6780, and significantly lowers decryption latency, ranging from x175 to x4250. Moreover, our method requires as little as one sample per class in dataset for encryption and addresses hessian/gradient-based search insensitive problems. This strategy offers a highly efficient and flexible solution for securing SNNs in diverse applications.</p>
    </div>
</div>

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Visual Intelligence</div><img src='https://github.com/Macaronlin/LLaMA3-Quantization/raw/master/images/overview.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**An Empirical Study of LLaMA3 Quantization: From LLMs to MLLMs**](https://link.springer.com/article/10.1007/s44267-024-00070-x) <img src='https://img.shields.io/github/stars/Macaronlin/LLaMA3-Quantization.svg?style=social&label=Star' alt="sym" height="100%">

**Wei Huang**, Xingyu Zheng, Xudong Ma, Haotong Qin, Chengtao Lv, Hong Chen, Jie Luo, Xiaojuan Qi, Xianglong Liu, Michele Magno

- Explore the performance of LLaMA3 series models under existing post-training quantization and LoRA-finetuning methods.
- Point out the significant performance loss of MLLMs based on LLaMA3 under low-bit post-training quantization.
- Highlights the significant performance gap under low bit-width that needs to be bridged in future developments.

<div style="display: inline">
    <a href="https://arxiv.org/abs/2404.14047"> <strong>[paper]</strong></a>
    <a href="https://github.com/Macaronlin/LLaMA3-Quantization"> <strong>[code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> The LLaMA family has become one of the most powerful open-source Large Language Models (LLMs) and the popular LLM backbones of Multimodal Large Language Models (MLLMs), widely applied in Computer Vision (CV) and Natural Language Understanding (NLU) tasks. Notably, LLaMA3 models have recently been released and achieve impressive performance across various with super-large scale pre-training on over 15T tokens of data. Given the wide application of low-bit quantization for LLMs in resource-limited scenarios, we explore LLaMA3's capabilities when quantized to low bit-width. This exploration can potentially unveil new insights and challenges for low-bit quantization of LLaMA3 and other forthcoming LLMs, especially in addressing performance degradation problems that suffer in LLM compression. Specifically, we comprehensively evaluate the 10 existing post-training quantization and LoRA-finetuning methods of LLaMA3 on 1-8 bits and diverse datasets to reveal LLaMA3's low-bit quantization performance. To uncover the capabilities of low-bit quantized MLLM, we assessed the performance of the LLaMA3-based LLaVA-Next-8B model under 2-4 ultra-low bits with post-training quantization methods. Our experimental results indicate that LLaMA3 still suffers non-negligent degradation in linguistic and visual contexts, particularly under ultra-low bit widths. This highlights the significant performance gap under low bit-width that needs to be bridged in future developments. We expect that this empirical study will prove valuable in advancing future models, driving LLMs and MLLMs to achieve higher accuracy at lower bit to enhance practicality. </p>
    </div>
</div>

</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICML 2024</div><img src='https://github.com/Aaronhuang-778/BiLLM/raw/main/imgs/main.png?raw=true' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**BiLLM: Pushing the Limit of Post-Training Quantization for LLMs**](https://arxiv.org/abs/2402.04291) <img src='https://img.shields.io/github/stars/Aaronhuang-778/BiLLM.svg?style=social&label=Star' alt="sym" height="100%">

**Wei Huang**, Yangdong Liu, Haotong Qin, Ying Li, Shiming Zhang, Xianglong Liu, Michele Magno, Xiaojuan Qi

- Compress LLM weights to as low as 1.08-1.1 bit and exceeds the performance of previous quantization methods at 2-bit or even 3-bit.
- Implements high-performance binary LLM in PTQ mode, efficiently achieving 1bit LLM compression without additional training and backpropagation.

<div style="display: inline">
    <a href="https://arxiv.org/abs/2402.04291"> <strong>[paper]</strong></a>
    <a href="https://github.com/Aaronhuang-778/BiLLM"> <strong>[code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> Pretrained large language models (LLMs) exhibit exceptional general language processing capabilities but come with significant demands on memory and computational resources. As a powerful compression technology, binarization can extremely reduce model weights to a mere 1 bit, lowering the expensive computation and memory requirements. However, existing quantization techniques fall short of maintaining LLM performance under ultra-low bit-widths. In response to this challenge, we present BiLLM, a groundbreaking 1-bit post-training quantization scheme tailored for pretrained LLMs. Based on the weight distribution of LLMs, BiLLM first identifies and structurally selects salient weights, and minimizes the compression loss through an effective binary residual approximation strategy. Moreover, considering the bell-shaped distribution of the non-salient weights, we propose an optimal splitting search to group and binarize them accurately. BiLLM achieving for the first time high-accuracy inference (e.g. 8.41 perplexity on LLaMA2-70B) with only 1.08-bit weights across various LLMs families and evaluation metrics, outperforms SOTA quantization methods of LLM by significant margins. Moreover, BiLLM enables the binarization process of the LLM with 7 billion weights within 0.5 hours on a single GPU, demonstrating satisfactory time efficiency. </p>
    </div>
</div>

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Arxiv</div><img src='https://github.com/Aaronhuang-778/SliM-LLM/raw/main/imgs/ohq.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**On-Chip Hardware-Aware Quantization for Mixed Precision Neural Networks**](https://arxiv.org/abs/2309.01945) 

**Wei Huang**, Haotong Qin, Yangdong Liu, Jingzhuo Liang, Yulun Zhang, Ying Li, Xianglong Liu

- Combine IP-core-level chip runtime clock and power awareness with network sensitivity, achieving a better balance of computational efficiency and accuracy on edge devices.
- Allow target networks to be compressed and deployed with high accuracy on edge chips with limited computational resources and ultra-low power consumption.
- Efficiently perform online quantization and optimization without additional devices or data access.

<div style="display: inline">
    <a href="https://arxiv.org/abs/2309.01945"> <strong>[paper]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> Low-bit quantization emerges as one of the most promising compression approaches for deploying deep neural networks on edge devices. Mixed-precision quantization leverages a mixture of bit-widths to unleash the accuracy and efficiency potential of quantized models. However, existing mixed-precision quantization methods rely on simulations in high-performance devices to achieve accuracy and efficiency trade-offs in immense search spaces. This leads to a non-negligible gap between the estimated efficiency metrics and the actual hardware that makes quantized models far away from the optimal accuracy and efficiency, and also causes the quantization process to rely on additional high-performance devices. In this paper, we propose an On-Chip Hardware-Aware Quantization (OHQ) framework, performing hardware-aware mixed-precision quantization on deployed edge devices to achieve accurate and efficient computing. Specifically, for efficiency metrics, we built an On-Chip Quantization Aware pipeline, which allows the quantization process to perceive the actual hardware efficiency of the quantization operator and avoid optimization errors caused by inaccurate simulation. For accuracy metrics, we propose Mask-Guided Quantization Estimation technology to effectively estimate the accuracy impact of operators in the on-chip scenario, getting rid of the dependence of the quantization process on high computing power. By synthesizing insights from quantized models and hardware through linear optimization, we can obtain optimized bit-width configurations to achieve outstanding performance on accuracy and efficiency. We evaluate inference accuracy and acceleration with quantization for various architectures and compression ratios on hardware. OHQ achieves 70% and 73% accuracy for ResNet-18 and MobileNetV3, respectively, and can reduce latency by 15~30% compared to INT8 on real deployment. </p>
    </div>
</div>

</div>
</div>



# 📖 Educations
- *2023.09 - (now)*, Ph.D. Student in Department of Electrical Electronic Engineering, The University of HongKong.
- *2019.09 - 2023.06*, B.Eng. in Computer Science, School of Computer Science and Engineering, Beihang University.
  

# 🗒️ Academic Services

- Conference Reviewer: ICLR, Neurips, ICML, ECCV, AISTATS, ICCV
- Journal Reviewer: Neural Networks. 
- Program Committee member for Practical Deep Learning Workshop, IEEE CAI 2024.


# 🎖 Honors and Awards 
 
-2023 Outstanding Graduate, Beihang University.
  
-2023 Outstanding Project of the 16th National College Student Innovation and Entrepreneurship Competition, China.

-2022 Outstanding Project of the 15th National College Student Innovation and Entrepreneurship Competition, China.



# 💻 Internships & Teaching Services
- *2025.06 - Now*, Multimodal Large Language Model Intern, NVIDIA.
- *2022.09 - 2023.01*, AI algorithm internship on model inference acceleration, [Enflame](https://www.linkedin.com/company/enflame/), China.
- *2022.08 - 2023.01*, TA for **Frontiers in Artificial Intelligence**, Beihang University.
- *2022.08 - 2023.01*, TA for **Computer Hardware Basics**, the head of TA team, Beihang University.
- *2021.08 - 2022.01*, TA for **Computer Hardware Basics**, the head of TA team, Beihang University.
- *2021.03 - 2021.06*, TA for **Discrete Mathematics**, the head of TA team, Beihang University.
