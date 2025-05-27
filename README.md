# VLM-3R: Vision-Language Models Augmented with Instruction-Aligned 3D Reconstruction

**VLM-3R is a unified Vision-Language Model (VLM) framework integrating 3D reconstructive instruction tuning for deep spatial understanding from monocular video.**

The rapid advancement of Large Multimodal Models (LMMs) for 2D images and videos has motivated extending these models to understand 3D scenes, aiming for human-like visual-spatial intelligence. VLM-3R processes monocular video frames by employing a geometry encoder to derive implicit 3D tokens that represent spatial understanding. Through the utilization of Spatial-Visual–View Fusion technique and over 200K curated 3D reconstructive instruction tuning question-answer (QA) pairs, VLM-3R effectively aligns real-world spatial context with language instructions. This enables the model to perform monocular 3D spatial assistance and embodied reasoning.

[**Paper (arXiv)**](https://arxiv.org/abs/2403.xxxxx) **|** [**Project Page**](https://vlm-3r.github.io/) **|** [**Code (GitHub)**](https://github.com/VITA-Group/VLM-3R) **|** **Datasets & Benchmarks (Coming Soon)**

## 🧑‍💻 Authors

            <h3>Authors</h3>
            <p>
                <a href="https://zhiwenfan.github.io/" target="_blank">Zhiwen Fan</a><sup>1&dagger;*</sup>,
                <a href="https://jian-zhang-3dv.github.io/Jian-Zhang-3DV/" target="_blank">Jian Zhang</a><sup>2*</sup>,
                <a href="https://shadowiterator.github.io/" target="_blank">Renjie Li</a><sup>3</sup>,
                <a href="https://andy-zd.github.io/" target="_blank">Junge Zhang</a><sup>4</sup>,
                <a href="https://chenrunjin.github.io/" target="_blank">Runjin Chen</a><sup>1</sup>,
                <a href="https://alexhu.top/" target="_blank">Hezhen Hu</a><sup>1</sup>,
                <a href="https://www.kevin-ai.com/" target="_blank">Kevin Wang</a><sup>1</sup>,
                <a href="https://sites.google.com/view/qhz991029" target="_blank">Huaizhi Qu</a><sup>5</sup>,
                <a href="https://wdilin.github.io/" target="_blank">Dilin Wang</a><sup>6</sup>,
                <a href="https://sites.google.com/view/zhicheng-yan" target="_blank">Zhicheng Yan</a><sup>6</sup>,
                <a href="https://hyxu2006.github.io/" target="_blank">Hongyu Xu</a><sup>6</sup>,
                <a href="https://www.linkedin.com/in/justin-d-theiss" target="_blank">Justin Theiss</a><sup>6</sup>,
                <a href="https://tianlong-chen.github.io/" target="_blank">Tianlong Chen</a><sup>5</sup>,
                <a href="https://jiachenli94.github.io/" target="_blank">Jiachen Li</a><sup>4</sup>,
                <a href="https://vztu.github.io/" target="_blank">Zhengzhong Tu</a><sup>3</sup>,
                <a href="https://vita-group.github.io/research.html" target="_blank">Zhangyang Wang</a><sup>1</sup>,
                <a href="https://www.linkedin.com/in/rakesh-r-3848538" target="_blank">Rakesh Ranjan</a><sup>6</sup>
            </p>

¹UT Austin   ²XMU   ³TAMU   ⁴UCR   ⁵UNC   ⁶Meta

†Corresponding Author. \*Equal contribution.

(zhiwenfan@utexas.edu)

## Overview
![VLM-3R Project Overview](docs/images/teaser_00.jpg)

## 🚀 Key Innovations

- **End-to-End Monocular Video 3D Understanding:** VLM-3R directly processes monocular RGB videos without needing external depth sensors or pre-built 3D maps, significantly enhancing scalability and practical applicability.
- **3D Reconstructive Instruction Tuning:** Instruction tuning with over 200K QA pairs enables the model to effectively align visual information with 3D spatial context and language instructions.
- **Spatial-Visual-View Fusion:** A novel fusion mechanism integrates 3D geometric tokens, per-view camera tokens, and 2D appearance features for joint spatio-linguistic understanding.
- **Vision-Spatial-Temporal Intelligence Benchmark (VSTI-Bench):** A new benchmark with over 138.6K QA pairs, specifically designed to evaluate the model's understanding of spatio-temporal relationships evolving from camera motion within 3D environments.

## 🛠️ VLM-3R Architecture

The core of VLM-3R is a pre-trained Large Multimodal Model (LMM), integrated with modules for deriving geometric encodings, camera view encodings, and visual features from the input video; these diverse inputs are subsequently fused effectively with language representations. VLM-3R does not rely on pre-built 3D maps or external depth sensors. This design directly addresses key limitations of existing approaches, such as the common inadequacy of Video LLMs in perceiving rich spatial context from monocular video and the restrictive dependency of many specialized 3D-LLMs on prior 3D map or depth sensor inputs.

**Architecture Overview Diagram:**

[Video of VLM3R Network Architecture Demonstration](https://github.com/user-attachments/assets/f82f7905-879f-414a-a690-99fc471f2a50)

*Our method takes monocular video and language instruction as input. Visual Encoder coupled with Spatial Encoder extract frame-level appearance, camera view position, and globally aligned geometry. Visual-Geometry Fusion integrates these through attention and projection layers to create 3D-aware visual features for the LMM. During the inference stage, this fusion enables reliable spatial and temporal reasoning.*

**Key Components:**

- **3D Reconstructive Tokenization:** Utilizes the pre-trained CUT3R model to process monocular video frame-by-frame, extracting implicit latent representations (enriched feature tokens and camera view tokens). These tokens serve as rich 3D reconstructive tokens, compactly encoding observed 3D geometry and camera perspective without relying on explicit point clouds.

- **Spatial-Visual-View Fusion:** Employs a cross-attention mechanism where the VLM's native visual tokens (Hv) attend to a unified 3D representation (Z3D, formed by concatenated 3D feature tokens Ft′ and camera view tokens zt′). The output of this attention stage (Hattn) is then residually connected with the original visual tokens (Hv′=Hv+Hattn). This enriched representation Hv′ subsequently passes through a two-layer MLP projector for alignment with the LMM.

  ```
  Z_3D = Concat(F'_t, z'_t)
  H_attn = CrossAttention(Query: H_v, KeyValue: Z_3D)
  H'_v = H_v + H_attn
  ProjectedFeatures = MLP_2-layer(H'_v)
  ```

- **Training Objective & Fine-tuning Strategy:** Adopts the same learning objective as LLaVA-NeXT-Video. To achieve efficient adaptation, Low-Rank Adaptation (LoRA) is employed for fine-tuning, which involves updating parameters within the 3D fusion attention block and the projection layers.

## 📊 Datasets & Benchmarks

- **Multimodal Spatial Instruction Data Generation:** A scalable, automated data generation pipeline produced over **200,000** general question-answer pairs for spatial reasoning from monocular video, and **4,225** embodied route planning data instances generated using simulators. This data is derived from existing 3D datasets like ScanNet, ScanNet++, and ARKitScenes, processed via detailed spatio-temporal scene graphs to automatically generate QA pairs for tasks such as object counting, relative distance/direction, appearance order, object size, absolute distance, and room size.
- **Vision-Spatial-Temporal Intelligence Benchmark (VSTI-Bench):** Contains approximately **138,600** QA pairs, distributed across three main categories: Camera Dynamics (49.6%), Camera-Object Interactions (38.4%), and Object Relative Position (12.0%). It is designed to assess LMMs' ability to perceive and reason about relative camera/object motion, dynamic object-camera relationships, and evolving spatial configurations.

## ⚙️ Setup

### 1. Clone Repository and Submodules

```
git clone https://github.com/Jian-Zhang-3DV/VLM-3R.git
cd VLM-3R
git submodule update --init --recursive
```

### 2. Environment Setup

1. **Create conda environment:**

   ```
   conda create -n vlm3r python=3.10 -y
   conda activate vlm3r
   ```

2. **Install base packages:**

   ```
   pip install --upgrade pip
   conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
   ```

3. **Install project dependencies:**

   ```
   pip install -e ".[train]"
   # Note: The FlashAttention wheel URL might be specific. Consider verifying compatibility.
   pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   pip install decord openai accelerate==0.29.1
   ```

### 3. Install CUT3R

1. **Install requirements:**

   ```
   cd CUT3R
   pip install -r requirements.txt
   ```

2. **Build CUT3R extension:**

   ```
   cd src/croco/models/curope/
   python setup.py build_ext --inplace
   cd ../../../../ # Return to CUT3R root
   ```

3. **Download checkpoint:**

   ```
   cd src # Navigate to src within CUT3R
   pip install gdown
   gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
   cd ../.. # Return to VLM-3R root
   ```

## ▶️ Test Run

1. **Run Video Test Example:**

   ```
   CUDA_VISIBLE_DEVICES=0 bash scripts/video/demo/video_demo.sh \
       Journey9ni/vlm-3r-llava-qwen2-lora \
       qwen_1_5 32 2 average grid True \
       playground/demo/47334096.mp4 \
       lmms-lab/LLaVA-NeXT-Video-7B-Qwen2
   ```

   **Explanation:**

   - `CUDA_VISIBLE_DEVICES=0`: Specifies the GPU device number to use.
   - `Journey9ni/vlm-3r-llava-qwen2-lora`: Specifies the location of the model checkpoint.
   - `qwen_1_5`: Specifies the model version to use.
   - `32 2 average grid True`: These are parameter settings for model inference.
   - `playground/demo/47334096.mp4`: Specifies the path to the video file to be tested.
   - `lmms-lab/LLaVA-NeXT-Video-7B-Qwen2`: Specifies the base model path for the LoRA model.

## 📥 Model Weights

The model weights can be downloaded from Hugging Face:

```
# Download model weights from Hugging Face
git lfs install
git clone https://huggingface.co/Journey9ni/vlm-3r-llava-qwen2-lora
```

The model weights include:

- LoRA weight files
- Configuration files
- Other necessary model files

## 📝 TODO List

- [x] Release model weights and inference code
- [ ] Release training data, data generation scripts, and training scripts
- [ ] Evaluate on VSiBench
- [ ] Release VSTiBench data and evaluation code

## 🙏 Acknowledgements

We would like to express our gratitude to the following projects for their valuable contributions:

- [CUT3R](https://github.com/CUT3R/CUT3R): Provides the spatial feature encoder used in our model.
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT): Serves as the foundation for our codebase.
- [thinking-in-space](https://github.com/vision-x-nyu/thinking-in-space): Offers important evaluation methods for 3D understanding capabilities of VLM.
