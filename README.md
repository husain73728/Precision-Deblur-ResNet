# Precision-Deblur-Framework-using-a-ResNet

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)

An advanced deep learning framework designed to invert heavy non-linear motion blur degradation ($35\text{px}$ kernel diameter) in high-resolution images. Built around a highly optimized 4-stage feature-collapsing residual network design, this repository implements a custom composite structural-perceptual loss mechanism to achieve high-fidelity image restoration without artifact generation or structural hallucination.

## 🚀 Key Achievements
* **Performance Gain:** Achieved a **+3.30 dB PSNR increase** over baseline, peaking at **27.30 dB Validation PSNR** and **0.6619 SSIM**.
* **Zero Halos / Artifacts:** Engineered with a specialized **Zero-Initialization Tail** to guarantee smooth, non-saturated convergence and prevent hallucinated textures.
* **Mathematical Stability:** Outfitted with an adaptive **Cosine Annealing Warm Restarts (SGDR)** learning rate scheduler to seamlessly climb out of local optimization minima.

---

## 🏛️ System Architecture

Traditional convolutional networks struggle with heavy motion blurs due to constrained local receptive fields. This framework introduces a **4-Stage Spatial Downsampling Down-Up Hierarchy** built using deep residual structures (`DeblurResNet`) to address this bottleneck.

## 🏛️ System Architecture

Traditional convolutional networks struggle with heavy motion blurs due to constrained local receptive fields. This framework introduces a **4-Stage Spatial Downsampling Down-Up Hierarchy** built using deep residual structures (`DeblurResNet`) to address this bottleneck:

```text
       [ Blurred Image Input: Shape (B, 3, H, W) ]
                            │
                            ▼
 ┌─────────────────────────────────────────────────────┐
 │            Dynamic Padding Layer (infer.py)         │
 │   - Monitors incoming tensor spatial dimensions    │
 │   - Applies reflection pad to be divisible by 8    │
 └─────────────────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────┐
 │         4-Stage Downsampling Grid (model.py)        │
 │   - Collapses spatial resolution by 1/8th scale     │
 │   - Expands receptive field across 35px blur path   │
 └─────────────────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────┐
 │       Deep Residual Processing Loop (model.py)      │
 │   - Multi-layer Bottleneck ResNet Feature Extraction │
 │   - Configured with specialized Zero-Init Tail      │
 └─────────────────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────┐
 │       Multi-Objective Loss Boundary Evaluation     │
 │   - Charbonnier + SSIM + Laplacian + VGG Loss       │
 └─────────────────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────┐
 │         Dynamic Slicing pass (infer.py)             │
 │   - Crops out pad vectors post-inference            │
 │   - Guarantees exact original pixel resolution      │
 └─────────────────────────────────────────────────────┘
                            │
                            ▼
       [ Restored Sharp Output (PSNR: 27.30 dB) ]
