# WaveUNet: An Efficient Deep Network Based on Discrete Wavelet Transform and Attention Mechanism for InSAR Phase Unwrapping

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📖 Introduction

WaveUNet is a novel deep learning framework for InSAR phase unwrapping, combining **Discrete Wavelet Transform (DWT)** with **Attention Mechanisms** to achieve state-of-the-art performance. This repository contains the official implementation of our paper.

<img src="image/image2.png" width="800" alt="Flowchart of this study">
*Figure 1: Overall architecture of the proposed WaveUNet framework*

## ✨ Key Features

- **Wavelet-based Multi-scale Analysis**: Leverages DWT to capture both global and local phase patterns
- **Attention Mechanism**: Enhances feature representation in challenging regions (steep slopes, discontinuities)
- **End-to-End Learning**: Direct mapping from wrapped to unwrapped phase
- **Efficient Architecture**: Balanced trade-off between accuracy and computational cost

## 🏗️ Model Architecture

The WaveUNet consists of three main components:
1. **Encoder**: Hierarchical feature extraction with DWT-based downsampling
2. **Bottleneck**: Attention-augmented feature transformation
3. **Decoder**: Progressive upsampling with skip connections

## 📋 Requirements

```bash
Python >= 3.8
PyTorch >= 1.11.0
CUDA >= 11.0 (optional, for GPU acceleration)
