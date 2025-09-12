# UICSurv: Uncertainty-aware Multimodal Survival Prediction for Rectal Cancer

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.10%2B-red.svg)](https://pytorch.org/)


This repository contains the official implementation of **UICSurv**, a novel multimodal survival prediction framework designed for rectal cancer prognosis.  
UICSurv enhances **robustness** and **interpretability** by addressing both **modality heterogeneity** and **site heterogeneity** through:

- Iterative cross-attention based multimodal fusion (backbone from **HEALNET**).  
- **Survival Contrastive Learning (SCL)** for semantic-preserving embedding.  
- **EvidenceHit**: uncertainty-aware survival prediction module with evidence learning.  

---

## 🖼️ Framework Overview

The overall framework of **UICSurv** is illustrated below:

<p align="center">
  <img src="Figs/fig2.png" alt="UICSurv Framework"/>
</p>


## 🚀 Key Features
- Multimodal survival prediction with **uncertainty quantification**.  
- **Cross-site evaluation** to ensure robustness and generalization.  
- Support for **Kaplan–Meier (KM) analysis** and statistical tests (paired *t*-test, log-rank test).  

---

