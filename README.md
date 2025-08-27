<div align="center">

<h1><a href="https://doi.org/10.1109/TGRS.2025.3581078">ACR-Net: Adaptive Correlation Refined Hyperspectral Unmixing</a></h1>

**IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2025**

<div align="center">
  <img src="resource/ACR_Net.png" alt="ACR-Net Overview" width="420"/>
</div>
</div>

## Share us a \:star: if you're interested in this repo

**Official PyTorch implementation** of the TGRS 2025 paper:
**“ACR-Net: Adaptive Correlation Refined Hyperspectral Unmixing.”**

---

## 🧾 Abstract

Hyperspectral unmixing (HU) aims to resolve the prevalent issue of mixed pixels in hyperspectral imagery and serves as an effective technique for subpixel-level image interpretation. Recent years have seen the emergence of advanced unmixing algorithms that integrate both spatial information and spectral information. However, existing methods mainly focus on spatial context and lack depth in modeling spatial correlations. Both relevant spatial information and irrelevant spatial information are introduced into the spectral mixing model for local pixels, with the irrelevant information acting as noise that impacts the unmixing accuracy. To address these challenges, we propose an advanced spectral mixing model, adaptive correlation refined network (ACR-Net) that integrates refined spatial correlation based on self-attention. A key component of ACR-Net is the adaptive correlation aggregated decoder (ACAD), which extracts affinity information from the encoder’s feature map and adaptively amplifies the influence of highly correlated regions in the unmixing process. Additionally, the composite active spatial attention (CASA) module emphasizes unique spectral characteristics across bands, improving spatial distribution representation and enabling more accurate abundance estimation. We conducted abundant evaluations on six datasets, including three real-world and two synthetic HU datasets as well as a large benchmark classification dataset. Extensive experiments have demonstrated that the proposed algorithm can achieve exceptional or comparable unmixing results among various state-of-the-art algorithms. Moreover, the proposed method achieved optimal classification performance on the classic PaviaU dataset, indicating its strong potential for hyperspectral classification task.

---

## 🧠 Algorithm (Brief)

* **Self-Attention–based Spatial Correlation Refinement:** filters out irrelevant neighborhood cues to reduce noise in the mixing model.
* **ACAD — Adaptive Correlation Aggregated Decoder:** mines affinity from encoder features and **amplifies** highly correlated regions during unmixing.
* **CASA — Composite Active Spatial Attention:** highlights band-wise spectral signatures to improve spatial representation and **abundance estimation**.

> Experiments span **six datasets** (three real, two synthetic HU datasets, and one large classification benchmark), with **state-of-the-art** or competitive unmixing results and **best** classification on **PaviaU**.

---

## 📣 News

* 🔜 **Code will be released soon.** (代码将很快 release)

---

## 📚 Citation

M. Hu, C. Wu, R. Liu and L. Zhang, “**ACR-Net: Adaptive Correlation Refined Hyperspectral Unmixing**,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 63, pp. 1–15, 2025, Art no. 5517315, doi: **10.1109/TGRS.2025.3581078**.

```bibtex
@article{Hu2025ACRNet,
  author  = {Hu, M. and Wu, C. and Liu, R. and Zhang, L.},
  title   = {ACR-Net: Adaptive Correlation Refined Hyperspectral Unmixing},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  volume  = {63},
  pages   = {1--15},
  year    = {2025},
  doi     = {10.1109/TGRS.2025.3581078},
  artnum  = {5517315}
}
```

---

> If you use this work, please cite the paper and consider leaving a ⭐ to support the project.
