# Intraoperative US to MRI Synthesis

基于 PyTorch 的脑肿瘤术中超声到术中 MRI 跨模态合成项目。

- 输入：`USpreimri`（术中超声）+ `3DAXT1postcontrast`（术前 MRI 结构先验）
- 输出：`Intraop 2DAXT2BLADE`（术中 MRI）
- 框架：`Dual EDM + correlation loss`（Karras et al., 2022）
- 数据集：`ReMIND`

## 项目目标

本项目面向 ReMIND 数据集中的术中场景，通过术中超声和术前 MRI 生成术中 MRI。术前 MRI 提供高质量解剖结构先验，术中超声提供实时手术视野，两者联合生成术中 MRI。

## 核心方法：Dual EDM

采用 EDM（Elucidating the Design Space of Diffusion-Based Generative Models, Karras et al., 2022）替代 DDPM，主要改进：

- **连续噪声调度**：用连续 σ 替代离散时间步，σ 从 log-normal 分布采样
- **预条件机制**：c_skip, c_in, c_out, c_noise 稳定训练
- **高效采样**：Heun's 2阶采样器，18步即可生成高质量图像（vs DDPM 1000步）
- **EDM加权损失**：(σ² + σ_data²) / (σ · σ_data)² 加权去噪损失

## 当前实现

- `preprocess_remind.py`
  - 从 `metadata.csv` 自动筛选 `USpreimri -> Intraop 2DAXT2BLADE` + 术前 `3DAXT1postcontrast`
  - 以术中 MRI 为空间参考
  - 对 US 和术前 MRI 分别做刚性配准并重采样到术中 MRI 网格
  - 裁剪重叠 ROI
  - 输出 `us.npy`、`mr.npy`、`preop_mr.npy`、`overlap_mask.npy`、`meta.json`
- `dataset.py`
  - 读取预处理后的 3D 体数据
  - 进行 2.5D 切片采样
  - 只优先抽取有足够重叠的有效切片
  - 返回 US、MR、术前 MR 和 overlap mask
- `model.py`
  - `EDMPreconditioning`：EDM 预条件系数和 σ 采样
  - `EDMUS`：超声自编码去噪（3ch 输入/输出）
  - `EDMMR`：MRI 条件生成（5ch 输入 = c_in·x_σ + US + preop MR，1ch 输出）
  - `DualEDMCorrelationModel`：双 EDM + 相关性损失联合训练
  - Heun's 2阶采样器 + 随机噪声注入（stochastic sampler）
- `loss.py`
  - EDM 加权去噪损失
  - 基于皮尔逊相关系数的跨模态相关性损失
  - 相关性损失只在 `overlap_mask` 区域内计算
- `train.py`
  - 联合训练双 EDM
  - 使用 `AdamW`（lr=1e-4）
  - 支持 EDM 预条件和 σ 采样
- `sample.py`
  - Heun's 2阶采样（默认18步）
  - 支持随机噪声注入参数调节（s_churn, s_tmin, s_tmax, s_noise）
  - 支持术前 MRI 作为结构先验

## 模型架构

```
EDM_US (自编码去噪)            EDM_MR (条件生成)
┌──────────────┐              ┌──────────────┐
│ US (3ch)     │              │ c_in·x_σ(1ch)│
│   ↓          │              │ US cond(3ch) │
│ UNet Encoder │──features──→ │ Preop MR(1ch)│
│   ↓          │              │   ↓          │
│ UNet Decoder │              │ UNet Encoder │
│   ↓          │              │   + fusion   │
│ US F_theta   │              │   ↓          │
└──────────────┘              │ UNet Decoder │
                              │   ↓          │
  EDM预条件:                   │ MR F_theta   │
  x0 = c_skip·x_σ + c_out·F  └──────────────┘
```

## EDM vs DDPM 对比

| | DDPM | EDM |
|---|---|---|
| 噪声调度 | 离散时间步，线性/余弦调度 | 连续 σ，log-normal 采样 |
| 网络预条件 | 无 | c_skip, c_in, c_out, c_noise |
| 采样步数 | 1000步（DDIM可降至100步） | 18步（Heun's 2阶） |
| 训练稳定性 | 一般 | 更好（预条件归一化输出范围） |
| 数学框架 | 离散马尔可夫链 | 连续SDE |

## 推荐目录

```text
us-mri/
├─ preprocess_remind.py
├─ dataset.py
├─ model.py
├─ loss.py
├─ train.py
├─ sample.py
├─ README.md
└─ .gitignore
```

## 环境依赖

- Python 3.10+
- torch
- numpy
- SimpleITK

```bash
pip install torch numpy SimpleITK
```

## 数据准备

确保本地目录中包含：

- `metadata.csv`
- ReMIND 原始 DICOM 数据

当前脚本会从 `metadata.csv` 中自动筛选以下病例对子：

- `Study Description = Intraop`
- `US Series = USpreimri`
- `MR Series = 2DAXT2BLADE`
- `Preop MR Series = 3DAXT1postcontrast`

## 预处理

```bash
python preprocess_remind.py ^
  --dataset-root "path\to\dataset" ^
  --metadata-csv "path\to\metadata.csv" ^
  --output-root "path\to\preprocessed" ^
  --overwrite
```

## 训练

```bash
python train.py ^
  --data-root "path\to\preprocessed" ^
  --save-dir "path\to\checkpoints" ^
  --epochs 100 ^
  --batch-size 4 ^
  --lr 1e-4 ^
  --sigma-data 0.5 ^
  --sigma-min 0.002 ^
  --sigma-max 80.0 ^
  --lambda-corr 0.1
```

## 采样

```bash
python sample.py ^
  --checkpoint "path\to\checkpoints\best.pt" ^
  --preprocessed-root "path\to\preprocessed" ^
  --subject-id ReMIND-001 ^
  --slice-index 32 ^
  --num-steps 18 ^
  --s-churn 40.0 ^
  --output-dir "path\to\samples"
```

采样参数说明：

- `--num-steps`：采样步数，默认18，越多越精细但越慢
- `--s-churn`：随机噪声注入强度，0=确定性采样，40=默认随机采样
- `--s-tmin / --s-tmax`：噪声注入的 σ 范围
- `--s-noise`：噪声注入缩放因子

## GitHub

- https://github.com/xiaoqi983/us-mri.git
