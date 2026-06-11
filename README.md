# Intraoperative US to MRI Synthesis

基于 PyTorch 的脑肿瘤术中超声到术中 MRI 跨模态合成项目，当前任务设定固定为：

- 输入：`USpreimri` + `3DAXT1postcontrast`（术前 MRI 结构先验）
- 输出：`Intraop 2DAXT2BLADE`
- 框架：`Dual DDPM + correlation loss`
- 数据集：`ReMIND`

## 项目目标

本项目面向 `ReMIND` 数据集中的术中场景，重点研究通过术中超声和术前 MRI 生成术中 MRI 的可行性。术前 MRI 提供高质量解剖结构先验，术中超声提供实时手术视野，两者联合生成术中 MRI。

为降低目标域异质性，当前版本不混合多种术中 MRI 序列，而是只保留最稳定、样本量最大的目标序列 `2DAXT2BLADE`。

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
  - 实现双 DDPM 结构
  - `DDPM_US` 负责源域自编码去噪（3ch 输入/输出）
  - `DDPM_MR` 负责目标 MRI 条件生成（5ch 输入 = noisy MR + US + preop MR，1ch 输出）
  - 术前 MRI 作为额外条件通道拼接到 MR UNet 输入
- `loss.py`
  - 实现去噪损失
  - 实现基于皮尔逊相关系数的跨模态相关性损失
  - 相关性损失只在 `overlap_mask` 区域内计算
- `train.py`
  - 联合训练双 DDPM
  - 使用 `AdamW`
  - 支持术前 MRI 条件输入
- `sample.py`
  - 支持基于训练好的模型进行条件采样
  - 支持术前 MRI 作为结构先验

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

原始数据、预处理结果、checkpoint 不建议提交到 Git 仓库。

## 环境依赖

建议 Python 版本：

- `Python 3.10+`

主要依赖：

- `torch`
- `numpy`
- `SimpleITK`

可按需自行安装，例如：

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

运行示例：

```bash
python preprocess_remind.py ^
  --dataset-root "c:\Users\小小祁\Desktop\北工大医院项目\超声术中生成——分类整理后" ^
  --metadata-csv "c:\Users\小小祁\Desktop\北工大医院项目\超声术中生成——分类整理后\metadata.csv" ^
  --output-root "c:\Users\小小祁\Desktop\北工大医院项目\超声术中生成——分类整理后\preprocessed_intraop_t2blade" ^
  --overwrite
```

预处理输出：

- `us.npy`
- `mr.npy`
- `preop_mr.npy`
- `overlap_mask.npy`
- `meta.json`
- `preprocessed_pairs.csv`

## 训练

运行示例：

```bash
python train.py ^
  --data-root "c:\Users\小小祁\Desktop\北工大医院项目\超声术中生成——分类整理后\preprocessed_intraop_t2blade" ^
  --save-dir "c:\Users\小小祁\Desktop\北工大医院项目\超声术中生成——分类整理后\checkpoints_t2blade" ^
  --epochs 100 ^
  --batch-size 4 ^
  --lambda-corr 0.1
```

训练阶段会：

- 按病例划分训练集和验证集
- 对 US 和 MRI 做联合扩散训练
- 将相关性约束限制在有效重叠区域
- 术前 MRI 作为额外条件通道参与 MR 生成

## 采样

运行示例：

```bash
python sample.py ^
  --checkpoint "c:\Users\小小祁\Desktop\北工大医院项目\超声术中生成——分类整理后\checkpoints_t2blade\best.pt" ^
  --preprocessed-root "c:\Users\小小祁\Desktop\北工大医院项目\超声术中生成——分类整理后\preprocessed_intraop_t2blade" ^
  --subject-id ReMIND-001 ^
  --slice-index 32 ^
  --sampling ddim ^
  --ddim-steps 100 ^
  --output-dir "c:\Users\小小祁\Desktop\北工大医院项目\超声术中生成——分类整理后\samples"
```

## 模型架构

```
DDPM_US (自编码去噪)          DDPM_MR (条件生成)
┌──────────────┐              ┌──────────────┐
│ US (3ch)     │              │ Noisy MR(1ch)│
│   ↓          │              │ US cond(3ch) │
│ UNet Encoder │──features──→ │ Preop MR(1ch)│
│   ↓          │              │   ↓          │
│ UNet Decoder │              │ UNet Encoder │
│   ↓          │              │   + fusion   │
│ US recon(3ch)│              │   ↓          │
└──────────────┘              │ UNet Decoder │
                              │   ↓          │
                              │ MR pred(1ch) │
                              └──────────────┘
```

## 当前实验建议

针对当前数据集，推荐优先采用以下实验路线：

1. 固定任务为 `USpreimri + Preop 3DAXT1postcontrast -> Intraop 2DAXT2BLADE`
2. 先检查预处理后的配准与重叠区域是否合理
3. 再训练当前 DDIC 版本
4. 后续可增加 `UNet baseline` 作为对照实验

不建议第一版直接混合：

- `2DAXT2BLADE`
- `3DSAGT2FLAIR`
- `2DAXT2FLAIR`

否则目标域差异会过大，影响稳定训练。

## GitHub

当前仓库远端地址：

- [https://github.com/xiaoqi983/us-mri.git](https://github.com/xiaoqi983/us-mri.git)

如果本地已经完成 git 初始化，可直接推送：

```bash
git push -u origin main
```

如果推送失败，通常是以下原因之一：

- 本机网络无法访问 GitHub
- 需要代理或 VPN
- 需要切换为 SSH 方式推送

## 说明

本仓库当前重点是建立一个面向 `ReMIND` 的、可运行的术中超声到术中 MRI 合成管线。后续如果需要扩展，可继续补充：

- `requirements.txt`
- baseline 模型
- 评估脚本
- 可视化脚本
