# Polymer Carrier Mobility Prediction

基于 D-MPNN 与贝叶斯个性化排序 (BPR) 的聚合物载流子迁移率对比预测模型。给定一对聚合物，模型同时预测电子迁移率 (μ_e) 和空穴迁移率 (μ_h) 的相对高低。

## Pipeline 概览

```
MonomerA + MonomerB (含 * 标记)
        │
        ▼
  ① 单体拼接 (A+B → 重复单元，处理非对称单体)
        │
        ▼
  ② 重复单元环化 → 环状模型化合物 (标记连接点 CP)
        │
        ▼
  ③ chemprop v2 D-MPNN 编码 (多结构平均) + 额外特征拼接
        │
        ▼
  ④ BPR 排序损失 + Delta 回归损失
        │
        ▼
  ⑤ 多任务同时预测 μ_e / μ_h
```

## 模块说明

| 模块 | 说明 |
|------|------|
| `config.py` | 常量定义与配置 dataclass (`ModelConfig`, `TrainingConfig`, `FinetuneConfig`, `PredictConfig`) |
| `chemistry.py` | 化学预处理：SMILES 规范化、单体拼接、环化、额外特征提取、DataFrame 预处理 |
| `featurizer.py` | 自定义原子特征化器 (chemprop `MultiHotAtomFeaturizer` + CP 二值特征) 及工厂函数 |
| `dataset.py` | `PairDataset` / `CachedPairDataset` 数据集类及 collate 函数 |
| `model.py` | `DMPNNEncoder` (chemprop v2 BondMessagePassing) + `PolymerRankingModel` (孪生网络 + FFN) |
| `loss.py` | `MultiTaskBayesianRankingLoss`：BPR 排序损失 + Delta 回归损失的加权组合 |
| `training.py` | 训练逻辑：`EarlyStopping`、epoch 运行、指标计算、`train()` / `finetune()` |
| `predict.py` | 推理逻辑：checkpoint 加载、单对预测 (`predict_pair`)、批量预测 (`predict_batch`) |
| `cli.py` | 命令行入口，支持 `train` / `finetune` / `predict` 三种模式 |

## 依赖

- Python >= 3.9
- PyTorch
- chemprop >= 2.0.0
- RDKit
- scikit-learn
- scipy
- pandas
- numpy

## 数据格式

训练 CSV 需包含以下列（以 `_1` / `_2` 后缀区分两个聚合物）：

| 列名 | 说明 |
|------|------|
| `Materials_1` / `Materials_2` | 聚合物名称 |
| `MonomerA_1` / `MonomerA_2` | 单体 A 的 SMILES（含 2 个 `*` 标记） |
| `MonomerB_1` / `MonomerB_2` | 单体 B 的 SMILES（含 2 个 `*` 标记） |
| `conjugation_{s}` | 共轭性 (linear=1, 其他=0) |
| `Isomer_{s}` | 异构体标记 |
| `CentroSymmetry_{s}` | 中心对称性 |
| `E_LUMO (eV)_{s}` | LUMO 能级 |
| `E_HOMO (eV)_{s}` | HOMO 能级 |
| `mu_e_{s}` | 电子迁移率 (原始值，程序内部取 log10) |
| `mu_h_{s}` | 空穴迁移率 (原始值，程序内部取 log10) |

其中 `{s}` 为 `1` 或 `2`。

### 数据预处理

使用 `preprocess.py` 将原始单体数据按 DOI 分组生成配对数据：

```bash
python preprocess.py
# 读取 contrastive_monomer.csv → 输出 contrastive_monomer_paired.csv
```

## 使用方法

### 训练

```bash
python polymer_ranking.py --mode train --csv contrastive_monomer_paired.csv
```

可选参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 1000 | 训练轮数 |
| `--batch_size` | 32 | 批大小 |
| `--hidden_size` | 300 | D-MPNN 隐藏层维度 |
| `--depth` | 6 | 消息传递深度 |
| `--dropout` | 0.1 | Dropout 率 |
| `--ffn_hidden` | 256 | FFN 隐藏层维度 |
| `--lr` | 1e-3 | 学习率 |
| `--weight_decay` | 1e-5 | 权重衰减 |
| `--patience` | 25 | Early stopping 耐心值 |
| `--val_ratio` | 0.1 | 验证集比例 |
| `--test_ratio` | 0.1 | 测试集比例 |
| `--seed` | 42 | 随机种子 |
| `--save_dir` | checkpoints | 模型保存目录 |

### 微调

在已有 checkpoint 基础上使用全量数据微调：

```bash
python polymer_ranking.py --mode finetune \
    --csv contrastive_monomer_paired.csv \
    --checkpoint checkpoints/best_model.pt \
    --finetune_epochs 10 \
    --finetune_lr 1e-5
```

### 预测

对新聚合物对进行批量预测：

```bash
python polymer_ranking.py --mode predict \
    --predict_csv new_mol.csv \
    --checkpoint checkpoints/final_model.pt \
    --output predictions.csv
```

预测输出包含以下列：

| 列名 | 说明 |
|------|------|
| `score_mu_e_1` / `score_mu_e_2` | 两个聚合物的 μ_e 打分 |
| `score_mu_h_1` / `score_mu_h_2` | 两个聚合物的 μ_h 打分 |
| `prob_mu_e` / `prob_mu_h` | 排序置信概率 (sigmoid(Δscore)) |
| `preferred_mu_e` / `preferred_mu_h` | 迁移率更高的聚合物名称 |

## 模型架构

```
SMILES (多结构) ──► D-MPNN ──► 平均嵌入 ──► mol_emb [H]
                                                    │
extra_features [5] ────────────────────────────────┤
                                                    ▼
                                              Concatenate
                                                    │
                                                    ▼
                                    Linear → LayerNorm → SiLU → Dropout
                                                    │
                                                    ▼
                                    Linear → SiLU → Dropout
                                                    │
                                                    ▼
                                         Linear → [score_e, score_h]
```

两个聚合物共享同一套参数（孪生网络），每个聚合物可能因非对称单体拼接产生多个环化结构，其 D-MPNN 嵌入在送入 FFN 前取平均。

## 损失函数

对每个任务 t ∈ {μ_e, μ_h}：

- **BPR 排序损失**：`L_bpr = -log(σ(sign(y1 - y2) · (s1 - s2)))`，最大化正确排序的对数似然
- **Delta 回归损失**：`L_reg = MSE(s1 - s2, y1 - y2)`，约束打分差与真实差值一致

总损失 = Σ_t λ_t · (α · L_bpr_t + β · L_reg_t)，默认 α=0.6, β=0.4。

## 评估指标

| 指标 | 说明 |
|------|------|
| `{task}_pair_acc` | 配对排序准确率 |
| `{task}_spearman` | Spearman 秩相关系数 |
| `{task}_avg_prob` | 平均排序置信概率 |
