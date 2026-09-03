# Polymer Carrier Mobility Prediction via D-MPNN

基于 D-MPNN（Directed Message Passing Neural Network）的聚合物载流子迁移率预测模型，使用贝叶斯个性化排序（BPR）损失进行聚合物对的排序学习，同时预测电子迁移率（μ_e）和空穴迁移率（μ_h）。

## 工作流程

```
单体A + 单体B ──► 单体拼接（重复单元）──► 环化（环状模型化合物）──► D-MPNN 编码 ──► FFN ──► 排序分数
                                                                      │
                                          额外特征（共轭、异构、对称性、LUMO、HOMO）──┘
                                              │
                                              ▼
                                   边权重（sp3 C 降权 0.2，其余 1.0）
```

1. **单体拼接**：将两个单体（含 `*` 标记）通过 `*` 原子连接，生成聚合物的重复单元，对不对称单体自动生成所有可能的连接方向
2. **环化**：将重复单元的剩余 `*` 原子连接，形成环状模型化合物，并标记连接点（Connection Points）
3. **D-MPNN 编码（带边权重）**：使用 chemprop v2 的 `WeightedBondMessagePassing` 对分子图进行编码。在消息传递过程中，对触及**非环** sp3 杂化碳原子（非共轭侧链体系，如烷基侧链）的边施加权重 `sp3_weight=0.2`，环内 sp3 碳（如环结构中的碳）及其余边权重为 1.0，从而降低非共轭侧链对迁移率学习的影响。同一聚合物的多个环化结构取平均
4. **排序预测**：拼接额外特征后通过 FFN 输出分数，孪生网络共享参数，比较两个聚合物的迁移率高低

## 项目结构

```
ofet_monomer_dmpnn/
├── polymer_ranking.py           # 主入口脚本
├── preprocess.py                # 数据预处理：生成对比配对数据
├── preprocess_new.py            # SMILES 校验与规范化
├── check_smiles.py              # SMILES 连接正确性检查
├── polymer_ranking/             # 核心包
│   ├── __init__.py              # 包初始化
│   ├── config.py                # 模型/训练/预测配置
│   ├── model.py                 # WeightedBondMessagePassing 编码器 + 排序模型
│   ├── dataset.py               # 数据集与 collate 函数
│   ├── chemistry.py             # 化学预处理（拼接、环化、特征提取）
│   ├── featurizer.py            # 自定义原子特征化器 + sp3 边权重计算
│   ├── loss.py                  # 多任务 BPR 损失函数
│   ├── training.py              # 训练/微调/早停
│   ├── predict.py               # 预测逻辑
│   └── cli.py                   # 命令行接口
├── contrastive_monomer.csv      # 原始数据
├── contrastive_monomer_paired.csv # 预处理后的配对数据
├── new_mol.csv                  # 新分子预测数据
└── checkpoints/                 # 模型检查点
```

## 环境依赖

- Python >= 3.10
- PyTorch >= 2.0
- [chemprop >= 2.0.0](https://github.com/chemprop/chemprop)
- RDKit
- pandas, numpy, scikit-learn, scipy

```bash
pip install torch chemprop rdkit pandas numpy scikit-learn scipy
```

## 使用方法

### 1. 数据预处理

将原始 CSV 数据按 DOI 分组生成对比配对：

```bash
python preprocess.py
```

输入 `contrastive_monomer.csv`，输出 `contrastive_monomer_paired.csv`。

### 2. 训练模型

```bash
python polymer_ranking.py --mode train --csv contrastive_monomer_paired.csv
```

可选参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epochs` | int | 1000 | 训练轮数 |
| `--batch_size` | int | 32 | 批次大小 |
| `--lr` | float | 1e-3 | 学习率 |
| `--hidden_size` | int | 300 | 隐藏层维度 |
| `--depth` | int | 6 | D-MPNN 深度 |
| `--dropout` | float | 0.1 | Dropout 比例 |
| `--sp3_weight` | float | 0.2 | 触及 sp3 杂化碳原子的边的消息传递权重 |
| `--patience` | int | 50 | 早停耐心值 |
| `--seed` | int | 42 | 随机种子 |
| `--save_dir` | str | checkpoints | 模型保存目录 |

### 3. 微调模型

在全量数据上对最佳模型进行微调：

```bash
python polymer_ranking.py --mode finetune \
    --csv contrastive_monomer_paired.csv \
    --checkpoint checkpoints/best_model.pt \
    --finetune_epochs 10 \
    --finetune_lr 1e-5
```

### 4. 预测

对新分子对进行迁移率排序预测：

```bash
python polymer_ranking.py --mode predict \
    --predict_csv new_mol.csv \
    --checkpoint checkpoints/final_model.pt \
    --output predictions.csv
```

### 5. SMILES 校验

检查单体连接是否正确生成目标聚合物：

```bash
python check_smiles.py
```

### 6. 新数据预处理

对新分子 CSV 进行 SMILES 规范化：

```bash
python preprocess_new.py
```

## 模型架构

```
PolymerRankingModel (孪生网络)
├── DMPNNEncoder
│   ├── WeightedBondMessagePassing (继承自 chemprop v2 BondMessagePassing)
│   │   ├── 初始隐藏状态 h0 = W_i(atom_feat || bond_feat)
│   │   ├── 消息传递 m_vw = Σ w_uv · h_uv   (w_uv = 0.2 若触及 sp3 C，否则 1.0)
│   │   └── 最终聚合 M_v = Σ w_wv · h_wv
│   └── Aggregation (mean/sum/norm)
└── FFN
    ├── Linear(hidden + extra_dim → ffn_hidden)
    ├── LayerNorm + SiLU + Dropout
    ├── Linear(ffn_hidden → ffn_hidden/2)
    ├── SiLU + Dropout
    └── Linear(ffn_hidden/2 → num_tasks)
```

- **边权重机制**：在环化构建等效无限长链聚合物分子并用 D-MPNN 进行学习的基础上，对**非环** sp3 杂化的 C 原子（对迁移率影响较小的非共轭体系，主要是侧链）降低消息传递权重。若一条边的源原子或目标原子为非环 sp3 杂化碳，则该边的消息传递权重为 `sp3_weight`（默认 0.2），环内 sp3 碳及其余边权重为 1.0。权重在 D-MPNN 的每次消息传递迭代及最终原子级聚合中均生效
- **额外特征**（5 维）：共轭类型、异构体、中心对称性、E_LUMO、E_HOMO
- **损失函数**：`MultiTaskBayesianRankingLoss` = α · BPR Loss + β · Delta Regression Loss
- **优化器**：AdamW + CosineAnnealingLR
- **评估指标**：Pairwise Accuracy、Spearman 相关系数、平均排序概率

## 输入数据格式

### 训练/微调数据（paired CSV）

需包含以下列（`_1` 和 `_2` 后缀分别对应两个聚合物）：

| 列名 | 说明 |
|------|------|
| `MonomerA_1`, `MonomerB_1` | 聚合物1的两个单体 SMILES |
| `MonomerA_2`, `MonomerB_2` | 聚合物2的两个单体 SMILES |
| `conjugation_1`, `conjugation_2` | 共轭类型（linear=1, 其他=0） |
| `Isomer_1`, `Isomer_2` | 异构体 |
| `CentroSymmetry_1`, `CentroSymmetry_2` | 中心对称性 |
| `E_LUMO (eV)_1`, `E_LUMO (eV)_2` | LUMO 能级 (eV) |
| `E_HOMO (eV)_1`, `E_HOMO (eV)_2` | HOMO 能级 (eV) |
| `mu_e_1`, `mu_e_2` | 电子迁移率目标值 |
| `mu_h_1`, `mu_h_2` | 空穴迁移率目标值 |

### 预测数据（new_mol.csv）

格式与训练数据相同，可不含 `mu_e` 和 `mu_h` 目标列。

## 输出

- **训练**：`checkpoints/best_model.pt` — 验证集最优模型
- **微调**：`checkpoints/final_model.pt` — 全量数据微调后的模型
- **预测**：`predictions.csv` — 包含每对的预测分数、排序概率和偏好聚合物