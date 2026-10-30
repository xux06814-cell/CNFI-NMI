<div align="center">

# CNFI-NMI

**融合复杂网络交互与归一化互信息的稳健特征选择框架**

录用稿 · *Big Data Mining and Analytics* · BDMA-2026-0010

[方法与代码对应关系](docs/METHODOLOGY.md) · [复现实验规范](docs/REPRODUCIBILITY.md) · [English](README.md)

</div>

![CNFI-NMI 总体框架，来源于录用稿图 1](docs/assets/framework.png)

## 方法概述

CNFI-NMI 将两类互补信息融合为统一的特征排序：

1. **CNFI 阶段**：通过 FP-Growth 挖掘两两关联规则，动态融合置信度、
   提升度、支持度、确信度和杠杆率，构建加权有向特征网络，再使用
   Eigen-Rank 计算交互重要性。
2. **NMI 阶段**：独立地对全部输入特征计算其与目标变量的归一化
   互信息，衡量非线性相关性。
3. **秩融合阶段**：CNFI 和 NMI 都给出全部输入特征的排序；先将两份
   排名按特征名称对应，再以 `alpha=0.5` 为默认权重融合秩得分，最后
   按 `n_features_to_select` 保留所需数量的特征。

## 安装与测试

```bash
git clone https://github.com/xux06814-cell/CNFI-NMI.git
cd CNFI-NMI
python -m pip install -e ".[test]"
pytest
python examples/quickstart.py
```

## 与原文保持一致的关键点

- CNFI 连续变量采用 5 档等频离散化；NMI 使用未经 CNFI 离散化的原始列。
- PMI 代理严格使用 `log2(Lift)`，与原文统一以 bit 为单位。
- NMI 使用几何均值归一化，对应原文式 (26)。
- 秩得分严格实现 `(N-rank)/N`，最后一名得分为 0。
- 默认参数 `min_support=0.01`、`min_confidence=0.1`、`alpha=0.5`、
  `kappa=1.0`、`theta=0.0` 以及两组先验权重均对应原文。
- 关联规则、网络、阶段得分和最终排序均作为拟合后属性公开，方便审计。

## 学术复现边界

当前版本已实现并测试论文算法，但不虚构“完全复现”结论。原仓库和所附
PDF 没有包含三套原始数据的精确归档、哈希、完整预处理脚本、折划分种子
及完整 LightGBM 搜索网格，因此仓库暂不声称能够逐点还原论文全部数值图表。
复现前须按照 [数据清单](data/README.md) 补齐来源和校验信息，再使用
`scripts/run_experiment.py` 保留外层测试折；特征选择在整个外层训练折上完成后才进行内层调参，因此内层验证并非完全嵌套，详见 `docs/REPRODUCIBILITY.md`。

## 目录结构

```text
src/cnfi_nmi/       维护中的论文算法实现
tests/              公式与接口回归测试
scripts/            五折 LightGBM 实验脚本（内层调参限制见文档）
examples/           可直接运行的合成数据示例
docs/               方法映射与严格复现规范
paper/              论文来源与获取说明
data/               数据来源与校验要求（不直接分发原始数据）
legacy/             原仓库代码快照，仅供审计追溯
```

## 许可说明

仓库尚未声明软件或论文许可证。在作者明确添加许可证文件之前，软件、论文
和图件的相关权利仍归各作者及出版方所有。
