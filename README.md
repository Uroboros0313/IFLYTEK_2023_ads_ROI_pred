# 科大讯飞跨境电商效果广告ROI预测

- 初赛: Rank8
- 复赛: Rank9

## PREFIX
一场Label有问题的比赛，和其余选手沟通之后发现前排上分主要由特征泄露和标签平滑构成。本人使用了特征泄露实际上不合比赛规则。

本场比赛主要的问题在于，大部分人没有经过数据分析，因此不知道一个广告上线了多天，并且带`cum_`前缀的所有字段都是从广告上线日期开始计算的。

## 方案

- **LGBM单模合业务特征工程**

- 特征:
1. 价格特征，不同价格的商品ROI差异较大
2. 拟ROI，收入除以支出
3. 上线日期
4. 漂移量较小的类别型特征combo后做标签编码/计数编码
5. 投放国家字段切割
6. 剩余的时间序列特征、数值型combo和时间拆分提点有限
7. **漂移特征/标签平滑**，本人测试赛段前期标签平滑后没有用处，遂放弃，但后验分析时发现误差主要集中在高ROI的广告上，因此大概率是标签平滑处理出错。

- 模型：LGBM单模无KFOLDS，CGB有效，但是今后的比赛不想再把重点放在做融合了

- 其余技巧：
1. 小于0值修正为0
2. 样本加权


