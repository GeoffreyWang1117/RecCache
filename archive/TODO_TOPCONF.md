# RecCache 顶会级别改进任务清单

> 创建时间: 2026-01-01
> 目标会议: KDD / WWW / RecSys / SIGIR
> 预计周期: 8-10周

---

## 当前实验结果问题总结

### 致命问题
1. **RecCache未超越LRU基线** - 统计检验p>0.05，无显著差异
2. **2/4数据集使用合成数据** - Amazon-Books和Yelp需要真实数据
3. **核心组件消融不完整** - Quality Predictor和Reranker未单独消融

### 已完成的实验
- [x] 4个数据集基础实验 (ML-100K, ML-1M, Amazon-Books合成, Yelp合成)
- [x] 4个推荐模型基线 (Pop, ItemKNN, BPR-MF, MF)
- [x] 5个缓存策略基线 (LRU, LFU, FIFO, Random, RecCache)
- [x] 基础消融实验 (w/o Clustering)
- [x] 参数敏感性分析 (n_clusters, cache_size)
- [x] 用户分组分析 (Cold, Sparse, Normal, Active)
- [x] 统计显著性检验

---

## Phase 1: 修复核心问题 [P0-致命] (Week 1-2)

### Task 1.1: 修复RecCache核心算法
- [ ] **分析问题根因**: 检查CacheManager中quality_predictor的使用方式
- [ ] **实现质量感知决策**:
  ```python
  # 在 src/reccache/cache/manager.py 中添加
  def quality_aware_decision(self, user_id, cached_recs):
      predicted_quality = self.quality_predictor.predict(user_id, cached_recs)
      if predicted_quality < self.quality_threshold:
          return self.compute_fresh_recommendations(user_id)
      return cached_recs
  ```
- [ ] **添加缓存新鲜度追踪**: 记录缓存创建时间，实现TTL机制
- [ ] **实现动态质量阈值**: 根据用户类型调整阈值
- [ ] **验证改进效果**: RecCache应显著优于LRU (p<0.05)

**预期结果**: RecCache Hit Rate提升5-10%，或NDCG显著优于LRU

### Task 1.2: 下载真实数据集
- [ ] **Amazon-Books**:
  - 来源: https://jmcauley.ucsd.edu/data/amazon/
  - 文件: ratings_Books.csv 或 reviews_Books_5.json.gz
  - 放置: data/amazon/
- [ ] **Yelp Open Dataset**:
  - 来源: https://www.yelp.com/dataset
  - 需要申请下载
  - 放置: data/yelp/
- [ ] **Gowalla**:
  - 来源: https://snap.stanford.edu/data/loc-gowalla.html
  - 文件: loc-gowalla_totalCheckins.txt.gz
  - 放置: data/gowalla/
- [ ] **LastFM**:
  - 来源: http://millionsongdataset.com/lastfm/
  - 放置: data/lastfm/
- [ ] **更新data_loader.py**: 支持新数据集的加载和预处理

### Task 1.3: 完善消融实验
- [ ] **w/o Quality Predictor**: 移除质量预测，仅用聚类+LRU
- [ ] **w/o Reranker**: 移除重排序，直接返回缓存结果
- [ ] **w/o L2 Cache**: 仅使用本地缓存，移除Redis层
- [ ] **w/o Online Update**: 使用静态聚类，不在线更新
- [ ] **不同聚类方法**: K-Means vs MiniBatch K-Means vs DBSCAN

**消融实验表格模板**:
```
| 配置                    | ML-1M Hit Rate | ML-1M NDCG | 变化   |
|------------------------|----------------|------------|--------|
| RecCache (Full)        | 0.xxx          | 0.xxx      | -      |
| w/o Quality Predictor  | 0.xxx          | 0.xxx      | -x.x%  |
| w/o Reranker           | 0.xxx          | 0.xxx      | -x.x%  |
| w/o L2 Cache           | 0.xxx          | 0.xxx      | -x.x%  |
| w/o Online Update      | 0.xxx          | 0.xxx      | -x.x%  |
```

---

## Phase 2: 完善实验 [P0-必要] (Week 3-4)

### Task 2.1: 效率对比实验
- [ ] **延迟分解测试**:
  ```python
  # 需要测量的各组件延迟
  latency_components = [
      "user_embedding_lookup",
      "cluster_assignment",
      "cache_lookup",
      "quality_prediction",
      "reranking",
      "total_latency"
  ]
  ```
- [ ] **吞吐量测试**: 不同并发数下的QPS
- [ ] **内存占用分析**: 各组件内存消耗
- [ ] **可扩展性测试**: 用户数从1K到1M的延迟变化曲线

**效率分析表格模板**:
```
| 组件              | 延迟(ms) | 内存(MB) | 占比   |
|-------------------|----------|----------|--------|
| Embedding Lookup  | x.xx     | xxx      | xx%    |
| Cluster Assignment| x.xx     | xxx      | xx%    |
| Cache Lookup      | x.xx     | xxx      | xx%    |
| Quality Prediction| x.xx     | xxx      | xx%    |
| Reranking         | x.xx     | xxx      | xx%    |
| Total             | x.xx     | xxx      | 100%   |
```

### Task 2.2: 更多推荐基线
- [ ] **LightGCN**: 图卷积推荐 (SIGIR 2020)
- [ ] **SASRec**: 自注意力序列推荐 (ICDM 2018)
- [ ] **NGCF**: 神经图协同过滤 (SIGIR 2019)
- [ ] **SLIM**: 稀疏线性方法 (经典基线)
- [ ] **MultVAE**: 变分自编码器推荐 (WWW 2018)

### Task 2.3: Oracle缓存上界
- [ ] **实现Belady最优算法**: 基于未来请求的最优缓存替换
- [ ] **计算理论上界**: 最优缓存的Hit Rate和NDCG
- [ ] **Gap分析**: RecCache与Oracle的差距

---

## Phase 3: 理论深化 [P1-重要] (Week 5-6)

### Task 3.1: 问题形式化
- [ ] **定义QARC问题** (Quality-Aware Recommendation Caching):
  ```
  输入: 用户集U, 物品集I, 推荐模型f, 缓存容量C
  目标: max Σ_t [α·Q(cache_t, fresh_t) + (1-α)·hit_rate]
  约束: |Cache| ≤ C
  ```
- [ ] **证明NP-hard性**: 问题复杂度分析
- [ ] **设计近似算法**: 提供近似比保证

### Task 3.2: 理论分析
- [ ] **聚类误差界**: K-Means聚类误差与推荐质量损失的关系
  ```
  Theorem 1: 若聚类误差 ≤ ε，则质量损失 ≤ O(ε·d)
  ```
- [ ] **收敛性分析**: 在线K-Means的收敛速度
  ```
  Theorem 2: 在线K-Means在T轮后误差 ≤ O(1/√T)
  ```
- [ ] **竞争比分析**: RecCache相对于Oracle的竞争比
  ```
  Theorem 3: RecCache的竞争比 ≥ 1 - O(1/K)
  ```

### Task 3.3: 复杂度分析
- [ ] **时间复杂度**: 各组件的时间复杂度
- [ ] **空间复杂度**: 内存使用的理论分析
- [ ] **通信复杂度**: 分布式场景下的通信开销

---

## Phase 4: 创新增强 [P2-加分] (Week 7-8)

### Task 4.1: 自适应聚类粒度 (推荐)
- [ ] **核心思想**: 用户密集区域细粒度，稀疏区域粗粒度
- [ ] **算法设计**:
  ```python
  class AdaptiveClustering:
      def compute_local_density(self, user_embedding):
          # 计算局部密度
          pass

      def adaptive_cluster_assignment(self, user):
          density = self.compute_local_density(user)
          if density > threshold:
              return self.fine_grained_cluster(user)
          else:
              return self.coarse_grained_cluster(user)
  ```
- [ ] **理论保证**: 在质量约束下最大化缓存效率
- [ ] **实验验证**: 对比固定粒度的改进

### Task 4.2: 强化学习缓存决策 (备选)
- [ ] **MDP建模**: State-Action-Reward定义
- [ ] **算法选择**: DQN / PPO / A2C
- [ ] **收敛性证明**: 策略收敛性分析

### Task 4.3: 时序与公平性分析
- [ ] **概念漂移检测**: 用户兴趣随时间变化
- [ ] **公平性指标**: 用户群体间推荐质量差异
- [ ] **动态调整**: 根据漂移程度调整缓存策略

---

## 实验运行命令

### 运行完整实验
```bash
cd /home/coder-gw/Projects/RecCache
python scripts/run_experiments.py
```

### 运行单个数据集
```bash
python scripts/run_experiments.py --dataset ml-1m
```

### 查看结果
```bash
cat results/experiment_results.json | python -m json.tool
```

---

## 文件结构参考

```
RecCache/
├── src/reccache/
│   ├── cache/
│   │   ├── manager.py        # 需要修改: 添加quality-aware决策
│   │   ├── baselines.py      # 已有: 缓存基线
│   │   └── warming.py        # 缓存预热
│   ├── models/
│   │   ├── baselines.py      # 已有: 推荐基线
│   │   ├── quality_predictor.py  # 需要增强
│   │   └── reranker.py       # 需要验证效果
│   ├── clustering/
│   │   └── online_kmeans.py  # 需要添加自适应粒度
│   ├── evaluation/
│   │   └── experiment.py     # 已有: 实验框架
│   └── utils/
│       └── data_loader.py    # 需要添加新数据集支持
├── scripts/
│   └── run_experiments.py    # 实验运行脚本
├── results/
│   └── experiment_results.json  # 实验结果
├── data/                     # 数据集目录
│   ├── ml-100k/
│   ├── ml-1m/
│   ├── amazon/              # 需要下载
│   ├── yelp/                # 需要下载
│   └── gowalla/             # 需要下载
├── RESEARCH_GAP_ANALYSIS.md  # 差距分析文档
└── TODO_TOPCONF.md           # 本文件
```

---

## 进度追踪

| 阶段 | 任务 | 状态 | 完成日期 |
|------|------|------|----------|
| Phase 1 | Task 1.1 修复核心算法 | [ ] 待开始 | |
| Phase 1 | Task 1.2 下载真实数据集 | [ ] 待开始 | |
| Phase 1 | Task 1.3 完善消融实验 | [ ] 待开始 | |
| Phase 2 | Task 2.1 效率对比实验 | [ ] 待开始 | |
| Phase 2 | Task 2.2 更多推荐基线 | [ ] 待开始 | |
| Phase 2 | Task 2.3 Oracle缓存上界 | [ ] 待开始 | |
| Phase 3 | Task 3.1 问题形式化 | [ ] 待开始 | |
| Phase 3 | Task 3.2 理论分析 | [ ] 待开始 | |
| Phase 3 | Task 3.3 复杂度分析 | [ ] 待开始 | |
| Phase 4 | Task 4.1 自适应聚类 | [ ] 待开始 | |
| Phase 4 | Task 4.2 RL缓存决策 | [ ] 可选 | |
| Phase 4 | Task 4.3 时序公平性 | [ ] 可选 | |

---

## 下次会话启动指令

```
请阅读 /home/coder-gw/Projects/RecCache/TODO_TOPCONF.md
继续按照任务清单执行改进工作，从Phase 1的Task 1.1开始。
```
