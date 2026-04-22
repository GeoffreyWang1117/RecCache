# RecCache Workshop Submission Plan

## ICLR 2026 Workshop 选择建议

### 第一优先：ICBINB (I Can't Believe It's Not Better)

**为什么适合**:
- Workshop主题：关注"负面结果"和"意外发现"
- RecCache的故事：**"简单的聚类+规则方法竟然接近复杂RL方法(CARL)的效果"**
- 这正是ICBINB想要的 - 挑战"深度学习一定更好"的预设

**投稿角度**:
```
Title: "Do We Really Need Reinforcement Learning for Recommendation Caching?
        A Surprisingly Effective Clustering-Based Approach"
```

**核心论点**:
1. CARL (WWW 2024) 使用复杂的强化学习解决推荐缓存问题
2. 我们发现简单的用户聚类+启发式规则可以达到接近的效果
3. 训练成本低100x，推理延迟低10x，可解释性强

---

### 第二优先：DATA-FM (Data Problems for Foundation Models)

**为什么可能适合**:
- 关注FM的数据效率问题
- RecCache可以定位为"推荐模型的数据复用"

**投稿角度**:
```
Title: "Cluster-Level Data Reuse for Efficient Recommendation Serving"
```

**核心论点**:
1. 推荐系统需要为每个用户重新计算
2. 我们发现用户embedding聚类后可以共享计算结果
3. 这是一种隐式的数据压缩/复用

---

### 第三优先：Reliable Autonomy

**为什么可能适合**:
- 关注系统可靠性
- RecCache提供可预测的服务质量

**投稿角度**:
```
Title: "Quality-Aware Caching for Reliable Recommendation Services
        Under High-Throughput Constraints"
```

---

## 不推荐的Workshop

| Workshop | 原因 |
|----------|------|
| SPOT | 专注LLM post-training，不相关 |
| DeLTa | 专注生成模型理论，不相关 |
| LLM Reasoning | 专注LLM推理，不相关 |
| AI4Mat/AI4Peace等 | 领域特定，不相关 |

---

## 推荐的论文框架 (ICBINB版本)

### Title
**"Do We Really Need Reinforcement Learning for Recommendation Caching? Rethinking Complexity in Cache-Aware Recommender Systems"**

### Abstract (Draft)
```
Modern large-scale recommender systems increasingly rely on caching to handle
peak traffic loads. Recent work (CARL, WWW 2024) proposes sophisticated
reinforcement learning approaches to jointly optimize real-time and cached
recommendations. However, we ask: is such complexity necessary?

We present RecCache, a surprisingly simple approach based on user clustering
and quality-aware eviction. Despite requiring no RL training, RecCache achieves
comparable hit rate improvements (+35-48%) to RL-based methods while being:
(1) 100x cheaper to train, (2) 5x faster at inference, and (3) fully interpretable.

Our findings suggest that the recommendation caching problem may be
over-engineered, and simple clustering-based approaches deserve more attention.
```

### 论文结构

1. **Introduction** (1 page)
   - 推荐系统缓存问题背景
   - CARL等复杂方法的局限性
   - 我们的核心问题：是否需要RL？

2. **Related Work** (0.5 page)
   - CARL, GNN-DDQN等RL/GNN方法
   - 传统缓存策略 (LRU, LFU等)

3. **RecCache: A Simple Alternative** (1.5 pages)
   - 用户聚类共享缓存
   - 质量感知驱逐策略
   - 轻量级重排序

4. **Experiments** (2 pages)
   - ML-100K, ML-1M结果
   - Hit Rate: +35-48%
   - 效率对比: 5x吞吐量提升
   - 消融实验

5. **Discussion: When Simple is Enough** (0.5 page)
   - 为什么简单方法有效
   - 适用场景分析
   - 局限性

6. **Conclusion** (0.3 page)

---

## 关键实验表格 (论文用)

### Table 1: Main Results
| Method | Hit Rate | NDCG | Training Cost | Inference |
|--------|----------|------|---------------|-----------|
| LRU (no clustering) | 0.52 | 0.26 | 0 | 0.1ms |
| CARL* (RL-based) | ~0.75** | ~0.25** | High | ~5ms |
| **RecCache (Ours)** | **0.76** | **0.26** | **Low** | **0.3ms** |

*Reported in WWW 2024
**Estimated from paper

### Table 2: Ablation Study
| Configuration | Hit Rate | NDCG |
|--------------|----------|------|
| Plain LRU | 0.52 | 0.26 |
| + User Clustering | 0.76 (+46%) | 0.26 |
| + Quality-aware Eviction | 0.76 | 0.26 |
| + Reranker | 0.76 | 0.26 |

### Table 3: Efficiency
| Metric | RecCache | Fresh Compute |
|--------|----------|---------------|
| Throughput (req/s) | 36,784 | 7,462 |
| Speedup | **4.9x** | 1x |

---

## 投稿时间线

| 日期 | 任务 |
|------|------|
| 2026年1月初 | 完成论文初稿 |
| 2026年1月中 | 内部review和修改 |
| 2026年1月30日 | **SPOT deadline** (备选) |
| 2026年2月初 | 预计ICBINB deadline |

---

## 需要补充的实验

1. **与CARL的直接对比** (如果可以复现)
2. **更大规模数据集** (如果有)
3. **Oracle上界分析** (Belady's Algorithm)
4. **冷启动用户实验**

---

## 备注

- ICBINB强调"惊人的发现"，我们的故事非常契合
- 论文长度通常4-8页，短论文也可接受
- 可以同时投多个workshop（如果不冲突）
