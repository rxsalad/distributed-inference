# DPA + EP

> 超大规模 MoE 模型（如 DeepSeek）推理中，Attention 与 MoE 两部分采用不同的并行策略：Attention 用 **DPA（Data Parallel Attention）**，MoE 用 **EP（Expert Parallelism）**。本文讨论**怎么执行**，按"为什么这样组合 → 两阶段的并行粒度 → 每个阶段怎么做 → 一层的完整流程 → 挑战"展开；这个组合的价值与成本、边界账（含通信量估算）见 `ep_values.md`。Attention 侧的细节见 `dp_attention.md`，MoE 本身的结构见 `moe.md`，All-to-All 等集合通信的含义见 `parallelism-basics.md`。

## 1. 为什么这样组合

- **参数分布不均**：超大 MoE 模型的参数规模主要集中在 FFN / Expert 部分，Attention 参数相对较小（如 DeepSeek-V3 的 Attention 约占总参数的 1.7%，复制代价的分析见 `dp_attention.md` 第 5 节），没有必要让 Attention 参与复杂的 TP 通信。
- **MLA 让 Attention 的 TP 效率下降**：MLA 使 Attention 的 KV Cache 与 head 并行之间的关系变复杂，要么复制 KV，要么跨卡通信（详见 `dp_attention.md` 第 3 节）。
- **通信资源要用在刀刃上**：把通信密集的 Attention 从 TP 中解耦出来，用 Data Parallel 在不同 GPU 上独立执行，GPU 间的通信资源就可以集中留给真正占据大部分参数量的 MoE Experts。

核心思想：**Attention 使用 Data Parallel，避免 TP 带来的频繁通信；MoE 使用 Expert Parallel，把大量 Experts 分布到不同 GPU，以支持更大的模型容量。**

## 2. 两个阶段的并行粒度

| | Attention 阶段 | MoE 阶段 |
| --- | --- | --- |
| 并行方式 | DP（Data Parallel） | EP（Expert Parallel） |
| 并行粒度 | **request** | **token** |
| 数据放置 | request 绑定一个 DP rank（通常对应一张 GPU），KV Cache 与 hidden states 都在该 GPU 本地 | Router 按每个 token 的 hidden state 选 Expert，token 被发到拥有目标 Expert 的 GPU |
| 通信 | 无 AllReduce | EP Group 内的 All-to-All（Dispatch / Combine） |
| 数据迁移 | 不迁移 | token 的 hidden state 临时搬运，**request 不迁移** |

### EP Group：Expert 并行的通信范围

表里和后文多次出现的 **EP Group**，指一组**共同持有模型全部 Expert、彼此之间做 All-to-All 的 GPU**。

- **Expert 怎么分**：全部 Expert 被切分到组内各 GPU，每个 GPU 只持有其中一部分。组内 GPU 数就是 **EP 度**。例如（示意）256 个 routed Expert、EP 度为 32，则每张 GPU 持有 8 个 Expert。
- **通信范围**：Dispatch / Combine 的 All-to-All 只在组内进行，不会发到组外。Router 选中的 Expert 可能在组内任意一张 GPU 上，所以组内每个 rank 都可能向其他任意 rank 发 token，也都要接收别人发来的 token。
- **所有 rank 必须同步参与**：All-to-All 是集合通信，组内每个 rank 都要参与才能完成，这也是第 7 节"快的 rank 要等慢的"的原因。
- **与 DPA 的关系**：DPA + EP 下，同一批 GPU 在两个阶段扮演不同角色——Attention 阶段每张 GPU 是一个 **DP rank**（各管自己的 request），MoE 阶段每张 GPU 是 EP Group 里的一个 **EP rank**（各管一部分 Expert）。EP Group 通常覆盖所有这些 GPU，所以 EP 度 = TP × DP（TP = 1 时即 DP 度，见 `parallelism-basics.md` 第 6 节）。
- **跨机**：EP 度大时，EP Group 会跨多台机器：机内 All-to-All 走 NVLink，跨机走节点间网络，后者带宽更低，网络拓扑与通信量见 `ep_values.md` 第 3 节，EP 度随负载类型如何取见第 4 节。
- **多副本**：部署多个模型副本时，每个副本有自己的 EP Group，组间互不通信。

## 3. Attention 阶段：按 request 做 DP

- 每个 request 绑定到一个 DP rank，由该 GPU 独立完成 Attention，包括访问对应的 KV Cache 和 hidden states；
- KV Cache 保持本地，避免 TP Attention 的 AllReduce 通信。

**request affinity（请求亲和性）**：理论上，只要某个 GPU 拥有该 request 的 hidden states 和 KV Cache，下一层 Attention 就可以由任意 GPU 执行。但实际系统通常让同一个 request 始终由同一个 DP rank 负责后续 Attention，以避免 KV Cache 和中间状态在 GPU 之间迁移带来的额外开销。因此，**MoE 的 EP 通信不会改变 request 的 Attention owner**。

## 4. MoE 阶段：按 token 做 EP

进入 MoE 层后，并行粒度从 request 变成 token：

1. **Router**：根据每个 token 的 hidden state，选出对应的 Expert；
2. **Dispatch**：通过 EP Group 中的 All-to-All，把 token 分发到拥有目标 Expert 的 GPU；
3. **Expert 计算**：各 GPU 对收到的 token 计算对应的 Expert；
4. **Combine**：计算结果通过 All-to-All 返回，恢复原来的 token 顺序，并按路由权重加权求和。

所以 MoE 阶段发生的是 **token 在 GPU 之间的临时搬运**，而不是整个 request 的迁移。

**进入 MoE 的是什么**：不是"下一个 token 的 hidden state"，而是**当前 token 位置对应的 hidden state**。每一层都会对序列中各 token 位置的 hidden state 做 Attention 和 MoE/FFN 计算，经过多层变换后，最终得到用于预测下一个 token 的输出分布。Decode 阶段每生成一个新 token，只有这个新位置的 hidden state 需要经过后续 MoE 计算（历史 token 的 K/V 已在 KV Cache 里）。

## 5. 一层的完整流程

```
        DP rank 0                DP rank 1                DP rank 2
   ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
   │ Attention     │        │ Attention     │        │ Attention     │   按 request，本地 KV
   │ (request A,B) │        │ (request C)   │        │ (request D,E) │
   └───────┬───────┘        └───────┬───────┘        └───────┬───────┘
           │   Router 选 Expert      │                        │
           └──────────────┬──────────┴────────────────────────┘
                          ▼
              Dispatch：All-to-All（token → Expert 所在 GPU）
                          ▼
        ┌─────────────┬─────────────┬─────────────┐
        │ GPU0: E0~E3 │ GPU1: E4~E7 │ GPU2: E8~E11│                    按 token，Expert 计算
        └─────────────┴─────────────┴─────────────┘
                          ▼
              Combine：All-to-All（结果 → token 原 GPU，恢复顺序）
                          ▼
           回到各 request 的 owner GPU，进入下一层 Attention
```

**DPA + EP 中，Attention 的并行单位是 request，request 通常固定绑定一个 GPU；MoE 的并行单位是 token，token 会在 Expert GPU 之间临时迁移。每层 MoE 完成后，hidden states 通常会回到 request owner GPU，然后继续下一层 DPA Attention。**

## 6. 其他搭配：DPA + TP（DTP）与 TP + EP（TEP）

MoE 层也可以不用 EP，而是对 Expert（或稠密 FFN）做 TP 切分。此时每个 GPU 持有 FFN 权重的一部分，需要**全部 token** 才能算出部分和，数据重排的方式随之不同：

- 进入 FFN 前：**All-Gather**，让每个 GPU 拿到所有 DP rank 的 token；
- FFN 计算：每个 GPU 用自己那份权重对全部 token 算出部分和；
- 离开 FFN 时：**Reduce-Scatter**，在求和的同时把结果按 token 归属切回各 DP rank（相当于 AllReduce 的前半步：求和后每个 GPU 只拿自己那份）。

**四种方案的通信对比**（每层；简称的定义见 `parallelism-basics.md` 第 6 节）：

| 方案 | 简称 | Attention 部分 | FFN / MoE 部分 |
| --- | --- | --- | --- |
| 纯 TP | TP | AllReduce | AllReduce |
| DPA + TP | DTP | 无 | All-Gather + Reduce-Scatter |
| DPA + EP | DEP | 无 | Dispatch + Combine（两次 All-to-All） |
| TP Attention + EP | TEP | AllReduce | 每个 GPU 已有全部 token，按 Router 结果取出属于本机 Expert 的 token 计算，再把各 GPU 的结果求和合并（AllReduce 或 Reduce-Scatter，取决于实现） |

**TEP 与 DEP 的取舍**：两者 MoE 侧都是 EP，区别在 Attention。TEP 的 Attention 仍是 TP，KV Cache 按 head 切，对 MHA / GQA 模型友好（TP 度不超过 KV 组数时没有复制问题），但每层要付 Attention 的 AllReduce；对 MLA 模型则仍会遇到 `dp_attention.md` 第 3 节的 KV 无法切分的矛盾，这正是 DEP 的适用场景。

引入 DPA 后，**Attention 内部的 AllReduce 消失**，换成进出 FFN / MoE 的两次集合通信。具体采用哪种通信形式取决于引擎和配置，以上是通用情况。通信量估算（DPA + EP 与 DPA + TP 相差 P/k 因子）见 `ep_values.md` 第 3 节。

## 7. 挑战

这套组合让 Attention 保持简单高效的本地计算，但把压力转移到了 MoE 前后的集合通信和负载上。其中**以下两条是 DPA + EP 流程特有的挑战**；All-to-All 开销、Expert 热点、EP 度边界等 EP 层面的代价，以及热点的工程缓解（EPLB、Wide EP），见 `ep_values.md` 第 6 节：

| 挑战 | 说明 |
| --- | --- |
| DP rank 间负载不均 | 各 rank 的 request 数、序列长度不同。进出 MoE 的集合通信要所有 DP rank 同步参与，快的 rank 要等慢的 |
| request 粘性 | request 固定在一个 DP rank，路由策略和 prefix cache 的命中受此限制 |

## 8. 小结

- **Attention**：按 request 做 DP，KV 留在本地，没有 Attention 内的 AllReduce；
- **MoE**：按 token 做 EP，通过 All-to-All 把 token 临时搬到 Expert 所在的 GPU，算完再搬回；
- **为什么合理**：MLA 使 Attention 的 TP 效率下降，而 DeepSeek 一类模型的主要参数集中在 MoE Experts。把 Attention 从 TP 中解耦、采用 DPA，把 GPU 间通信资源集中用于 MoE 的 EP，更符合超大 MoE 推理的特点。
