# KVComm 复现阅读笔记

这份 README 面向第一次复现论文的读者，目标不是替代官方说明，而是帮助你把代码库、论文问题和可复现实验路径串起来。

KVComm 的核心问题是：两个 LLM 智能体不直接传自然语言答案，也不传全量 layer-wise KV cache，而是让 sender 先阅读上下文并产生 KV cache，再选择其中一部分 KV 作为隐空间通信载体传给 receiver。receiver 只看到自己的问题输入，但在生成时可以注意到 sender 的部分 KV，从而完成问答、数学、代码补全或摘要任务。

## 代码库结构

```text
.
├── com.py                  # 单 sender + 单 receiver 的主入口
├── com_ms.py               # 多 sender 场景入口，A1/A2 -> B
├── com_online.py           # 混合任务上的在线层校准入口
├── eval.py                 # baseline / skyline / KVComm / AC / NLD / CIPHER 的评测逻辑
├── eval_ms.py              # 多 sender 评测逻辑
├── eval_online.py          # 在线校准评测逻辑
├── models.py               # KVComm 的核心实现：选择性 KV cache 注入
├── models_ms.py            # 多 sender 版本的 KVComm
├── models_ac.py            # Activation Communication 对照方法
├── models_cipher.py        # CIPHER 对照方法
├── model_attn.py           # attention tracer，用于统计 B 对 A 的 KV 注意力
├── layer_importance.py     # 层重要性计算与 top layer 选择
├── dataloader/             # 各数据集的 prompt_A / prompt_B 构造与指标评估
├── utils/                  # F1、EM、日志、run name 等工具函数
└── requirements.txt
```

建议第一遍阅读顺序：

1. `com.py`：看清楚有哪些实验开关，以及模型、数据集、通信器如何被组装。
2. `eval.py`：理解 baseline、skyline 和 communication 三种输入设置的区别。
3. `models.py`：重点读 `CVCommunicator.prepare_key_cache()` 和 `forward()`，这是 partial KV sharing 的核心。
4. `layer_importance.py`：理解 `--top_layers` 如何从校准样本中选择通信层。
5. `dataloader/`：确认每个任务的 `prompt_A`、`prompt_B` 和评估指标。

## 实验语义

代码中有三条最重要的评测基线：

- `baseline`：B 只看到问题或待完成内容，不看到 A 的上下文。
- `skyline`：单个模型同时看到上下文和问题，相当于信息完整上界。
- `communication`：A 看到上下文，B 看到问题；A 的 KV cache 被传给 B，B 用隐空间信息生成答案。

在 `eval.py` 中，communication 的数据流是：

1. `input_ids_A` 由 `prompt_A` 构造，例如上下文、hint、代码上下文或摘要第一部分。
2. `input_ids_B` 由 `prompt_B` 构造，例如问题、待补全代码或摘要第二部分。
3. A 前向计算 `out_A.past_key_values`。
4. `CVCommunicator` 将 A 的部分 KV cache 映射到 B 的层上。
5. B 调用 `generate()`，输入是 `input_ids_B`，额外接收 `out_A_past_key_values`。

## KVComm 核心机制

核心类是 `models.py` 中的 `CVCommunicator`。

### 层映射

如果 A 和 B 层数不同，代码用：

```python
layer_map[l_a] = round((l_a + 0.5) * L_B / L_A - 0.5)
```

把 A 的第 `l_a` 层映射到 B 的某一层。相同模型时基本是一一对应。

### 部分 KV 选择

`prepare_key_cache()` 会遍历 A 的所有层：

- 如果该层在 `layers_list` 中，则传完整 key/value cache。
- 如果不在 `layers_list` 中，则只保留第一个 token 的 KV。
- 第一个 token 被保留是为了 attention sink，避免完全空 cache 破坏生成过程。

这也是后续研究最值得动手的位置：当前代码主要按“层”选择 KV，而你的研究可以扩展为按层、head、token、语义片段、注意力贡献或压缩预算选择有效 KV。

## 层重要性选择

当运行时传入 `--top_layers > 0`，`com.py` 会先做一次校准：

1. 构造 `apply_attn_tracer=True` 的 `CVCommunicator`。
2. 在 `calib_size` 条样本上让 B 生成。
3. 通过 `model_attn.py` 中的 tracer 保存 B 的 query/key。
4. `calc_attn_weights_from_qk()` 重算 attention weights。
5. `layer_importance.py` 统计 B 对 A 传入 KV 区间的注意力强度。
6. `get_top_layers()` 取排名靠前的层，写回 `cfg.layers_list`。
7. 再用选出的层跑正式 communication 测试。

层重要性最终是两部分的加权：

```text
importance = alpha * measured_attention + (1 - alpha) * gaussian_prior
```

相关参数：

- `--top_layers`：选择层比例，例如 `0.3` 表示选 30% 的层。
- `--calib_size`：用于估计层重要性的校准样本数。
- `--alpha`：真实注意力统计和高斯先验的权重。
- `--mu`：高斯先验中心，按层数归一化。
- `--sigma`：高斯先验宽度。
- `--random_selection`：随机选相同比例的层，用作消融。
- `--do_layer_curve`：按层重要性排名逐步增加层数，观察性能曲线。

## 环境安装

推荐新建环境：

```bash
conda create -n kvcomm python=3.10 -y
conda activate kvcomm
pip install -r requirements.txt
```

注意：

- `transformers==4.53.3` 是硬性要求，官方 README 也提醒 4.54.0 后模型定义发生变化。
- 默认模型是 `meta-llama/Llama-3.1-8B-Instruct`，需要 Hugging Face 权限和本地登录。
- 默认使用 `torch_dtype=torch.bfloat16` 和 `attn_implementation="sdpa"`。
- 单机同时加载 A 和 B 两个 8B 模型，显存压力较大；如果只是验证代码路径，先用 `--limit 1` 或换小模型试跑。

## 数据集

数据加载统一由 `dataloader.get_evaluator()` 选择。

| 任务 | 来源 | 典型形式 |
| --- | --- | --- |
| `hotpotqa` | Hugging Face | A 读 supporting facts，B 回答问题 |
| `qasper` | Hugging Face | 科学论文问答 |
| `musique` | Hugging Face | 多跳组合问答 |
| `multifieldqa_en` | Hugging Face | 多领域长上下文 QA |
| `twowikimqa` | Hugging Face | 多跳 Wikipedia QA |
| `tipsheets` | 本地 jsonl | 合成推理任务 |
| `countries` | 本地 jsonl | 国家相关问答 |
| `tmath` | 本地 TMATH | A 读 hint，B 解数学题 |
| `repobench` | 数据加载器内定义 | A 读代码上下文，B 补全下一行 |
| `samsum` | 数据加载器内定义 | 摘要任务 |

大多数任务用 F1 类指标评估；具体规则在 `BaseEvaluator.evaluate_item()` 和各任务 evaluator 中。

## 最小复现路线

第一次复现不要一上来跑全量。建议按下面顺序来，每一步先用 `--limit 1` 或很小的样本数确认流程。

### 1. Baseline

```bash
python com.py \
  --test_task hotpotqa \
  --do_test_baseline \
  --model_A meta-llama/Llama-3.1-8B-Instruct \
  --model_B meta-llama/Llama-3.1-8B-Instruct \
  --limit 1
```

这一步验证：环境、模型下载、tokenizer、数据集加载、基础生成和评估都正常。

### 2. Skyline

```bash
python com.py \
  --test_task hotpotqa \
  --do_test_skyline \
  --model_A meta-llama/Llama-3.1-8B-Instruct \
  --model_B meta-llama/Llama-3.1-8B-Instruct \
  --limit 1
```

这一步验证：模型在完整信息下的上界表现。

### 3. 指定层 KVComm

```bash
python com.py \
  --test_task hotpotqa \
  --do_test \
  --model_A meta-llama/Llama-3.1-8B-Instruct \
  --model_B meta-llama/Llama-3.1-8B-Instruct \
  --layers_list 20 21 22 23 24 25 26 \
  --limit 1
```

这一步绕开自动层选择，直接测试 partial KV sharing 主路径。

### 4. 自动 top layer KVComm

```bash
python com.py \
  --test_task hotpotqa \
  --do_test \
  --model_A meta-llama/Llama-3.1-8B-Instruct \
  --model_B meta-llama/Llama-3.1-8B-Instruct \
  --top_layers 0.3 \
  --calib_size 4 \
  --limit 10
```

这一步复现论文最关键的“先校准层重要性，再只传部分层 KV”的流程。

### 5. 随机层消融

```bash
python com.py \
  --test_task hotpotqa \
  --do_test \
  --model_A meta-llama/Llama-3.1-8B-Instruct \
  --model_B meta-llama/Llama-3.1-8B-Instruct \
  --top_layers 0.3 \
  --random_selection \
  --limit 10
```

用于比较“注意力重要性选层”是否优于随机选层。

## 其他通信方法

这些方法主要用于对照。

### Activation Communication

```bash
python com.py \
  --test_task tipsheets \
  --do_test_ac \
  --layer_k 26 \
  --layer_j 26 \
  --f replace
```

A 输出 hidden states，B 在指定层融合 hidden activation。融合方式包括 `replace`、`sum`、`mean`。

### Natural Language Debate

```bash
python com.py \
  --test_task hotpotqa \
  --do_test_nld \
  --nld_max_tokens_model_A_and_B_phase1 256 \
  --sender_aware
```

A 和 B 先分别生成自然语言答案，再让 B 看 A 的自然语言输出并修正答案。

### CIPHER

```bash
python com.py \
  --test_task hotpotqa \
  --do_test_cipher \
  --nld_max_tokens_model_A_and_B_phase1 256 \
  --sender_aware
```

CIPHER 通过 embedding 表示进行对照通信。

## 多 sender 和在线校准

多 sender：

```bash
python com_ms.py \
  --test_task hotpotqa \
  --do_test \
  --model_A1 meta-llama/Llama-3.1-8B-Instruct \
  --model_A2 meta-llama/Llama-3.1-8B-Instruct \
  --model_B meta-llama/Llama-3.1-8B-Instruct \
  --top_layers 0.3 \
  --limit 10
```

在线校准：

```bash
python com_online.py \
  --test_task countries_tipsheets \
  --mix_method concat \
  --do_test \
  --top_layers 0.3 \
  --calib_interval 10 \
  --limit 20
```

第一阶段复现建议先不要碰这两条线，等单 sender KVComm 跑通后再扩展。

## 输出与日志

每次运行会在 `snapshots/<run_name>_<timestamp>/` 下保存日志：

```text
snapshots/
└── <run_name>_<timestamp>/
    └── log.log
```

如果开启 `--use_wandb`，代码会把结果和耗时写到 W&B。

## 和后续研究最相关的改动点

如果你的目标是“抽出有效 KV 作为指标”，建议优先关注这些位置：

- `models.py::prepare_key_cache()`：当前是按层保留完整 KV，未选层保留 attention sink。可以改成按 token、head、层内位置或预算选择。
- `layer_importance.py::calc_layer_importance()`：当前层重要性来自 B 对 A KV 区间的注意力总量。可以尝试更细粒度的重要性指标。
- `model_attn.py`：保存 attention 的 query/key 输入，是做注意力诊断和指标可视化的入口。
- `eval.py::CommunicationEvaluator.inference()`：A 生成 KV、B 接收 KV 的完整调用链。适合插入统计、保存 cache、对比压缩率。
- `dataloader/`：如果要构造更能检验隐空间通信的任务，可以从 `prompt_A`/`prompt_B` 切分方式入手。

一个合理的研究推进顺序：

1. 跑通 baseline、skyline、固定层 KVComm。
2. 跑通 `--top_layers`，记录选出的层和性能。
3. 复现 random selection 消融。
4. 保存每层 attention importance，画 layer curve。
5. 把 selection 从“层级”细化到“层 + token/head 级”。
6. 设计通信预算指标：传输 KV 数量、cache token 数、层数、head 数、性能/预算曲线。

## 常见问题

### transformers 版本不对

请固定：

```bash
pip install transformers==4.53.3
```

代码里对 Llama/Qwen/Gemma attention 类做了替换和追踪，版本变化会导致类路径或 forward 参数不兼容。

### 显存不够

默认会同时加载 A 和 B 两个 8B 模型。先用小样本 `--limit 1` 验证；必要时换更小的 causal LM 做代码路径测试，再回到论文模型。

### Hugging Face 数据集或模型下载失败

确认已经登录：

```bash
huggingface-cli login
```

Llama 模型还需要申请访问权限。

### 自动选层结果不稳定

小 `calib_size` 会带来较大方差。调试阶段可以小一点，正式复现建议增大校准样本数并固定 `--seed`。

### `layers_list` 参数怎么传

命令行中直接写多个整数：

```bash
--layers_list 10 11 12 13
```

默认值 `[-1]` 表示不手动指定层；如果 `--top_layers > 0`，代码会先把所有层用于校准，再自动选择。

## 复现检查清单

- [ ] 能用 `--limit 1` 跑通 baseline。
- [ ] 能用 `--limit 1` 跑通 skyline。
- [ ] 能用手动 `--layers_list` 跑通 KVComm。
- [ ] 能用 `--top_layers` 跑通校准 + 正式测试。
- [ ] 记录 baseline、skyline、KVComm、random selection 的结果。
- [ ] 保存 `log.log` 中的 `Layer ranking` 和 `New layers list`。
- [ ] 明确每次实验的模型、任务、样本数、校准样本数、层比例和随机种子。

