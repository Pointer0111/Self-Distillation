# SDFT（路线2）移植改进点清单（目标：在自建服务器上基于 Self-Distillation 跑 Qwen3-8B）

目标：以 `https://github.com/idanshen/Self-Distillation` 作为“可跑骨架”，把 `tinker-cookbook` 里的关键改动移植进去，在你自己的训练栈（Transformers/TRL/Accelerate/DeepSpeed 等）上训练 `Qwen/Qwen3-8B`。

本清单只记录“需要移植/实现的改进点”，不包含环境安装与命令细节。

---

## 1. 核心算法改动（必须做）

### 1.1 Full-vocab KL → Top-K 蒸馏（K=20）

动机：原论文/官方实现使用全词表 forward KL；但工程上全词表 KL 成本高且依赖全 logits。cookbook 的实现使用 Top-K 近似并验证效果等价。

你需要实现：
- 对每个 student completion 的 token 位置 `t`，从 teacher 的 logits 取 Top-K token id 与 logprob
- 在 Top-K 上做归一化得到 `p_teacher_topk`
- 取 student 在这些 Top-K token 上的 `log p_student`
- 损失按 token 平均：
  - `L = mean_t( - sum_{k=1..K} p_teacher_topk(k|t) * log p_student(k|t) )`

关键细节（对齐 cookbook/reference）：
- 默认 `K=20`
- 对每条 completion 默认跳过前 `skip_first_n_tokens=3` 个 token 不计入蒸馏损失（参考 cookbook 的默认设置）
- teacher/student 都是同一个 base model 起步（同 tokenizer / 同 chat template）

建议增加的开关：
- `--distill_mode fullkl|topk`
- `--topk 20`
- `--skip_first_n_tokens 3`

对应参考实现位置（tinker-cookbook）：
- Top-K 权重与 target 构造逻辑：`tinker_cookbook/distillation/sdft.py` 中 `build_topk_distillation_datums(...)`

---

### 1.2 EMA teacher → Static teacher（默认）

动机：官方实现 teacher 为 student 的 EMA；cookbook 默认 teacher 冻结为初始 base weights，避免 EMA 权重同步/维护开销，且消融验证差异很小。

你需要实现：
- 初始化时 `teacher = deepcopy(base_model)` 并冻结梯度
- 训练过程中 teacher 不更新（默认）

可选（保持可回退/做对照）：
- 增加 `--teacher_mode static|ema`
- 若你仍想保留“近似 EMA 的工程折中”，可以做 `teacher_sync_every=N`：每 N step 用 student 权重覆盖 teacher（不是 EMA，但便于实验对照）

对应参考说明（tinker-cookbook）：
- recipe README 中 “Static teacher instead of EMA”

---

## 2. Prompt 与 token 对齐（必须做，否则 loss 对不齐）

### 2.1 Teacher prompt：把 golden answer 作为 in-context demo

动机：SDFT 的 teacher 之所以能给出更好的 token 分布，是因为它“看到了问题 + 标准答案（演示）”。

你需要实现：
- teacher 的输入不是单纯 question，而是类似：
  - `{question}`
  - `This is an example for a response to the question:`
  - `{golden_answer}`
  - `Now answer with a response of your own, including the thinking process.`
- student 的输入仍是原本 question（或原本对话 prompt），student 在其后生成 completion
- 计算 teacher 的 token 分布时，要对 student 的 completion 做 teacher-forcing（teacher 输入 = teacher_prompt + completion_tokens）

关键点：
- 必须严格对齐“哪些 token 属于 completion”，也就是训练时 mask 覆盖的那部分 token
- teacher prompt 太长时要截断：保证 `len(teacher_prompt) + len(completion) <= max_context_length`

对应参考实现位置（tinker-cookbook）：
- `tinker_cookbook/distillation/sdft.py` 中 `DEFAULT_DEMO_TEMPLATE` 与 `build_sdft_teacher_prompt(...)`

---

## 3. 数据与格式适配（推荐做，尤其你用 Qwen3 thinking）

### 3.1 Science 数据：支持 thinking 输出格式（推荐）

动机：官方 science 数据通常是 `<reasoning>...</reasoning><answer>...</answer>`；Qwen3/3.5 等 thinking 模型更常见 `<think>...</think>` 与 “The answer is X.”。

你需要实现（可选，但很建议）：
- 把 golden answer 从 XML 格式转换成更适合 thinking 模型的格式，例如：
  - SFT：`<think>\n{reasoning}\n</think>\n\nThe answer is {X}.`
  - SDFT teacher demo：`{reasoning}\n\nThe answer is {X}.`
- 评测时支持从多种输出里鲁棒抽取答案（见第 4 节）

对应参考实现位置（tinker-cookbook）：
- `tinker_cookbook/recipes/sdft/datasets.py` 中 `_convert_golden_answer_to_thinking_format(...)`

---

## 4. 评测口径（强烈建议做，保证对齐论文/官方）

### 4.1 Science：答案抽取的鲁棒性

你需要实现：
- 从模型输出中抽取最终选项字母（A-D），兼容：
  - `<answer>...</answer>`
  - “The answer is X”
  - `</think>` 之后单独一行 `A/B/C/D`
  - 兜底：文本中最后一个独立的 A-D

对应参考实现位置（tinker-cookbook）：
- `tinker_cookbook/recipes/sdft/eval.py` 中 `extract_xml_answer(...)` 与 `evaluate_science_correctness(...)`

---

### 4.2 Tool-use：Action / Action Input 精确匹配

你需要实现：
- 从输出中抽取所有 `Action:` 字段（动作序列）
- 从输出中抽取所有 `Action Input:` JSON，并合并成一个 dict
- 与 GT 的动作多重集合（Counter）与参数 dict 做精确匹配

对应参考实现位置（tinker-cookbook）：
- `tinker_cookbook/recipes/sdft/eval.py` 中 `evaluate_tooluse_correctness(...)`

---

## 5. 实验脚手架（推荐做，方便你复现论文关键结论）

### 5.1 Continual Learning 两阶段实验（Stage1 tooluse → Stage2 science）

建议你实现一个脚本/配置，跑如下对照：
- Stage 1：在 tooluse 上训练；评估 tooluse + science（看是否“遗忘” science）
- Stage 2：从 Stage1 checkpoint 继续在 science 上训练；评估 science + tooluse retention

建议至少对比这两条线：
- SFT baseline（标准交叉熵对 golden answer）
- SDFT-TopK（本清单第 1.1/1.2）

对应参考实现位置（tinker-cookbook）：
- `tinker_cookbook/recipes/sdft/run_continual_learning.py`（组织方式与日志结构可参考）

---

## 6. 参数建议（针对 Qwen3-8B 的起步配置）

必须参数：
- `model_name = Qwen/Qwen3-8B`
- `topk = 20`
- `skip_first_n_tokens = 3`
- `teacher_mode = static`（先用静态 teacher，把变量降到最少）

建议保留的开关（方便做对照/排查）：
- `teacher_mode static|ema`
- `distill_mode fullkl|topk`
- `max_context_length`
- `temperature`（student rollout 的采样温度）

---

## 7. 最小实现顺序（推荐按这个顺序落地）

1) 先用 Self-Distillation 原版跑通 Qwen3-8B（SFT 或原版 SDFT 任意）  
2) 加入 Static teacher（不动 loss）  
3) 加入 Top-K 蒸馏 loss（K=20 + skip_first_n_tokens=3）  
4) 补齐 science 的 thinking 格式转换与答案抽取  
5) 补齐 tooluse 的 action/input 精确评测  
6) 搭建 Stage1/Stage2 continual learning 脚本，复现“遗忘 vs 保持”的差异  

