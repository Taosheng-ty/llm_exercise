# LLM/RL Training Coding Exercises

117 coding exercises (1184 tests) for learning RL-based LLM training concepts, inspired by the [slime](https://github.com/example/slime) codebase and general LLM training patterns.

## Structure

Each exercise has 3 files:
- `problem.py` - Function signatures with docstrings and TODO placeholders (what you implement)
- `solution.py` - Reference solution
- `test_solution.py` - Pytest test cases (import from solution.py; copy and change import to test your own)

## Categories

| # | Category | Exercises | Framework | Difficulty |
|---|----------|-----------|-----------|------------|
| 01 | **RL Fundamentals** - GAE, PPO clipping, KL divergence, GRPO, REINFORCE | 5 | numpy | Easy-Medium |
| 02 | **Reward Functions** - Math normalization, F1 score, reward shaping, outcome RM | 4 | numpy/stdlib | Easy-Medium |
| 03 | **Data Processing** - Chat templates, sequence packing, seq-len balancing, loss masks | 4 | numpy | Easy-Hard |
| 04 | **Distributed Training** - GPU placement, weight sharding, async scheduling | 3 | numpy | Medium-Hard |
| 05 | **Rollout Pipeline** - Data sources, replay buffers, filters, best-of-N sampling | 4 | numpy | Easy-Medium |
| 06 | **Metrics & Logging** - Pass@k, training tracker, compression/repetition detection | 3 | numpy/stdlib | Easy-Medium |
| 07 | **Loss & Masking** - Cross-entropy, log probs, OPSM, dual-clip PPO | 4 | numpy | Easy-Hard |
| 08 | **Attention Mechanisms** - Flash attention, RoPE, GQA, KV cache, sliding window | 8 | PyTorch | Easy-Hard |
| 09 | **Torch RL Training** - PPO/GRPO/KL loss in PyTorch, GAE, entropy, importance sampling | 8 | PyTorch | Easy-Hard |
| 10 | **Model Architecture** - RMSNorm, SwiGLU, LoRA, transformer block, LM head | 8 | PyTorch | Easy-Hard |
| 11 | **Sampling & Decoding** - Top-k/p, beam search, speculative decoding, repetition penalty | 8 | PyTorch/numpy | Easy-Hard |
| 12 | **Distributed Primitives** - All-reduce, tensor parallel, pipeline schedule, gradient accumulation | 7 | PyTorch/numpy | Easy-Hard |
| 13 | **Weight Conversion** - QKV split, gate-up split, LoRA merge, dtype conversion, name mapping | 7 | PyTorch/numpy | Easy-Medium |
| 14 | **Memory & Efficiency** - Gradient checkpointing, mixed precision, FLOPs, CPU offloading | 7 | PyTorch/numpy | Easy-Hard |
| 15 | **Evaluation & Benchmarks** - Perplexity, ECE, majority voting, exact match, MCQA | 7 | PyTorch/numpy | Easy-Medium |
| 16 | **MoE & Routing** - Top-k routing, load balancing, expert dispatch, routing replay | 7 | PyTorch/numpy | Easy-Hard |
| 17 | **Training Loop Patterns** - LR scheduler, DPO loss, EMA, SFT step, curriculum learning | 8 | PyTorch/numpy | Easy-Hard |
| 18 | **RLHF & Alignment** - Bradley-Terry RM, RLOO, IPO, KTO, win-rate/ELO | 5 | PyTorch/numpy | Medium-Hard |
| 19 | **Optimizers & Tokenization** - AdamW internals, BPE, ZeRO sharding, special tokens, constrained decoding | 5 | PyTorch/numpy | Easy-Hard |
| 20 | **Inference & Serving** - PagedAttention, INT8 quantization, ALiBi, cross-attention, multi-token prediction | 5 | PyTorch | Medium-Hard |

## Getting Started

```bash
# Run all solution tests (1184 tests)
python -m pytest exercises/ -v

# Work on an exercise
cp exercises/01_rl_fundamentals/ex01_gae/problem.py exercises/01_rl_fundamentals/ex01_gae/my_solution.py
# Edit my_solution.py to implement the functions
# Then test against the test cases (modify import in test file)
```

## Prerequisites

- Python 3.10+
- numpy
- pytest
- PyTorch (for categories 08-17)

Categories 01-07 use numpy only. Categories 08-17 require PyTorch (CPU is sufficient, no GPU needed).

## Exercise List

### 01 - RL Fundamentals (numpy)
1. **GAE** (Medium) - Generalized Advantage Estimation with gamma/lambda discounting
2. **PPO Clipping** (Medium) - Clipped surrogate objective for policy optimization
3. **KL Divergence** (Easy) - k1/k2/k3 approximation methods
4. **GRPO Advantages** (Easy) - Group Relative Policy Optimization normalization
5. **REINFORCE Baseline** (Medium) - Discounted returns with baseline subtraction

### 02 - Reward Functions (numpy/stdlib)
1. **Math Answer Normalization** (Medium) - Strip LaTeX formatting for answer comparison
2. **F1 Score** (Easy) - Token-level precision/recall/F1
3. **Reward Shaping** (Easy) - Length penalties and format bonuses
4. **Outcome Reward Model** (Medium) - Multi-strategy answer extraction and comparison

### 03 - Data Processing (numpy)
1. **Chat Template** (Easy) - ChatML message formatting
2. **Sequence Packing** (Medium) - Bin-packing variable-length sequences
3. **Sequence Length Balancing** (Hard) - Karmarkar-Karp partitioning algorithm
4. **Loss Mask Generation** (Medium) - Multi-turn SFT loss masks

### 04 - Distributed Training (numpy)
1. **GPU Placement** (Medium) - Actor/critic/rollout GPU allocation
2. **Weight Sharding** (Medium) - Tensor splitting and gathering across workers
3. **Async Training Scheduler** (Hard) - Overlapping generation and training

### 05 - Rollout Pipeline (numpy)
1. **Data Source** (Medium) - Epoch-tracking prompt dataset
2. **Replay Buffer** (Medium) - FIFO + priority experience buffer
3. **Dynamic Sampling Filter** (Easy) - Chainable sample group filters
4. **Best-of-N Sampling** (Medium) - Greedy, weighted, and rejection sampling

### 06 - Metrics & Logging (numpy/stdlib)
1. **Pass@k** (Medium) - Unbiased pass rate estimation
2. **Training Metrics Tracker** (Medium) - Moving averages and anomaly detection
3. **Compression Repetition Detection** (Easy) - zlib ratio and n-gram methods

### 07 - Loss & Masking (numpy)
1. **Cross-Entropy Loss** (Medium) - Numerically stable masked LM loss
2. **Log Probs from Logits** (Easy) - Per-token log probability extraction
3. **Off-Policy Masking** (Easy) - OPSM: KL + advantage-based sequence masking
4. **Dual-Clip PPO** (Hard) - Extended PPO with lower bound clipping

### 08 - Attention Mechanisms (PyTorch)
1. **Scaled Dot-Product Attention** (Medium) - Q@K^T/sqrt(d_k) with causal mask
2. **Multi-Head Attention** (Medium) - Head splitting, parallel attention, concatenation
3. **Flash Attention Tiling** (Hard) - Block-wise attention with online softmax trick
4. **Causal Mask** (Easy) - Lower-triangular autoregressive mask generation
5. **Rotary Positional Embedding** (Hard) - RoPE rotation matrices for Q, K
6. **Grouped Query Attention** (Medium) - GQA with KV head expansion
7. **KV Cache** (Medium) - Incremental cache for autoregressive decoding
8. **Sliding Window Attention** (Medium) - Local attention with window size limit

### 09 - Torch RL Training (PyTorch)
1. **Policy Gradient Loss** (Medium) - PPO clipped loss with autograd
2. **GAE in PyTorch** (Medium) - Vectorized GAE with batch support
3. **Value Function Loss** (Easy) - Clipped value loss for critic training
4. **Entropy Bonus** (Easy) - Entropy computation from logits
5. **Importance Sampling Ratio** (Medium) - Per-token ratios with TIS clipping
6. **Reward Normalization** (Easy) - Running EMA normalization
7. **KL Penalty Loss** (Medium) - Per-token KL with loss mask
8. **GRPO Loss** (Hard) - Full GRPO: group advantages + PPO clip + KL penalty

### 10 - Model Architecture (PyTorch)
1. **RMSNorm** (Easy) - Root Mean Square Layer Normalization
2. **SwiGLU** (Easy) - Gated activation for FFN
3. **Transformer Block** (Hard) - Full decoder block with pre-norm residuals
4. **Positional Encoding** (Easy) - Sinusoidal position embeddings
5. **LoRA Linear** (Medium) - Low-rank adaptation with merge/unmerge
6. **Embedding with Tying** (Medium) - Shared input/output embeddings
7. **Residual Stream** (Easy) - Pre-norm residual connections with scaling
8. **Simple LM Head** (Medium) - Minimal GPT-style language model

### 11 - Sampling & Decoding (PyTorch + numpy)
1. **Temperature Scaling** (Easy) - Logit temperature adjustment
2. **Top-K Sampling** (Easy) - Keep only top-k logits
3. **Top-P Sampling** (Medium) - Nucleus sampling with cumulative probability
4. **Repetition Penalty** (Easy) - Penalize repeated tokens
5. **Beam Search** (Hard) - Multi-beam decoding with length normalization
6. **Speculative Decoding** (Hard) - Draft-verify acceleration
7. **Logit Processor Chain** (Medium) - Composable logit transformations
8. **Stop Criteria** (Easy) - Token/length/string stopping conditions

### 12 - Distributed Primitives (PyTorch + numpy)
1. **All-Reduce Simulation** (Medium) - Sum/mean/max across virtual workers
2. **Gradient Accumulation** (Medium) - Micro-batch gradient accumulation
3. **Distributed Whitening** (Medium) - Global advantage normalization
4. **Tensor Parallel Linear** (Hard) - Column and row parallel linear layers
5. **Pipeline Schedule** (Medium) - GPipe scheduling with bubble ratio
6. **Data Parallel Partitioning** (Easy) - Contiguous/interleaved/balanced splits
7. **Checkpoint Sharding** (Medium) - Shard state dict across files

### 13 - Weight Conversion (PyTorch + numpy)
1. **QKV Split** (Medium) - Split fused QKV weight for GQA
2. **Gate-Up Split** (Easy) - Split fused MoE gate/up projections
3. **Weight Name Mapping** (Medium) - Megatron to HuggingFace name conversion
4. **Dtype Conversion** (Easy) - fp32/bf16/fp16/fp8 conversion with error tracking
5. **Tensor Hash Verification** (Easy) - Checkpoint integrity via uint32 hashing
6. **LoRA Merge** (Medium) - Merge/unmerge LoRA adapters into base model
7. **Expert Weight Reorder** (Medium) - Reorder MoE experts by routing frequency

### 14 - Memory & Efficiency (PyTorch + numpy)
1. **Gradient Checkpointing** (Hard) - Recompute activations during backward
2. **Mixed Precision Training** (Medium) - fp16/bf16 forward with loss scaling
3. **Activation Memory Estimation** (Medium) - Predict memory usage from config
4. **CPU Offloading** (Medium) - Parameter offload/onload for memory saving
5. **FLOPs Counter** (Medium) - Count FLOPs for transformer operations
6. **Memory Budget Planner** (Medium) - GPU memory allocation planning
7. **Throughput Calculator** (Easy) - tokens/sec, TFLOPs, MFU metrics

### 15 - Evaluation & Benchmarks (PyTorch + numpy)
1. **Perplexity** (Medium) - Compute PPL from masked log probabilities
2. **Calibration Metrics** (Medium) - Expected Calibration Error (ECE)
3. **Majority Voting** (Medium) - Self-consistency with weighted voting
4. **Exact Match** (Easy) - Normalized text comparison
5. **MCQA Evaluation** (Medium) - Multiple-choice answer extraction
6. **Eval Config Builder** (Medium) - Hierarchical config resolution
7. **Benchmark Aggregation** (Easy) - Macro/weighted averages with bootstrap CI

### 16 - MoE & Routing (PyTorch + numpy)
1. **Top-K Routing** (Medium) - Expert selection with router logits
2. **Load Balancing Loss** (Medium) - Auxiliary loss for uniform utilization
3. **Expert Parallel Dispatch** (Hard) - Token scatter/gather through experts
4. **Routing Replay** (Medium) - Cache and replay routing decisions
5. **Shared Expert** (Medium) - Shared + routed expert architecture
6. **Expert Frequency Analysis** (Easy) - Utilization and dead expert detection
7. **Capacity Factor** (Medium) - Expert token capacity limiting

### 17 - Training Loop Patterns (PyTorch + numpy)
1. **Learning Rate Scheduler** (Medium) - Warmup-cosine and warmup-linear decay
2. **Gradient Clipping** (Easy) - Global norm clipping
3. **EMA Model** (Medium) - Exponential moving average of weights
4. **SFT Training Step** (Medium) - Masked cross-entropy for supervised fine-tuning
5. **DPO Loss** (Hard) - Direct Preference Optimization
6. **Curriculum Scheduler** (Medium) - Difficulty-based data ordering
7. **Training State Manager** (Medium) - Checkpointing with early stopping
8. **Online vs Offline RL** (Medium) - Data flow simulation and staleness

### 18 - RLHF & Alignment (PyTorch + numpy)
1. **Bradley-Terry Reward Model** (Hard) - Pairwise ranking loss for reward model training
2. **RLOO Advantages** (Medium) - REINFORCE Leave-One-Out variance reduction
3. **IPO Loss** (Medium) - Identity Preference Optimization squared-loss variant
4. **KTO Loss** (Medium) - Kahneman-Tversky Optimization with unpaired feedback
5. **Win-Rate & ELO** (Medium) - Pairwise comparison metrics and rating systems

### 19 - Optimizers & Tokenization (PyTorch + numpy)
1. **AdamW Optimizer** (Medium) - First/second moment, bias correction, decoupled weight decay
2. **BPE Tokenizer** (Medium) - Byte-Pair Encoding training, encoding, and decoding
3. **ZeRO Optimizer Sharding** (Hard) - Stage 1/2/3 optimizer state partitioning
4. **Special Token Handler** (Easy) - BOS/EOS/PAD management and batch padding
5. **Constrained Decoding** (Medium) - FSM-based structured output generation

### 20 - Inference & Serving (PyTorch)
1. **Paged Attention** (Hard) - Block-based KV cache with allocation/free
2. **INT8 Quantization** (Medium) - Symmetric/asymmetric per-channel quantization
3. **ALiBi Attention** (Medium) - Attention with Linear Biases for position encoding
4. **Cross-Attention** (Medium) - Encoder-decoder cross-attention mechanism
5. **Multi-Token Prediction** (Medium) - N-ahead prediction heads from single hidden state
