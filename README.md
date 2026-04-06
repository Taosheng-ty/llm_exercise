# LLM/RL Training Coding Exercises

27 coding exercises for learning RL-based LLM training concepts, inspired by the [slime](https://github.com/example/slime) codebase.

## Structure

Each exercise has 3 files:
- `problem.py` - Function signatures with docstrings and TODO placeholders (what you implement)
- `solution.py` - Reference solution
- `test_solution.py` - Pytest test cases (import from solution.py; copy and change import to test your own)

## Categories

| # | Category | Exercises | Difficulty |
|---|----------|-----------|------------|
| 01 | **RL Fundamentals** - GAE, PPO clipping, KL divergence, GRPO, REINFORCE | 5 | Easy-Medium |
| 02 | **Reward Functions** - Math normalization, F1 score, reward shaping, outcome RM | 4 | Easy-Medium |
| 03 | **Data Processing** - Chat templates, sequence packing, seq-len balancing, loss masks | 4 | Easy-Hard |
| 04 | **Distributed Training** - GPU placement, weight sharding, async scheduling | 3 | Medium-Hard |
| 05 | **Rollout Pipeline** - Data sources, replay buffers, filters, best-of-N sampling | 4 | Easy-Medium |
| 06 | **Metrics & Logging** - Pass@k, training tracker, compression/repetition detection | 3 | Easy-Medium |
| 07 | **Loss & Masking** - Cross-entropy, log probs, OPSM, dual-clip PPO | 4 | Easy-Hard |

## Getting Started

```bash
# Run all solution tests (should be 285 passing)
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

No GPU or torch required - all exercises use numpy for portability.

## Exercise List

### 01 - RL Fundamentals
1. **GAE** (Medium) - Generalized Advantage Estimation with gamma/lambda discounting
2. **PPO Clipping** (Medium) - Clipped surrogate objective for policy optimization
3. **KL Divergence** (Easy) - k1/k2/k3 approximation methods
4. **GRPO Advantages** (Easy) - Group Relative Policy Optimization normalization
5. **REINFORCE Baseline** (Medium) - Discounted returns with baseline subtraction

### 02 - Reward Functions
1. **Math Answer Normalization** (Medium) - Strip LaTeX formatting for answer comparison
2. **F1 Score** (Easy) - Token-level precision/recall/F1
3. **Reward Shaping** (Easy) - Length penalties and format bonuses
4. **Outcome Reward Model** (Medium) - Multi-strategy answer extraction and comparison

### 03 - Data Processing
1. **Chat Template** (Easy) - ChatML message formatting
2. **Sequence Packing** (Medium) - Bin-packing variable-length sequences
3. **Sequence Length Balancing** (Hard) - Karmarkar-Karp partitioning algorithm
4. **Loss Mask Generation** (Medium) - Multi-turn SFT loss masks

### 04 - Distributed Training
1. **GPU Placement** (Medium) - Actor/critic/rollout GPU allocation
2. **Weight Sharding** (Medium) - Tensor splitting and gathering across workers
3. **Async Training Scheduler** (Hard) - Overlapping generation and training

### 05 - Rollout Pipeline
1. **Data Source** (Medium) - Epoch-tracking prompt dataset
2. **Replay Buffer** (Medium) - FIFO + priority experience buffer
3. **Dynamic Sampling Filter** (Easy) - Chainable sample group filters
4. **Best-of-N Sampling** (Medium) - Greedy, weighted, and rejection sampling

### 06 - Metrics & Logging
1. **Pass@k** (Medium) - Unbiased pass rate estimation
2. **Training Metrics Tracker** (Medium) - Moving averages and anomaly detection
3. **Compression Repetition Detection** (Easy) - zlib ratio and n-gram methods

### 07 - Loss & Masking
1. **Cross-Entropy Loss** (Medium) - Numerically stable masked LM loss
2. **Log Probs from Logits** (Easy) - Per-token log probability extraction
3. **Off-Policy Masking** (Easy) - OPSM: KL + advantage-based sequence masking
4. **Dual-Clip PPO** (Hard) - Extended PPO with lower bound clipping
