# LLM Exercise Repo

## Project overview

117 coding exercises (1184 tests) for learning RL-based LLM training, inspired by the slime codebase and general LLM training patterns. Each exercise lives in its own directory with `problem.py` (stubs), `solution.py` (reference), and `test_solution.py` (pytest tests).

Categories 01-07 use numpy only. Categories 08-20 require PyTorch (CPU-only is fine).

## Test runner

```bash
python -m pytest exercises/ -v
```

pytest.ini sets `--import-mode=importlib` which is **required** — do not remove it.

## Import pattern for test files

Every exercise directory has identically-named files (`solution.py`, `test_solution.py`). This causes module name collisions under default pytest import mode. Two patterns coexist in the repo; both work with `--import-mode=importlib`:

**Pattern A — importlib path-based (preferred for new exercises):**
```python
import importlib.util, os
_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MyFunc = _mod.MyFunc
```

**Pattern B — relative import (requires `__init__.py` in exercise dir):**
```python
from .solution import MyFunc
```

When creating new exercises, use Pattern A. It does not require `__init__.py` files and avoids all module collision issues.

**Never use bare `from solution import ...`** — this breaks when running the full test suite because Python caches the first `solution` module it finds.

## Adding a new exercise

1. Create `exercises/<category>/ex<NN>_<name>/` with `problem.py`, `solution.py`, `test_solution.py`
2. Use Pattern A imports in test files
3. No `__init__.py` needed
4. Verify: `python -m pytest exercises/<category>/ex<NN>_<name>/ -v`
5. Then verify full suite still passes: `python -m pytest exercises/ -v`

## Dependencies

- `numpy` and `pytest` — required for all exercises
- `torch` (PyTorch) — required for categories 08-20 (CPU is sufficient, no GPU needed)
