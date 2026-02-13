# CLAUDE.md — Tactical Asset Allocation System

## Identity & Role

You are building a production-grade Tactical Asset Allocation system for a university dissertation (UWE Bristol, UFCEKP-30-3). The standard is institutional, not academic. Every decision you make should be defensible to a quantitative portfolio manager, not just a marking rubric. The full system specification lives in `execution_blueprint.md` — read it before every session and treat it as your source of truth.

## Environment

- Windows, VSCode, Python 3.11 virtual environment (`.venv`)
- PyTorch CPU-only (no CUDA). This means:
  - Keep model parameters under 500K total. Start TCN hidden channels at 32, not 64
  - Hyperparameter search budget: 15 configurations max, not 50
  - Default batch size: 64. Do not exceed 128
  - If any single training run takes longer than 20 minutes, stop and reduce complexity
- Always verify `.venv` is active before running anything
- Never install packages globally. Always `pip install` inside the venv

## Master Plan

The project has 8 phases and 6 gates defined in `execution_blueprint.md`. You must:
- Work through phases sequentially. Never jump ahead
- At each gate, print a clear PASS/FAIL verdict with evidence before continuing
- If a gate fails, propose exactly 3 corrective actions ranked by likelihood of fixing the issue
- Never silently skip a gate or mark it as passed without running the actual checks

## Hard Constraints — Violating Any of These Invalidates the Entire Project

1. **No look-ahead bias.** Every feature at time t must use only data available at or before time t. Every normalization statistic (mean, std) must come from a rolling backward-looking window (252 days). Global statistics across the full dataset are NEVER acceptable. When in doubt, write a unit test to prove no leakage exists.

2. **No training on test data.** The test set is touched exactly once, at the very end, during Phase 7. If you find yourself looking at test metrics to "debug" or "tune," you have contaminated the evaluation. Stop, revert, and use the validation set instead.

3. **Transaction costs everywhere.** 10 basis points per unit of turnover, included in both the differentiable training loss AND the backtest engine. A strategy that looks good without costs but falls apart with costs is worthless.

4. **Portfolio constraints by construction.** Softmax output layer enforces weights ≥ 0 and sum-to-1. Maximum single-asset weight of 30% enforced via clamping and renormalization after softmax. These are not optional post-processing steps — they are part of the architecture.

5. **Reproducibility.** Set `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.use_deterministic_algorithms(True)` at the top of every script. Pin all package versions in `requirements.txt`. Anyone re-running the code must get results within ±5%.

## Decision-Making Priorities

When you face a design choice, apply these priorities in order:

1. **Correctness over performance.** A model with Sharpe 0.6 and zero data leakage is infinitely better than one with Sharpe 1.5 and contaminated features. Always verify correctness first.

2. **Simplicity over cleverness.** If two approaches achieve similar results, pick the one with fewer moving parts. Don't add complexity to impress — add it only when a simpler version has demonstrably failed.

3. **Robustness over peak performance.** A model that delivers Sharpe 0.7 across all 3 test folds is better than one that delivers 1.5 on one fold and -0.2 on the others. Stability across regimes is the goal.

4. **Transparency over black-box.** Every model output should be interpretable. Log weight allocations, plot attention maps, run feature ablation. If you can't explain why the model made a decision, the model is not ready.

## How to Write Code for This Project

- **Modular.** One function does one thing. One file owns one responsibility. Follow the directory structure in `execution_blueprint.md` Section 7.2 exactly.
- **Typed.** Use type hints on all function signatures. This is a quantitative system — ambiguous types cause silent bugs with catastrophic consequences.
- **Tested.** Every component that touches data must have at least one assertion or unit test that validates its output shape, value range, and temporal integrity.
- **Config-driven.** No magic numbers in code. Every hyperparameter, file path, date range, and threshold lives in `config.yaml`. Functions read from config, not from hardcoded values.
- **Logged.** Print progress at every meaningful step. When training, log epoch, train loss, validation loss, validation Sharpe, elapsed time. When backtesting, log date range, number of rebalances, total turnover, final portfolio value.

## Common Failure Modes — Check for These Constantly

| Failure | How to Detect | What to Do |
|---------|--------------|------------|
| Look-ahead bias in features | Run the truncation test from Section 5.1 of the blueprint | Fix the feature computation to use only backward-looking windows |
| Softmax collapse (all weight on 1 asset) | Monitor weight entropy; should be > 0.5 × ln(N) | Add entropy regularization to the loss, or increase softmax temperature |
| Training Sharpe >> Validation Sharpe | Compare at end of each epoch | Increase dropout, reduce model size, add L2 regularization |
| NaN in loss or gradients | Check after every backward pass | Clip gradients to max norm 1.0, check for division by zero in Sharpe computation (add epsilon=1e-8 to denominator) |
| Model outputs equal weights | Compute cosine similarity to 1/N vector | Model has collapsed — increase learning rate, reduce regularization, verify features have signal |
| Backtest returns look unrealistically good | Sharpe > 3.0 on any period | Almost certainly a bug. Check for future data leakage, verify execution uses next-day open prices |
| Extremely high turnover | Monthly turnover > 100% | Increase turnover penalty λ in loss function |

## When You're Stuck

If a phase isn't working after 3 attempts:
1. State exactly what's failing and what you've already tried
2. Check whether the issue is in data, features, model, or evaluation — isolate the layer
3. Simplify: reduce to 5 assets, 2 years of data, smallest model. If it works small but fails large, the issue is scale-related. If it fails small too, the issue is fundamental
4. Never paper over a bug with a workaround. Find the root cause

## File Naming and Git Discipline

- Snake_case for all Python files and functions
- Every script must be runnable standalone: `python features/engineering.py` should work
- Add a docstring at the top of every file explaining what it does and which phase/gate it belongs to
- After each completed phase, suggest a git commit message summarizing what was built

## Key Technical Reminders

- Sharpe annualization: multiply daily Sharpe by √252, not 252
- Portfolio return: `r_portfolio = sum(w_i * r_i)` where r_i is the simple daily return of asset i
- Turnover: `sum(abs(w_new - w_old))` across all assets
- Always compute returns from adjusted close, never raw close
- Walk-forward retraining uses expanding windows, not sliding windows
- Embargo of 21 trading days between train end and test start in each fold
- Backtest executes at next-day open, not same-day close
