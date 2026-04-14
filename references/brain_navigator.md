# Brain Navigator Summary

Source: `/Users/hoddukzoa/Downloads/Brain Navigator Content.pdf`

## Core Concepts

- BRAIN alpha is an expression combining data fields and operators.
- A simulation backtests alpha logic and returns IS metrics/checks.
- Basic onboarding examples focus on reversion-like ideas:
  - `-returns`
  - `rank(-returns)`

## Metrics Interpretation

- `Sharpe`: return efficiency versus volatility. Higher is generally better.
- `Fitness`: combined quality signal balancing profitability and stability.
- `Turnover`: average position change rate; very high turnover can imply cost sensitivity.
- `PnL`: cumulative profit and loss over the test period.

## Practical Guidance for This Skill

- Use dataset intent from user language to select plausible fields first.
- Prefer simple and interpretable first expressions before complex chains.
- In report narration, explain why a candidate may pass/fail checks, not just score values.
- When live simulation is blocked by quota/rate limit, still provide dry-run candidate rationale.

## Korean Reporting Tone

- Keep output short and actionable.
- Include:
  - selected dataset reason
  - expression intent (what behavior it captures)
  - metric reading (Sharpe/Fitness/Turnover)
  - checks-based risk note
