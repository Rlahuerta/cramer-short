# Trading Strategies: Research Distillation

One-line purpose: distill the listed trading-strategy references into actionable design and validation guidance for Cramer-Short trading research workflows.

## Source Coverage

| PDF | Inferred title | Year/authors if available | Extraction status | Relevance |
|---|---|---|---|---|
| `./1-s2.0-S0952197624015239-main.pdf` | An adaptive financial trading strategy based on proximal policy optimization and financial signal representation | 2024; Lin Wang, Xuerui Wang | Extracted via `pdfinfo` and `pdftotext`; 23 pages | Proposes FSRPPO, combining financial signal representation with PPO to reduce noise, trading frequency, costs, and risk. |
| `./2412.20138v7.pdf` | TradingAgents: Multi-Agents LLM Financial Trading Framework | 2025; Yijia Xiao, Edward Sun, Di Luo, Wei Wang | Extracted via `pdfinfo` and `pdftotext`; 38 pages | Proposes specialized LLM agents for fundamental, sentiment, technical, trading, bull/bear research, and risk-management roles. |

## Practical Synthesis for Cramer-Short

- Do not equate price prediction with profitable trading. Both papers emphasize strategy construction, risk, costs, and decision processes beyond raw forecasts.
- Noise handling is central. FSRPPO uses CEEMDAN plus modified rescaled range analysis for signal representation before PPO decision-making.
- Reward design matters. FSRPPO explicitly includes transaction-fee effects and aims to reduce unnecessary trading frequency.
- Multi-agent LLM trading is most useful when roles are separated: analysts gather evidence, bull/bear researchers debate, risk agents monitor exposure, and traders synthesize actions.
- Always evaluate with cumulative return, annualized return, Sharpe ratio, and maximum drawdown; TradingAgents uses these metrics against rule-based baselines.

## Implementation Implications

- Separate forecast generation from trade execution recommendations in the agent UI and logs.
- Add a strategy-validation template: data window, train/test split, transaction-cost model, position sizing, risk limits, benchmark, CR/AR/Sharpe/MDD, and look-ahead-bias checks.
- For reinforcement-learning strategy experiments, encode action space, reward decomposition, and cost model explicitly before training or replay.
- For LLM trading agents, require each role to cite evidence and expose disagreements before the final trade recommendation.
- Include a risk-manager pass that can veto or downsize trades based on drawdown, volatility, concentration, stale data, or conflicting signals.

## Reliability and Limitations

- FSRPPO is an experimental strategy paper; performance depends on market sample, hyperparameters, and whether noise decomposition generalizes.
- TradingAgents reports strong baseline improvements, including at least 23.21% cumulative return and 24.90% annual return on sampled stocks, but this remains a research benchmark rather than production proof.
- LLM-based trading frameworks are vulnerable to retrieval errors, narrative overfitting, prompt sensitivity, and weak calibration.
- Transaction costs, slippage, and execution timing must be modeled independently for Cramer-Short before adopting any strategy claim.
- All extracted claims were checked against local PDF text only.

## Verification Checklist

- [x] `./1-s2.0-S0952197624015239-main.pdf`
- [x] `./2412.20138v7.pdf`
