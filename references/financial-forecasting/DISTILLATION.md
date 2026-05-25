# Financial Forecasting: Research Distillation

One-line purpose: distill the listed financial forecasting references into actionable guidance for Cramer-Short forecasting, validation, and research-agent workflows.

## Source Coverage

| PDF | Inferred title | Year/authors if available | Extraction status | Relevance |
|---|---|---|---|---|
| `./1-s2.0-S1566253524005335-main.pdf` | Natural language processing in finance: A survey | 2025; Kelvin Du, Yazhi Zhao, Rui Mao, Frank Xing, Erik Cambria | Extracted via `pdfinfo` and `pdftotext`; 25 pages | Maps NLP applications across financial sentiment, narrative processing, forecasting, portfolio management, risk, compliance, ESG, and explainable AI. |
| `./2601.11958v1.pdf` | Autonomous Market Intelligence: Agentic AI Nowcasting Predicts Stock Returns | 2026; Zefeng Chen, Darcy Pu | Extracted via `pdfinfo` and `pdftotext`; 46 pages | Tests a fully agentic LLM nowcasting workflow on Russell 1000 stocks with explicit look-ahead-bias controls. |
| `./ssrn-5140015.pdf` | Applications of Time Series Analysis in Quantitative Finance | 2025; Yifan Guo | Extracted via `pdfinfo` and `pdftotext`; 21 pages; PDF metadata lacks reliable title/author fields | Reviews ARIMA, GARCH, and machine-learning time-series models for asset prediction, risk management, and portfolio optimization. |

## Practical Synthesis for Cramer-Short

- Treat financial forecasting as a mixed evidence problem: structured time-series models cover temporal dynamics; NLP/LLM tools add narrative, sentiment, and nowcasting signals.
- Separate directional predictions from tradable strategies. The agentic nowcasting paper reports alpha concentrated in the top-ranked Russell 1000 names, while weaker ranks dilute quickly.
- Preserve timestamp discipline. The nowcasting study emphasizes predictions collected at the current edge of time and traded with execution timing that avoids look-ahead bias.
- Use classical time-series checks before model choice: stationarity, autocorrelation, heteroskedasticity, and horizon length determine whether ARIMA, GARCH, or ML is appropriate.
- For text-derived financial signals, require source attribution and data-quality checks; the NLP survey repeatedly flags data quality, privacy, and explainability as finance-specific barriers.

## Implementation Implications

- Add forecast metadata to every agent-generated prediction: input timestamp, data cutoff, forecast horizon, target variable, source mix, and whether the output is directional or portfolio-actionable.
- For financial text workflows, route tasks by evidence type: sentiment/narrative extraction, event-risk summarization, question answering, portfolio commentary, or compliance/ESG monitoring.
- Evaluate strategies after transaction costs, not only raw predictive accuracy. The nowcasting study notes transaction costs below 10% of gross alpha for the tested top-20 strategy, but this is strategy-specific.
- Prefer model ensembles by role: ARIMA-like baselines for low-dimensional autocorrelation, GARCH-like models for volatility, ML/LLM models for nonlinear or narrative-rich settings.
- Keep rank-based LLM nowcasts conservative: use only high-confidence/top-tier outputs unless backtests show that broader ranks retain signal.

## Reliability and Limitations

- The NLP survey is broad and secondary; it supports taxonomy and risk framing more than specific trading performance claims.
- The agentic nowcasting result is recent, time-sensitive, and described as irreproducible once the information environment passes; do not treat its alpha as durable without fresh forward testing.
- `ssrn-5140015.pdf` is a review-style paper; it summarizes common models but does not establish a new benchmark for Cramer-Short.
- PDF metadata is incomplete for `ssrn-5140015.pdf`; title, author, and date are inferred from extracted title-page text.
- All claims here were limited to text extracted locally with `pdfinfo`/`pdftotext`; no external metadata was used.

## Verification Checklist

- [x] `./1-s2.0-S1566253524005335-main.pdf`
- [x] `./2601.11958v1.pdf`
- [x] `./ssrn-5140015.pdf`
