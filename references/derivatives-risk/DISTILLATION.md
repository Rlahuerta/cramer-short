# Derivatives Risk: Research Distillation

One-line purpose: convert the derivatives, volatility, and AI-forecasting risk references into practical guardrails for Cramer-Short forecasts, options interpretation, and risk warnings.

## Source Coverage

| PDF | Inferred title | Year/authors if available | Extraction status | One-sentence relevance |
|---|---|---|---|---|
| `./2602.14350v1.pdf` | *Hidden Risks and Optionalities in American Options* | 2026; Noura El Hassan, Bacel Maddah, Nassim N. Taleb | `pdfinfo` metadata checked; `pdftotext` full text extracted | Shows that American options can carry hidden convexity when rates, carry, exercise timing, liquidity, or model parameters are treated as fixed. |
| `./risks-14-00089-v2.pdf` | *Hidden Optionalities in American Options* | 2026; Noura El Hassan, Bacel Maddah, Nassim Nicholas Taleb | `pdfinfo` metadata checked; `pdftotext` full text extracted | Journal version of the hidden-optionality argument, with simulations across equity puts, currency options, and stochastic-rate dynamics. |
| `./ssrn-2715517.pdf` | *A Practical Guide to Quantitative Volatility Trading* | 2016; Daniel Bloch | `pdfinfo` checked; title/author/year inferred from first pages; `pdftotext` full text extracted | Broad volatility-trading manual linking multifractal returns, inefficient markets, implied-volatility surfaces, VaR limits, variance swaps, and dispersion trades. |
| `./ssrn-5440116.pdf` | *When LLMs Go Abroad: Foreign Bias in AI Financial Predictions* | 2026; Sean Cao, Charles C.Y. Wang, Xiang Yi | `pdfinfo` checked; title/author/year inferred from first pages; `pdftotext` full text extracted | Not a derivatives paper, but relevant to model-risk controls because it documents systematic LLM forecast bias from asymmetric information availability. |

## Practical Synthesis for Cramer-Short

- Treat listed option data as both a price signal and a risk signal: implied move, skew, and term structure encode tail demand, liquidity, and hedging pressure, not just expected volatility.
- When interpreting American-style options, add a hidden-optionality warning if the thesis depends on dividends, funding rates, foreign rates, borrow/carry, or likely early exercise.
- Do not reduce derivatives exposure to a single Black-Scholes-style point estimate; surface sensitivities to volatility, rate/carry shifts, moneyness, and exercise horizon.
- For volatility trades, prefer arbitrage-aware surface checks: calendar monotonicity, strike consistency, put-call parity where applicable, and liquidity-aware quotes before using IV as an input.
- Flag LLM-generated financial predictions as potentially jurisdiction- and source-biased; cross-border names need local-language/local-market context before forecasts are treated as reliable.
- For short theses, separate “option market is pricing high vol” from “equity is likely to fall”; the volatility surface can reflect crash insurance, event demand, dividend/carry effects, or dealer positioning.

## Modeling / Risk Implications

- **Hidden optionality:** American options embed a stochastic stopping-time problem. If Cramer-Short reports option-derived downside or upside, it should note that deterministic carry/rate assumptions can understate value and hedge sensitivity.
- **Fugit / exercise-time uncertainty:** Exercise horizon is itself a distribution, not a fixed date. A practical approximation is to compare deterministic exercise-time output against perturbed rate/carry scenarios and report direction and magnitude of change.
- **Rate and carry shocks:** Currency options, dividend-paying equities, hard-to-borrow shares, and rate-sensitive trades need scenario runs for both level and sign changes in `r1 - r2`; the papers explicitly warn that sign flips can create structural errors.
- **Monte Carlo caution:** Plain Monte Carlo is not automatically enough for American hidden optionality because stopping time is path-dependent and endogenous; least-squares Monte Carlo or hybrid methods are more appropriate when pricing precision matters.
- **Volatility surface risk:** The volatility guide supports using surface shape, skew, term structure, and dispersion relationships as diagnostics, but warns that VaR and Greek approximations inherit model dependence from the assumed surface dynamics.
- **Tail and multifractal behavior:** Financial time series can show heavy tails, long-range dependence in volatility amplitudes, recurrent crises, and nonlinear cycles; risk modules should avoid Gaussian-only extrapolation when estimating option or volatility-trade loss.
- **Forecasting model risk:** The foreign-bias paper implies that LLM forecasts should be audited against alternative models and richer local information, especially for cross-border securities where news coverage is uneven.

## Reliability and Limitations

- The two American-option PDFs substantially overlap; treat the *Risks* article as the more polished version and the arXiv file as supporting coverage, not as fully independent evidence.
- `ssrn-2715517.pdf` is a long working-paper/manual; this distillation uses its abstract, contents, and sampled risk/volatility sections, not a line-by-line replication of all 327 pages.
- `ssrn-5440116.pdf` appears in the derivatives-risk folder but is about AI financial prediction bias, not derivatives; its inclusion here is limited to forecasting/model-risk implications.
- Numerical magnitudes from the hidden-optionality simulations were not transcribed except where the extracted text gave qualitative direction; use the source before quoting exact table or figure values.
- All metadata is from local PDF metadata or extracted first pages; missing/blank PDF metadata was not supplemented from external sources.

## Verification Checklist

- [x] `./2602.14350v1.pdf` covered.
- [x] `./risks-14-00089-v2.pdf` covered.
- [x] `./ssrn-2715517.pdf` covered.
- [x] `./ssrn-5440116.pdf` covered.
