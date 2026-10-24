# Forecasting Methods: Research Distillation

One-line purpose: distill the listed forecasting-method references into model-selection and validation guidance for probabilistic and time-series forecasting.

## Source Coverage

| PDF | Inferred title | Year/authors if available | Extraction status | Relevance |
|---|---|---|---|---|
| `./2605.20119v1.pdf` | Toto 2.0: Time Series Forecasting Enters the Scaling Era | 2026; Emaad Khwaja, Chris Lettieri, Gerald Woo, Eden Belouadah, Marc Cenac, Guillaume Jarry, Enguerrand Paquin, Xunyi Zhao, Viktoriya Zhukova, Othmane Abou-Amal, Chenghao Liu, Ameet Talwalkar, David Asker | Extracted via `pdfinfo` and `pdftotext`; 19 pages | Shows time-series foundation models can improve with scale and reports state-of-the-art benchmark results on BOOM, GIFT-Eval, and TIME. |
| `./2605.21041v1.pdf` | Conditioning Gaussian Processes on Almost Anything | 2026; Henry B. Moss, Lachlan Astfalck, Thomas Cowperthwaite, Colin Doumont, Sam Willis, Philipp Hennig, Christopher Nemeth, Andrew Zammit-Mangion | Extracted via `pdfinfo` and `pdftotext`; 36 pages | Generalizes GP conditioning beyond linear-Gaussian observations using diffusion/ODE sampling and point-wise likelihood guidance. |
| `./Kalman1960.pdf` | A New Approach to Linear Filtering and Prediction Problems | 1960; R. E. Kalman | Extracted via `pdfinfo` and `pdftotext`; 12 pages; embedded PDF title is not the article title | Foundational state-space filtering and prediction method for noisy dynamic systems, nonstationary statistics, and optimal linear filters. |

## Practical Synthesis for Cramer-Short

- Match the method to the uncertainty structure: Kalman filters for linear state-space tracking, GPs for probabilistic functions and uncertainty, foundation models for broad time-series pattern transfer.
- Keep simple baselines. Modern large time-series models should beat naive, seasonal, ARIMA/state-space, and domain-specific baselines before they are trusted.
- Use probabilistic metrics when forecasts are distributions. Toto 2.0 reports CRPS-rank-style benchmark comparisons; Cramer-Short should avoid evaluating only point error.
- For real-world constraints, distinguish conditioning data types: numeric observations, nonlinear constraints, natural-language priors, and expert rules require different likelihood treatments.
- State-space filtering remains useful for online updates because it explicitly models latent state, observation noise, and prediction/update cycles.

## Implementation Implications

- Add a forecast-method selector that records why a method was chosen: online filtering, uncertainty quantification, long-horizon transfer, nonlinear constraints, or natural-language conditioning.
- For time-series foundation models, validate on contamination-resistant and out-of-sample slices; the Toto 2.0 paper explicitly highlights contamination-resistant TIME.
- Represent forecasts as distributions where possible: mean/median, interval, scenario samples, and calibration diagnostics.
- For Kalman-style workflows, maintain explicit transition, observation, process-noise, and measurement-noise assumptions in the run log.
- For GP-style workflows, expose conditioning assumptions and likelihood approximations; non-Gaussian/natural-language conditioning should be labeled experimental unless empirically validated.

## Reliability and Limitations

- Toto 2.0 claims are benchmark-specific and from a 2026 preprint; reproduce on Cramer-Short target data before production use.
- The GP conditioning paper is methodologically ambitious; practical reliability depends on likelihood quality, Monte Carlo approximation, and numerical implementation.
- Kalman filtering is optimal under linear-Gaussian assumptions; nonlinear or heavy-tailed financial regimes require extensions or robustness checks.
- `Kalman1960.pdf` has misleading embedded metadata (`Microsoft Word - Kalman15.doc`), so article metadata is inferred from extracted text.
- No external sources were used to fill missing metadata.

## Verification Checklist

- [x] `./2605.20119v1.pdf`
- [x] `./2605.21041v1.pdf`
- [x] `./Kalman1960.pdf`
