# LLM Forecasting: Research Distillation

One-line purpose: distill the listed LLM forecasting and financial-LLM papers into practical guidance for agentic forecasting, evidence synthesis, and financial analysis.

## Source Coverage

| PDF | Inferred title | Year/authors if available | Extraction status | Relevance |
|---|---|---|---|---|
| `./2409.13740v2.pdf` | Language Agents Achieve Superhuman Synthesis of Scientific Knowledge | 2024; Michael D. Skarlinski, Sam Cox, Jon M. Laurent, James D. Braza, Michaela Hinks, Michael J. Hammerling, Manvitha Ponnapati, Samuel G. Rodriques, Andrew D. White | Extracted via `pdfinfo` and `pdftotext`; 25 pages; PDF metadata title/author fields unreliable | Shows cited language-agent synthesis can match or exceed domain experts on literature-search tasks, relevant to evidence retrieval before forecasts. |
| `./2507.04562v3.pdf` | Evaluating LLMs on Real-World Forecasting Against Expert Forecasters | 2025; Janna Lu | Extracted via `pdfinfo` and `pdftotext`; 22 pages | Benchmarks frontier LLMs on 464 Metaculus questions and compares Brier scores against crowds and expert forecasters. |
| `./2603.25040v1.pdf` | Intern-S1-Pro: Scientific Multimodal Foundation Model at Trillion Scale | 2026; Intern-S1-Pro Team / many listed authors | Extracted via `pdfinfo` and `pdftotext`; 19 pages | Describes a trillion-parameter scientific multimodal foundation model with agent capabilities and specialized scientific-task coverage. |
| `./Financial-Statement-Analysis-with-Large-Language-Models.pdf` | Financial Statement Analysis with Large Language Models | 2024; Alex G. Kim, Maximilian Muhn, Valeri V. Nikolaev | Extracted via `pdfinfo` and `pdftotext`; 59 pages; PDF metadata title/author fields unreliable | Tests GPT-4 on anonymized financial statements for directional earnings-change prediction and trading-strategy construction. |
| `./Kong_et_al_2024_Large_language_models.pdf` | A Survey of Large Language Models for Financial Applications: Progress, Prospects and Challenges | 2024; Yuqi Nie, Yaxuan Kong, Xiaowen Dong, John M. Mulvey, H. Vincent Poor, Qingsong Wen, Stefan Zohren | Extracted via `pdfinfo` and `pdftotext`; 39 pages; PDF metadata title/author fields unreliable | Surveys financial LLM applications across linguistic tasks, sentiment, time series, reasoning, agent-based modeling, and decision support. |

## Practical Synthesis for Cramer-Short

- Use LLMs as forecast assistants, not automatic truth machines: they can synthesize evidence and generate calibrated forecasts, but expert forecaster crowds remain a strong benchmark.
- Require citations for research synthesis. The PaperQA2 paper frames factuality around cited, source-grounded answers and contradiction detection.
- For event forecasting, score with proper scoring rules. The Metaculus evaluation uses Brier scores and finds frontier models can surpass a human crowd yet still trail expert forecasters.
- For financial statement analysis, structured anonymized statements can be enough for useful LLM reasoning; the Kim/Muhn/Nikolaev paper reports GPT-4 outperforming analysts directionally and producing strategy alphas.
- Financial LLM workflows should be task-specific: text analysis, sentiment, time-series augmentation, reasoning, simulation, and agent-based trading each need different validation.

## Implementation Implications

- Add a forecasting mode that records probability, rationale, cited evidence, retrieval timestamp, model/provider, and Brier-ready resolution metadata.
- Use retrieval-augmented prompts for forecasts; include recent news or filings only up to the declared cutoff to avoid leakage.
- Prefer direct probabilistic prompts over elaborate narrative roleplay unless validated; the Metaculus paper reports narrative-prediction tables separately and does not establish roleplay as universally better.
- For financial-statement workflows, anonymize ticker/company cues during evaluation to test reasoning rather than memory.
- Add expert/crowd baselines where available; a model beating naive or crowd baselines is not enough if expert forecasters remain materially better.
- Treat very large scientific foundation models as capabilities signals, not drop-in financial forecasters, unless tested on finance-specific tasks.

## Reliability and Limitations

- `2603.25040v1.pdf` is about scientific multimodal modeling broadly; its direct relevance to financial forecasting is indirect.
- `2409.13740v2.pdf` evaluates scientific literature tasks, not market forecasts; use it for evidence-synthesis design, not alpha claims.
- The Metaculus benchmark is event-forecasting, not time-series asset forecasting; Brier-score lessons transfer better than domain performance.
- Financial-statement LLM results depend on prompts, data anonymization, and task definition; production use still needs leakage checks and independent validation.
- Several PDFs have unreliable embedded metadata, so authors/titles are inferred from extracted title pages where needed.

## Verification Checklist

- [x] `./2409.13740v2.pdf`
- [x] `./2507.04562v3.pdf`
- [x] `./2603.25040v1.pdf`
- [x] `./Financial-Statement-Analysis-with-Large-Language-Models.pdf`
- [x] `./Kong_et_al_2024_Large_language_models.pdf`
