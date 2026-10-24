# Sentiment Analysis References Distillation

Purpose: topic-level synthesis of the local PDFs in this folder for Cramer-Short sentiment features and financial research tooling.

## Source Coverage

| Source PDF | Inferred title | Year/authors if available | Extraction status | One-sentence relevance |
|---|---|---|---|---|
| `./a-review-on-feature-selection-techniques-for-sentiment-virrrmgn.pdf` | A Review on Feature Selection Techniques for Sentiment Analysis | 2022; Kriti Agarwal | Extracted title page, abstract, result table, and conclusion with `pdftotext`; PDF metadata author is `Admin`. | Reviews and tests feature-selection techniques for stock-market Twitter sentiment, showing simple word-frequency filtering can outperform more aggressive reductions for a Naive Bayes classifier. |
| `./jrfm-18-00412.pdf` | News Sentiment and Stock Market Dynamics: A Machine Learning Investigation | 2025; Milivoje Davidovic, Jacqueline McCleary | Extracted metadata, title page, abstract, model findings, conclusions, and limitations with `pdftotext`; some extracted lines contain encoding artifacts but core prose is readable. | Large financial-news study finding that TextBlob/VADER/FinBERT sentiment alone has weak standalone predictive power, while VIX, volume, controls, and weekend/holiday adjustments add more useful signal. |
| `./s42521-025-00162-3.pdf` | Financial sentiment analysis with FUNNEL: filtered UNion for NER-based ensemble labeling | 2025; William Nordansjö, Fredrik Fourong, Muhammad Qasim | Extracted title page, abstract, manual-evaluation table, discussion, conclusion, and limitations with `pdftotext`. | Provides an entity-aware ensemble-labeling framework that improves stock-specific financial-news labels by combining keyword heuristics with spaCy NER and weighted voting. |

## Practical Synthesis for Cramer-Short

- Do not treat sentiment scores as direct alpha. The JRFM study repeatedly finds limited standalone predictive power and stronger contributions from implied volatility, volume, and controls.
- Preserve simple lexical baselines. The 2022 feature-selection review found word-frequency filtering with minimum frequency 3 reached the highest reported accuracy, 74.00%, in its Naive Bayes experiment.
- Make entity linking a first-class step. FUNNEL shows that article-to-stock labeling quality can dominate downstream sentiment usefulness.
- Compare multiple sentiment models because they have different polarity biases: the FUNNEL paper reports systematic differences among FinBERT, RoBERTa, and VADER.
- Time handling matters: weekend and holiday sentiment may carry modest signal, but can also introduce noise depending on task and model.
- Use sentiment most safely for risk context, hedging/tactical rebalancing cues, and explanation, not unqualified short-term return prediction.

## Implementation Implications

- Sentiment pipelines should include: deduplication, timestamp normalization, entity/stock linking, source quality checks, model scoring, feature selection, and leakage-safe joining to market data.
- Add baseline sentiment features before complex models: term frequency/word-frequency filters, TF-IDF, document frequency, chi-square scores, VADER/TextBlob, and neutral/objective-rate diagnostics.
- For stock-specific news, implement ensemble labeling similar to FUNNEL: high-recall keyword/fuzzy matchers plus high-precision NER validation and a tunable weighted-vote threshold.
- Track model-specific distributions and bias warnings; FinBERT, RoBERTa, VADER, TextBlob, and composite scores should not be assumed interchangeable.
- Evaluate by horizon and task: classification F1/balanced accuracy, regression error, calibration, feature importance, ablations, and transaction-cost-aware strategy tests.
- Add guards against overclaiming: if sentiment-only features are weak or redundant after VIX/volume/macroeconomic controls, final answers should say so explicitly.

## Reliability and Limitations

- The 2022 review is useful for practical feature-selection baselines, but its experiment uses a specific Twitter stock-market dataset, Naive Bayes, and R implementation; results may not transfer to financial-news corpora or transformer pipelines.
- The JRFM paper is broad and directly finance-focused, but daily returns may miss intraday event effects, pretrained models may misclassify financial tone, and the study reports no reliably exploitable strategy after costs.
- The FUNNEL paper gives strong labeling-design guidance, but its empirical focus is MAG7 companies, uses pretrained sentiment models without additional fine-tuning, and reports non-trivial CPU/GPU costs.
- Across all sources, sentiment usefulness is dataset-, timestamp-, entity-linking-, horizon-, and cost-assumption-dependent; Cramer-Short should require local validation before surfacing trading implications.

## Verification Checklist

- [x] `./a-review-on-feature-selection-techniques-for-sentiment-virrrmgn.pdf`
- [x] `./jrfm-18-00412.pdf`
- [x] `./s42521-025-00162-3.pdf`
