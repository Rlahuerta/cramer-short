# Sequence Models: Research Distillation

One-line purpose: distill the listed sequence-models reference for Cramer-Short while explicitly noting that the provided PDF is about stock-selection strategies, not neural sequence models.

## Source Coverage

| PDF | Inferred title | Year/authors if available | Extraction status | Relevance |
|---|---|---|---|---|
| `./chung2015.pdf` | The selection of popular trading strategies | 2015; Yi-Tsai Chung, Tung Liang Liao, Yi-Chein Chiang | Extracted via `pdfinfo` and `pdftotext`; 21 pages | Despite the folder name, this paper compares popular stock-selection strategies using stochastic dominance rather than sequence models. |

## Practical Synthesis for Cramer-Short

- Treat this source as trading-factor evidence, not sequence-model evidence.
- The paper compares Size, book-to-market, earnings-to-price, cash-flow-to-price, and dividend-to-price strategies, including zero-investment premiums.
- Its main extracted finding is that highest E/P and CF/P strategies and corresponding premiums generally produce higher returns than the other examined strategies for US stock markets.
- The methodology uses stochastic dominance, a non-parametric approach suitable for comparing return distributions without relying only on mean/variance.
- Equal-weighted portfolio simulations are reported as more profitable than corresponding value-weighted simulations in the extracted practical-implications text.

## Implementation Implications

- Do not use this PDF to justify GRU/LSTM/Transformer sequence-model choices; no such sequence-model content was present in the extracted text.
- If incorporated into Cramer-Short, route it to factor-screening or trading-strategy validation notes rather than model-architecture notes.
- For factor screens, evaluate return distributions and downside behavior, not only average return.
- Keep metadata tags clear when references are misfiled so retrieval does not inject irrelevant evidence into sequence-model prompts.

## Reliability and Limitations

- The source is limited to US stock markets and explicitly notes that consistency with non-US markets needs further investigation.
- Findings are based on historical factor portfolios and stochastic-dominance comparisons; they do not validate modern sequence models.
- The PDF's embedded metadata is partially misleading but includes the article title and journal information; authors and abstract were inferred from extracted text.
- Because the single listed PDF is off-topic for sequence models, this distillation intentionally avoids making sequence-model architecture claims.

## Verification Checklist

- [x] `./chung2015.pdf`
