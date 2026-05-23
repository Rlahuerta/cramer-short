export { stripThinkingTags } from './strip-thinking.js';
export { buildSourcesFooter } from './sources-footer.js';
export { ensureStructuredDensityTable } from './density-table.js';
export {
  buildAbstainingMarkovAnswer,
  buildUnavailableDistributionAnswer,
} from './markov-answers.js';
export {
  buildAbstainingBtcShortHorizonForecastAnswer,
  shouldPreserveAbstainingBtcShortHorizonForecast,
} from './btc-forecast.js';
export { buildExplicitGoldCombinedForecastAnswer } from './gold-combined-forecast.js';
export {
  buildDistributionWarningPrefix,
  buildForecastDisagreementPrefix,
  buildLowConfidenceBtcShortHorizonForecastPrefix,
} from './warning-prefixes.js';
