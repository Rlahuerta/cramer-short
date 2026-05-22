// Compatibility barrel for src/agent/query-router.
// Keep this limited to the historical public surface of src/agent/query-router.ts.

// Classification
export {
  isForecastLabImprovementQuery,
  isForecastLabPlanOnlyQuery,
  isDistributionQuery,
  isExplicitTerminalDistributionQuery,
  inferTrajectoryRequest,
  isCryptoForecastQuery,
  isNonCryptoForecastQuery,
  isExplicitPolymarketForecastRequest,
  isExplicitCombinedMarkovPolymarketRequest,
  isExplicitGoldCombinedMarkovPolymarketRequest,
  detectExplicitSkillRequest,
} from './classification.js';

// Distribution
export {
  getBtcSelectiveMarkovConfidenceThreshold,
  inferDistributionTicker,
  inferDistributionHorizon,
  inferBtcShortHorizonForecastHorizon,
  inferMarkovQueryHorizon,
  isBtcShortHorizonForecastQuery,
  buildForcedMarkovArgs,
  shouldForceMarkovDistribution,
  shouldInjectBtcShortHorizonMixedEvidencePrompt,
  shouldInjectBtcShortHorizonLowConfidencePrompt,
} from './distribution.js';

// Forced tool args
export {
  buildForcedMarketDataArgs,
  buildForcedNonCryptoMarketDataArgs,
  buildForcedSocialSentimentArgs,
  buildForcedPolymarketForecastArgs,
  buildForcedNonCryptoPolymarketForecastArgs,
  buildForcedOnchainArgs,
  buildForcedFixedIncomeArgs,
  buildForcedCryptoForecastMarkovArgs,
} from './forced-tool-args.js';

// Tool call extractors
export {
  hasMarketDataQuery,
  extractCurrentPriceFromMarketDataQuery,
  extractCurrentPriceFromToolCalls,
  extractSentimentScoreFromToolCalls,
  extractMarkovReturnFromToolCalls,
  extractMarkovPredictionConfidenceForQuery,
  hasLowConfidenceBtcShortHorizonMarkov,
} from './tool-call-extractors.js';

// Coverage
export {
  hasSuccessfulMarkovDistributionForQuery,
  hasAbstainingMarkovDistributionForQuery,
  hasCompletedMarkovDistributionForQuery,
  hasUsableOnchainResultForCryptoQuery,
  hasUsableFixedIncomeResult,
  hasPolymarketForecastCoverage,
  hasCryptoPolymarketForecastCoverage,
  shouldRerunPolymarketForecastWithMarkov,
} from './coverage.js';

// Forecast arbitrator
export {
  shouldForceNonCryptoForecastFallback,
  shouldForceCryptoForecastTools,
  shouldForceGoldCombinedForecastTools,
  hasForecastArbitratorForQuery,
  detectBtcShortHorizonDisagreement,
  buildForcedForecastArbiterArgs,
  shouldForceForecastArbitrator,
  buildForcedGoldCombinedForecastArbiterArgs,
  shouldForceGoldCombinedForecastArbitrator,
} from './forecast-arbitrator.js';
