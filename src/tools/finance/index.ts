/**
 * Finance tool error contract:
 * - Provider, network, auth, and schema failures may throw inside lower-level
 *   clients and should be caught at DynamicStructuredTool boundaries where the
 *   public result becomes a formatted error string.
 * - Missing/unsupported market data should return null, an empty collection, or
 *   a formatted "no data" result instead of throwing.
 * - Public registry tool names remain stable snake_case; keep internal renames
 *   behind this barrel/registry boundary.
 */
export const FINANCE_TOOL_ERROR_CONTRACT = 'throw-provider-failures; null-or-empty-missing-data' as const;

export { getIncomeStatements, getBalanceSheets, getCashFlowStatements, getAllFinancialStatements } from './fundamentals.js';
export { getFilings, get10KFilingItems, get10QFilingItems, get8KFilingItems } from './filings.js';
export { getKeyRatios, getHistoricalKeyRatios } from './key-ratios.js';
export { getAnalystEstimates } from './estimates.js';
export { getSegmentedRevenues } from './segments.js';
export { fetchBinanceDailyCloses, fetchBinanceTicker24h, toBinanceSymbol } from './binance.js';
export {
  BITMEX_MARKET_DESCRIPTION,
  bitmexMarketTool,
  fetchBitmexDailyCloses,
  resolveBitmexHistoricalSymbol,
  toBitmexSymbolCandidates,
} from './bitmex.js';
export { getStockPrice, getStockPrices, getStockTickers, STOCK_PRICE_DESCRIPTION } from './stock-price.js';
export { getCryptoPriceSnapshot, getCryptoPrices, getCryptoTickers } from './crypto.js';
export { getInsiderTrades } from './insider_trades.js';
export { getEarnings } from './earnings.js';
export { getYahooAnalystTargets, getYahooAnalystRecommendations, getYahooUpgradeDowngradeHistory, getYahooIncomeStatements } from './yahoo-finance.js';
export { getFmpIncomeStatements, getFmpBalanceSheets, getFmpCashFlowStatements } from './fmp.js';
export { createGetFinancials } from './get-financials.js';
export { createGetMarketData } from './get-market-data.js';
export { createReadFilings } from './read-filings.js';
export { createScreenStocks } from './screen-stocks.js';
export { polymarketTool, POLYMARKET_DESCRIPTION } from './polymarket.js';
export { socialSentimentTool, SOCIAL_SENTIMENT_DESCRIPTION } from './social-sentiment.js';
export { getEarningsTranscript, EARNINGS_TRANSCRIPT_DESCRIPTION } from './earnings-transcripts.js';
export { createGetOnchainCryptoTool, getOnchainCrypto, ONCHAIN_CRYPTO_DESCRIPTION } from './onchain-crypto.js';
export { polymarketForecastTool, POLYMARKET_FORECAST_DESCRIPTION } from './polymarket-forecast.js';
export { createForecastArbitratorTool, forecastArbitratorTool, FORECAST_ARBITRATOR_DESCRIPTION, arbitrateForecast, classifyPolymarketQuestion } from './forecast-arbitrator.js';
export { readArbiterReplayBundles, appendArbiterReplayBundle, toForecastArbiterInput } from './arbiter-replay.js';
export { evaluateReplayLabelEligibility, labelReplayBundle, labelReplaySemanticMarket } from './arbiter-replay-labeler.js';
export { priceDistributionChartTool, PRICE_DISTRIBUTION_CHART_DESCRIPTION, buildPriceDistributionChart, extractPriceThresholds } from './price-distribution-chart.js';
export { markovDistributionTool, computeMarkovDistribution, MARKOV_DISTRIBUTION_DESCRIPTION } from './markov-distribution.js';
export { runArbiterReplay, compareReplayEvaluators } from './backtest/arbiter-replay-runner.js';
export {
  formatCombinedShortHorizonBenchmarkReport,
  runCombinedShortHorizonBenchmark,
  runCombinedShortHorizonBenchmarkFromFile,
} from './backtest/combined-short-horizon-benchmark.js';
export {
  DEFAULT_ARBITER_REPLAY_LABELED_REPORT_PATH,
  runReplayLabelBatch,
  runReplayLabelBatchFromFile,
  toReplayLabelBatchReportPath,
} from './backtest/replay-label-batch-runner.js';
export {
  DEFAULT_ARBITER_REPLAY_LABELED_BENCHMARK_REPORT_PATH,
  runReplayBenchmarkHandoffFromLabeledFile,
  runReplayLabelBenchmarkPipelineFromFile,
  toReplayLabelBenchmarkHorizonCounts,
  toReplayLabelBenchmarkReportPath,
} from './backtest/replay-label-benchmark-pipeline.js';
export {
  formatShortHorizonReplayBenchmarkReport,
  runShortHorizonReplayBenchmark,
  runShortHorizonReplayBenchmarkFromFile,
} from './backtest/polymarket-short-horizon-benchmark.js';
export { trumpPressureIndexTool, TRUMP_PRESSURE_DESCRIPTION, computeTrumpPressureIndex, checkTacoAlert } from './trump-pressure-index.js';
