import { runComparison } from './src/tools/finance/backtest/phase5-hybrid-break-fallback.js';

async function main() {
  const artifact = await runComparison();
  const bestHybrid = artifact.candidates.find(c => c.candidateId === 'HYB_L025_M050_H075_lambda025');
  console.log('bestHybrid defined:', !!bestHybrid);
  if (bestHybrid) {
    console.log('breakTrendingDirectionalAccuracy:', bestHybrid.deltaVsBaseline.breakTrendingDirectionalAccuracy);
    console.log('overallBrier:', bestHybrid.deltaVsBaseline.overallBrier);
    console.log('passesThresholds:', bestHybrid.passesThresholds);
    console.log('failureReasons:', JSON.stringify(bestHybrid.failureReasons));
  }
  const control = artifact.candidates.find(c => c.candidateId === 'C60');
  console.log('control defined:', !!control);
  if (control) {
    console.log('control breakTrendingDirectionalAccuracy:', control.deltaVsBaseline.breakTrendingDirectionalAccuracy);
    console.log('control passesThresholds:', control.passesThresholds);
  }
  console.log('winner candidateId:', artifact.winner.candidateId);
  console.log('winner reason:', artifact.winner.reason);
  console.log('candidates count:', artifact.candidates.length);
}

main();
