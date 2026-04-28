import { runComparison } from './src/tools/finance/backtest/phase4-trend-penalty-comparison.js';

async function main() {
  const artifact = await runComparison();
  console.log('--- BASELINE ---');
  console.log('totalSteps:', artifact.baseline.totalSteps);
  console.log('breakSteps:', artifact.baseline.breakSteps);
  console.log('overallRC 0.2 accuracy:', artifact.baseline.overallRC.find(p => p.threshold === 0.2)?.accuracy);
  console.log('overallRC 0.2 coverage:', artifact.baseline.overallRC.find(p => p.threshold === 0.2)?.coverage);
  console.log('breakContextRC 0.3 accuracy:', artifact.baseline.breakContextRC.find(p => p.threshold === 0.3)?.accuracy);
  console.log('breakContextRC 0.3 coverage:', artifact.baseline.breakContextRC.find(p => p.threshold === 0.3)?.coverage);
  console.log('--- EXPERIMENT ---');
  console.log('overallRC 0.2 accuracy:', artifact.experiment.overallRC.find(p => p.threshold === 0.2)?.accuracy);
  console.log('overallRC 0.2 coverage:', artifact.experiment.overallRC.find(p => p.threshold === 0.2)?.coverage);
  console.log('breakContextRC 0.3 accuracy:', artifact.experiment.breakContextRC.find(p => p.threshold === 0.3)?.accuracy);
  console.log('breakContextRC 0.3 coverage:', artifact.experiment.breakContextRC.find(p => p.threshold === 0.3)?.coverage);
  console.log('--- DELTA ---');
  console.log('changedStepCount:', artifact.delta.changedStepCount);
  console.log('changedBreakChopCount:', artifact.delta.changedBreakChopCount);
  console.log('changedBreakTrendingCount:', artifact.delta.changedBreakTrendingCount);
}

main();
