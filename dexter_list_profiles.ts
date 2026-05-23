import { listForecastLabProfiles } from './src/experiments/forecast-lab/profiles.js';

const profiles = listForecastLabProfiles();
const summary = profiles.map(p => ({
  id: p.id,
  mutationMode: p.mutation.mode,
  ...(p.mutation.mode === 'structured' && { allowedMutatorIds: p.mutation.allowedMutatorIds })
}));

console.log(JSON.stringify(summary, null, 2));
