import { describe } from 'bun:test';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { smokeTestSkill } from '../test-helpers/skill-smoke.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

describe('probability-assessment skill', () => {
  smokeTestSkill(join(__dirname, 'SKILL.md'), [
    'get_onchain_crypto',
    'get_fixed_income',
    'markov_distribution',
    'trajectory',
  ]);
});