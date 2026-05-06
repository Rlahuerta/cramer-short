import { describe, expect, it } from 'bun:test';

import { getForecastLabRoutingHint } from './forecast-lab-routing.js';

describe('forecast-lab routing hint', () => {
  it('builds a deterministic skill hint for forecast improvement queries', () => {
    const hint = getForecastLabRoutingHint(
      'Optimize the BTC 1d/2d/3d Markov forecast logic without broad self-editing.',
    );

    expect(hint).not.toBeNull();
    expect(hint?.recommendedProfileId).toBe('btc-markov-ultra-short-horizon');
    expect(hint?.mutationAllowed).toBe(true);
    expect(hint?.shouldInvokeSkill).toBe(true);
    expect(hint?.whyMatched).toContain('Matched improvement cues');
  });

  it('builds a guarded GOLD skill hint without enabling mutation yet', () => {
    const hint = getForecastLabRoutingHint(
      'Optimize the GLD 1d/2d/3d Markov forecast lane and review 7d/14d as GOLD guardrails.',
    );

    expect(hint).not.toBeNull();
    expect(hint?.recommendedProfileId).toBe('gold-markov-short-horizon');
    expect(hint?.mutationAllowed).toBe(false);
    expect(hint?.shouldInvokeSkill).toBe(true);
    expect(hint?.whyMatched).toContain('gold-markov-short-horizon');
  });

  it('does not build a skill hint for ordinary BTC forecast requests', () => {
    expect(
      getForecastLabRoutingHint('Give me a BTC forecast for the next 7 days and explain the drivers.'),
    ).toBeNull();
  });

  it('does not derive a hint when forecast-lab auto-routing is disabled', () => {
    expect(
      getForecastLabRoutingHint(
        'Optimize the BTC 1d/2d/3d Markov forecast logic without broad self-editing.',
        { enableAutoRoute: false },
      ),
    ).toBeNull();
  });

  it('does not inject a skill hint when forecast-lab skill hints are disabled', () => {
    expect(
      getForecastLabRoutingHint(
        'Optimize the BTC 1d/2d/3d Markov forecast logic without broad self-editing.',
        { enableSkillHint: false },
      ),
    ).toBeNull();
  });

  it('disables mutation when improvement intent lacks a deterministic profile match', () => {
    const hint = getForecastLabRoutingHint(
      'Optimize the general forecasting workflow with better orchestration and review loops.',
    );

    expect(hint).not.toBeNull();
    expect(hint?.recommendedProfileId).toBeNull();
    expect(hint?.mutationAllowed).toBe(false);
    expect(hint?.shouldInvokeSkill).toBe(true);
  });

  it('does not build a GOLD skill hint for portfolio-risk wording without Markov intent', () => {
    const hint = getForecastLabRoutingHint('Optimize my gold 14d hedge sizing for portfolio risk.');

    expect(hint).not.toBeNull();
    expect(hint?.recommendedProfileId).toBeNull();
    expect(hint?.mutationAllowed).toBe(false);
  });

  it('does not build a GOLD skill hint for dashboard wording without Markov intent', () => {
    const hint = getForecastLabRoutingHint('Fix gold 3d alert wording in the dashboard.');

    expect(hint).not.toBeNull();
    expect(hint?.recommendedProfileId).toBeNull();
    expect(hint?.mutationAllowed).toBe(false);
  });
});
