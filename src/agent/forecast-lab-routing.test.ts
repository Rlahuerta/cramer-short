import { describe, expect, it } from 'bun:test';

import {
  extractForecastLabMutatorId,
  getForecastLabRoutingHint,
  routeForecastLabIntent,
} from '../experiments/forecast-lab/query-router.js';

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

  it('builds a GOLD skill hint with structured mutation enabled once the shipped catalog exists', () => {
    const hint = getForecastLabRoutingHint(
      'Optimize the GLD 1d/2d/3d Markov forecast lane and review 7d/14d as GOLD guardrails.',
    );

    expect(hint).not.toBeNull();
    expect(hint?.recommendedProfileId).toBe('gold-markov-short-horizon');
    expect(hint?.mutationAllowed).toBe(true);
    expect(hint?.shouldInvokeSkill).toBe(true);
    expect(hint?.whyMatched).toContain('gold-markov-short-horizon');
  });

  it('builds a SOL skill hint with structured mutation enabled once the shipped catalog exists', () => {
    const hint = getForecastLabRoutingHint(
      'Optimize the SOL 1d/2d/3d Markov forecast lane and review 7d/14d as SOL guardrails.',
    );

    expect(hint).not.toBeNull();
    expect(hint?.recommendedProfileId).toBe('sol-markov-short-horizon');
    expect(hint?.mutationAllowed).toBe(true);
    expect(hint?.shouldInvokeSkill).toBe(true);
    expect(hint?.whyMatched).toContain('sol-markov-short-horizon');
  });

  it('builds a HYPE skill hint with structured mutation enabled once the shipped catalog exists', () => {
    const hint = getForecastLabRoutingHint(
      'Tune the HYPE 1d/2d/3d Markov forecast lane and review 7d/14d as HYPE guardrails.',
    );

    expect(hint).not.toBeNull();
    expect(hint?.recommendedProfileId).toBe('hype-markov-short-horizon');
    expect(hint?.mutationAllowed).toBe(true);
    expect(hint?.shouldInvokeSkill).toBe(true);
    expect(hint?.whyMatched).toContain('hype-markov-short-horizon');
  });

  it('does not build a skill hint for ordinary BTC forecast requests', () => {
    expect(
      getForecastLabRoutingHint('Give me a BTC forecast for the next 7 days and explain the drivers.'),
    ).toBeNull();
  });

  it('routes all forecast-lab intent hints through the compact facade', () => {
    const route = routeForecastLabIntent(
      'Optimize the BTC 1d/2d/3d Markov forecast logic without broad self-editing.',
    );

    expect(route.routingHint?.recommendedProfileId).toBe('btc-markov-ultra-short-horizon');
    expect(route.routingHint?.shouldInvokeSkill).toBe(true);
    expect(route.resetRequest).toBeNull();
    expect(route.promotionApproval).toBeNull();
    expect(route.catalogExtensionRequest).toBeNull();
  });

  it('does not build a skill hint for live BitMEX trade-brief prompts that avoid improvement wording', () => {
    expect(
      getForecastLabRoutingHint(
        [
          'Live BitMEX trade brief for SOLUSD and HYPEUSDT.',
          'This is a live market-analysis request, not an experiment request.',
          'Do not use any skill and do not use forecast_lab_run.',
          'Start with bitmex_market, then use markov_distribution for 1d, 2d, and 3d,',
          'then use forecast_arbitrator if one market still has a usable edge.',
          'Return one final decision: LONG SOLUSD, SHORT SOLUSD, LONG HYPEUSDT, SHORT HYPEUSDT, or NO TRADE.',
        ].join(' '),
      ),
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

  it('does not build a SOL skill hint for dashboard wording without Markov intent', () => {
    const hint = getForecastLabRoutingHint('Fix SOL 3d alert wording in the dashboard.');

    expect(hint).not.toBeNull();
    expect(hint?.recommendedProfileId).toBeNull();
    expect(hint?.mutationAllowed).toBe(false);
  });

  it('does not build a HYPE skill hint for portfolio-risk wording without Markov intent', () => {
    const hint = getForecastLabRoutingHint('Optimize my HYPE 14d position sizing for portfolio risk.');

    expect(hint).not.toBeNull();
    expect(hint?.recommendedProfileId).toBeNull();
    expect(hint?.mutationAllowed).toBe(false);
  });

  it('extracts both shipped and GOLD-prefixed mutator ids from explicit mutator requests', () => {
    expect(extractForecastLabMutatorId('Use mutator markov-shorter-reactive-window.')).toBe(
      'markov-shorter-reactive-window',
    );
    expect(extractForecastLabMutatorId('Use mutator gold-markov-shorter-reactive-window.')).toBe(
      'gold-markov-shorter-reactive-window',
    );
  });
});
