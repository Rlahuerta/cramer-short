import { describe, it, expect } from 'bun:test';

const {
  getForecastLabMarkovRuntimeDefaults,
  setForecastLabMarkovRuntimeDefaults,
} = await import('../tools/finance/markov-distribution.js');
const { getCurrentDate } = await import('../utils/date.js');
const {
  buildSystemPrompt,
  buildIterationPrompt,
  buildGroupSection,
  injectForecastLabRoutingHint,
  loadSoulDocument,
} = await import('./prompts.js');

const MOCK_TOOL_DESCRIPTIONS = 'mock tool descriptions';

describe('getCurrentDate', () => {
  it('returns a non-empty string', () => {
    const date = getCurrentDate();
    expect(typeof date).toBe('string');
    expect(date.length).toBeGreaterThan(0);
  });

  it('includes the current year', () => {
    const date = getCurrentDate();
    const currentYear = new Date().getFullYear().toString();
    expect(date).toContain(currentYear);
  });

  it('contains the day of the week', () => {
    const date = getCurrentDate();
    const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
    expect(days.some(d => date.includes(d))).toBe(true);
  });
});

describe('loadSoulDocument', () => {
  it('returns a value (string or null)', async () => {
    const result = await loadSoulDocument();
    // The bundled SOUL.md exists in the repo, so it should be a string
    expect(typeof result === 'string' || result === null).toBe(true);
  });

  it('returns a string since the bundled SOUL.md exists', async () => {
    const result = await loadSoulDocument();
    // The SOUL.md in the project root acts as the bundled fallback
    expect(result).not.toBeNull();
    expect(typeof result).toBe('string');
  });
});

describe('buildSystemPrompt', () => {
  it('contains tool descriptions', () => {
    const prompt = buildSystemPrompt('gpt-5.4', undefined, undefined, undefined, undefined, undefined, MOCK_TOOL_DESCRIPTIONS);
    expect(prompt).toContain('mock tool descriptions');
  });

  it('contains the current date', () => {
    const prompt = buildSystemPrompt('gpt-5.4', undefined, undefined, undefined, undefined, undefined, MOCK_TOOL_DESCRIPTIONS);
    const currentYear = new Date().getFullYear().toString();
    expect(prompt).toContain(currentYear);
  });

  it('includes soul content when provided', () => {
    const soulContent = 'I am a focused financial analyst.';
    const prompt = buildSystemPrompt('gpt-5.4', soulContent, undefined, undefined, undefined, undefined, MOCK_TOOL_DESCRIPTIONS);
    expect(prompt).toContain(soulContent);
  });

  it('does not include Identity section when soul is null', () => {
    const prompt = buildSystemPrompt('gpt-5.4', null, undefined, undefined, undefined, undefined, MOCK_TOOL_DESCRIPTIONS);
    expect(prompt).not.toContain('## Identity');
  });

  it('uses WhatsApp profile preamble when channel=whatsapp', () => {
    const prompt = buildSystemPrompt('gpt-5.4', null, 'whatsapp', undefined, undefined, undefined, MOCK_TOOL_DESCRIPTIONS);
    expect(prompt.toLowerCase()).toContain('whatsapp');
  });

  it('uses CLI profile when channel=cli', () => {
    const prompt = buildSystemPrompt('gpt-5.4', null, 'cli', undefined, undefined, undefined, MOCK_TOOL_DESCRIPTIONS);
    expect(prompt).toContain('CLI');
  });

  it('omits tables section for whatsapp channel', () => {
    const prompt = buildSystemPrompt('gpt-5.4', null, 'whatsapp', undefined, undefined, undefined, MOCK_TOOL_DESCRIPTIONS);
    expect(prompt).not.toContain('## Tables');
  });

  it('includes tables section for CLI channel', () => {
    const prompt = buildSystemPrompt('gpt-5.4', null, 'cli', undefined, undefined, undefined, MOCK_TOOL_DESCRIPTIONS);
    expect(prompt).toContain('Tables');
  });

  it('includes memory section', () => {
    const prompt = buildSystemPrompt('gpt-5.4');
    expect(prompt).toContain('Memory');
  });

  it('includes memory files list when provided', () => {
    const prompt = buildSystemPrompt('gpt-5.4', null, 'cli', undefined, ['goals.md', 'daily.md']);
    expect(prompt).toContain('goals.md');
    expect(prompt).toContain('daily.md');
  });

  it('includes memory context when provided', () => {
    const prompt = buildSystemPrompt(
      'gpt-5.4',
      null,
      'cli',
      undefined,
      [],
      'User is a long-term investor focused on dividends.',
    );
    expect(prompt).toContain('long-term investor');
  });

  it('includes group section when groupContext is provided', () => {
    const groupCtx = { groupName: 'Investors Club', activationMode: 'mention' as const };
    const prompt = buildSystemPrompt('gpt-5.4', null, 'whatsapp', groupCtx);
    expect(prompt).toContain('Investors Club');
    expect(prompt).toContain('Group Chat');
  });

  it('mentions Cramer-Short as the assistant name', () => {
    const prompt = buildSystemPrompt('gpt-5.4');
    expect(prompt).toContain('Cramer-Short');
  });
});

describe('buildIterationPrompt', () => {
  it('includes the original query', () => {
    const prompt = buildIterationPrompt('What is AAPL price?', '');
    expect(prompt).toContain('What is AAPL price?');
  });

  it('includes tool results when non-empty', () => {
    const results = '### web_search(query=AAPL)\n{"price":180}';
    const prompt = buildIterationPrompt('AAPL price?', results);
    expect(prompt).toContain('web_search');
    expect(prompt).toContain('180');
  });

  it('omits tool results section when empty', () => {
    const prompt = buildIterationPrompt('What is 2+2?', '');
    expect(prompt).not.toContain('Data retrieved');
  });

  it('omits tool results section when whitespace-only', () => {
    const prompt = buildIterationPrompt('test', '   ');
    expect(prompt).not.toContain('Data retrieved');
  });

  it('includes toolUsageStatus when provided', () => {
    const status = '## Tool Usage This Query\n\n- web_search: 2/3 calls';
    const prompt = buildIterationPrompt('test', '', status);
    expect(prompt).toContain('Tool Usage');
    expect(prompt).toContain('web_search: 2/3 calls');
  });

  it('does not include toolUsageStatus when not provided', () => {
    const prompt = buildIterationPrompt('test', '');
    expect(prompt).not.toContain('Tool Usage This Query');
  });

  it('always includes continuation instruction', () => {
    const prompt = buildIterationPrompt('test', '');
    expect(prompt).toContain('Continue working');
  });

  it('injects canonical markov guard when markov_distribution output is present', () => {
    const results = '### markov_distribution(ticker=BTC-USD)\n{"data":{"_tool":"markov_distribution","canonical":{"scenarios":{}}}}';
    const prompt = buildIterationPrompt('BTC query', results);
    expect(prompt).toContain('markov_distribution results are present');
    expect(prompt).toContain('Do NOT recompute');
  });

  it('injects abstention guard when markov_distribution abstains', () => {
    const results = '### markov_distribution(ticker=BTC-USD)\n{"data":{"_tool":"markov_distribution","status":"abstain","canonical":{"scenarios":null}}}';
    const prompt = buildIterationPrompt('BTC query', results);
    expect(prompt).toContain('markov_distribution explicitly abstained');
    expect(prompt).toContain('MUST NOT create, correct, extrapolate');
    expect(prompt).toContain('MAY provide fallback analysis such as a point forecast');
    expect(prompt).toContain('MUST clearly warn');
  });

  it('injects proxy and anchor-label guards for canonical markov output', () => {
    const results = [
      '### markov_distribution(ticker=BTC-USD)',
      '{"data":{"_tool":"markov_distribution","status":"abstain","canonical":{"scenarios":null,"diagnostics":{"trustedAnchors":6,"anchorBypassApplied":false,"calibrationMode":"anchored"}}}}',
      '### polymarket_forecast(ticker=BTC)',
      'Forecast return: +0.9%\nGrade: B',
    ].join('\n');
    const prompt = buildIterationPrompt(
      'Provide the Polymarket and Markov BTC forecast for 24 hours, also providing the density probabilities for the price range divided into 9 parts.',
      results,
    );
    expect(prompt).toContain('Do NOT substitute another asset, proxy ticker, or proxy-history narrative');
    expect(prompt).toContain('do NOT describe a BTC/BTC-USD run as using GLD, gold, or any commodity-equivalent history');
    expect(prompt).toContain('if diagnostics.anchorBypassApplied is false or diagnostics.calibrationMode is "anchored"');
    expect(prompt).toContain('do NOT call the run "model-only", "commodity bypass", or say trusted anchors were "unused"');
    expect(prompt).toContain('If trusted anchors are present, do NOT imply they were absent or ignored just because a displayed mixing split rounds to 100% Markov / 0% Anchors');
  });

  it('injects mixed-evidence guard when BTC short-horizon Markov and Polymarket disagree', () => {
    const results = [
      '### markov_distribution(ticker=BTC-USD)',
      '{"data":{"_tool":"markov_distribution","status":"ok","canonical":{"actionSignal":{"recommendation":"BUY","expectedReturn":0.032}}}}',
      '### polymarket_forecast(ticker=BTC-USD)',
      'Forecast return: -0.4%\nGrade: B',
    ].join('\n');
    const prompt = buildIterationPrompt('Provide a BTC forecast for the next 14 days', results);
    expect(prompt).toContain('BTC short-horizon signals are mixed');
    expect(prompt).toContain('downgrade the narrative confidence');
  });

  it('injects low-confidence selective-gate guard using the BTC runtime threshold', () => {
    const originalBtcRuntimeDefaults = getForecastLabMarkovRuntimeDefaults('btc');

    try {
      setForecastLabMarkovRuntimeDefaults('btc', {
        ...originalBtcRuntimeDefaults,
        recommendedConfidenceThreshold: 0.2,
      });

      const results = [
        '### markov_distribution(ticker=BTC-USD)',
        '{"data":{"_tool":"markov_distribution","status":"ok","canonical":{"actionSignal":{"recommendation":"BUY","expectedReturn":0.032},"diagnostics":{"predictionConfidence":0.18}}}}',
      ].join('\n');
      const prompt = buildIterationPrompt('Provide a BTC forecast for the next 14 days', results);
      expect(prompt).toContain('predictionConfidence is below the 0.20 selective threshold');
      expect(prompt).toContain('fallback context');
    } finally {
      setForecastLabMarkovRuntimeDefaults('btc', originalBtcRuntimeDefaults);
    }
  });

  it('does not inject mixed-evidence guard when BTC short-horizon markov confidence is below the BTC runtime threshold', () => {
    const originalBtcRuntimeDefaults = getForecastLabMarkovRuntimeDefaults('btc');

    try {
      setForecastLabMarkovRuntimeDefaults('btc', {
        ...originalBtcRuntimeDefaults,
        recommendedConfidenceThreshold: 0.2,
      });

      const results = [
        '### markov_distribution(ticker=BTC-USD)',
        '{"data":{"_tool":"markov_distribution","status":"ok","canonical":{"actionSignal":{"recommendation":"BUY","expectedReturn":0.032},"diagnostics":{"predictionConfidence":0.18}}}}',
        '### polymarket_forecast(ticker=BTC-USD)',
        'Forecast return: -0.4%\nGrade: B',
      ].join('\n');
      const prompt = buildIterationPrompt('Provide a BTC forecast for the next 14 days', results);
      expect(prompt).not.toContain('BTC short-horizon signals are mixed');
      expect(prompt).toContain('predictionConfidence is below the 0.20 selective threshold');
    } finally {
      setForecastLabMarkovRuntimeDefaults('btc', originalBtcRuntimeDefaults);
    }
  });

  it('injects trajectory-semantics guard when markov_distribution includes a trajectory payload', () => {
    const results = [
      '### markov_distribution(ticker=BTC-USD)',
      '{"data":{"_tool":"markov_distribution","status":"ok","canonical":{"actionSignal":{"recommendation":"BUY","confidence":"LOW"},"diagnostics":{"predictionConfidence":0.43,"regimeState":"bull","recommendationProvenance":"override note"}},"trajectory":[{"day":1,"expectedPrice":78000,"regime":"bear"}]}}',
    ].join('\n');
    const prompt = buildIterationPrompt('Provide a BTC forecast for the next 7 days', results);
    expect(prompt).toContain('includes both a terminal canonical forecast and a day-by-day trajectory');
    expect(prompt).toContain('Do NOT claim an "internal inconsistency" solely because');
    expect(prompt).toContain('latent HMM backdrop');
    expect(prompt).toContain('Do NOT relabel predictionConfidence with LOW/MEDIUM/HIGH');
    expect(prompt).toContain('recommendationProvenance');
  });

  it('injects regime-action mismatch guidance for latent bull with SELL action signal', () => {
    const results = [
      '### markov_distribution(ticker=BTC-USD)',
      '{"data":{"_tool":"markov_distribution","status":"ok","canonical":{"actionSignal":{"recommendation":"SELL","confidence":"LOW"},"diagnostics":{"predictionConfidence":0.44,"regimeState":"bull","recommendationProvenance":"converted from HOLD because P(up) is 49.3%"}}}}',
    ].join('\n');
    const prompt = buildIterationPrompt('Give me a BTC forecast for the next 24 hours', results);
    expect(prompt).toContain('contains a regime/action mismatch');
    expect(prompt).toContain('latent HMM backdrop');
    expect(prompt).toContain('weak tilt/lean');
  });

  it('injects arbiter precedence guidance when forecast_arbitrator returns NO_TRADE', () => {
    const results = [
      '### markov_distribution(ticker=BTC-USD)',
      '{"data":{"_tool":"markov_distribution","status":"ok","canonical":{"actionSignal":{"recommendation":"SELL","confidence":"LOW"},"diagnostics":{"predictionConfidence":0.44,"regimeState":"bull"}}}}',
      '### forecast_arbitrator(ticker=BTC-USD)',
      '{"data":{"result":{"verdict":"NO_TRADE","preferredDirection":"neutral","shouldEnterNow":false}}}',
    ].join('\n');
    const prompt = buildIterationPrompt('Give me a BTC forecast for the next 24 hours with trade setup', results);
    expect(prompt).toContain('forecast_arbitrator returned NO_TRADE');
    expect(prompt).toContain('final trading decision');
    expect(prompt).toContain('subordinate model evidence');
  });

  it('preserves explicit bucket-count requests when canonical Markov output is present', () => {
    const results = [
      '### markov_distribution(ticker=BTC-USD)',
      '{"data":{"_tool":"markov_distribution","status":"ok","canonical":{"scenarios":{"flat":0.8}}}}',
    ].join('\n');
    const prompt = buildIterationPrompt(
      'Provide the Polymarket and Markov BTC forecast for 24 hours, also providing the density probabilities for the price range divided into 9 parts.',
      results,
    );
    expect(prompt).toContain('divided into 9 parts');
    expect(prompt).toContain('Preserve that 9-part bucket granularity');
    expect(prompt).toContain('Do NOT compress it into fewer buckets');
    expect(prompt).toContain('density probabilities, that means per-bucket probability mass');
    expect(prompt).toContain('do NOT substitute it for the requested density table');
  });

  it('does not inject mixed-evidence guard for BTC horizons above 14 days', () => {
    const results = [
      '### markov_distribution(ticker=BTC-USD)',
      '{"data":{"_tool":"markov_distribution","status":"ok","canonical":{"actionSignal":{"recommendation":"BUY","expectedReturn":0.032}}}}',
      '### polymarket_forecast(ticker=BTC-USD)',
      'Forecast return: -0.4%\nGrade: B',
    ].join('\n');
    const prompt = buildIterationPrompt('Provide a BTC forecast for the next 30 days', results);
    expect(prompt).not.toContain('BTC short-horizon signals are mixed');
  });

  it('does not inject mixed-evidence guard for non-BTC assets', () => {
    const results = [
      '### markov_distribution(ticker=ETH-USD)',
      '{"data":{"_tool":"markov_distribution","status":"ok","canonical":{"actionSignal":{"recommendation":"BUY","expectedReturn":0.032}}}}',
      '### polymarket_forecast(ticker=ETH-USD)',
      'Forecast return: -0.4%\nGrade: B',
    ].join('\n');
    const prompt = buildIterationPrompt('Provide an ETH forecast for the next 14 days', results);
    expect(prompt).not.toContain('BTC short-horizon signals are mixed');
  });

  it('does not inject mixed-evidence guard for unrelated BUY text or unrelated bearish forecast text', () => {
    const results = [
      '### some_other_tool()',
      '{"recommendation":"BUY"}',
      '### unrelated_text()',
      'Forecast return: -0.4%\nGrade: B',
    ].join('\n');
    const prompt = buildIterationPrompt('Provide a BTC forecast for the next 7 days', results);
    expect(prompt).not.toContain('BTC short-horizon signals are mixed');
  });

  it('recognizes BTC-USD phrasing for mixed-evidence guard', () => {
    const results = [
      '### markov_distribution(ticker=BTC-USD)',
      '{"data":{"_tool":"markov_distribution","status":"ok","canonical":{"actionSignal":{"recommendation":"BUY","expectedReturn":0.032}}}}',
      '### polymarket_forecast(ticker=BTC-USD)',
      'Forecast return: -0.4%\nGrade: B',
    ].join('\n');
    const prompt = buildIterationPrompt('Provide a BTC-USD forecast for the next 7 days', results);
    expect(prompt).toContain('BTC short-horizon signals are mixed');
  });

  it('adds commodity proxy framing for gold fallback prompts after Markov abstains', () => {
    const results = '### markov_distribution(ticker=GLD)\n{"data":{"_tool":"markov_distribution","status":"abstain","canonical":{"scenarios":null}}}';
    const prompt = buildIterationPrompt('Provide a GOLD forecast based on markov chain for the next 30 days', results);
    expect(prompt).toContain('GLD is only the data proxy for Gold');
    expect(prompt).toContain('Frame the final answer in terms of the underlying commodity');
  });

  it('keeps combined GOLD fallback prompts in the combined Markov + Polymarket path across 1d/2d/3d/14d horizons', () => {
    const results = '### markov_distribution(ticker=GLD)\n{"data":{"_tool":"markov_distribution","status":"abstain","canonical":{"scenarios":null}}}';
    for (const days of [1, 2, 3, 14]) {
      const prompt = buildIterationPrompt(
        `Provide a GOLD price forecast based on markov chain and polymarket for the next ${days} day${days === 1 ? '' : 's'}`,
        results,
      );
      expect(prompt).toContain('markov_distribution explicitly abstained');
      expect(prompt).toContain('combined Markov + polymarket forecast');
      expect(prompt).toContain('Do not collapse the answer into a Markov-only diagnostics note or a Polymarket-only framing');
      expect(prompt).toContain('GLD is only the data proxy for Gold');
    }
  });

  it('preserves separate Markov and Polymarket evidence blocks for explicit GOLD combined requests', () => {
    const results = [
      '### markov_distribution(ticker=GLD)',
      '{"data":{"_tool":"markov_distribution","status":"ok","canonical":{"actionSignal":{"recommendation":"BUY","expectedReturn":0.018},"diagnostics":{"predictionConfidence":0.67}}}}',
      '### get_market_data(query=GLD current price)',
      '{"data":{"get_stock_price_GLD":{"ticker":"GLD","price":312.45}}}',
      '### polymarket_forecast(ticker=GLD)',
      '{"data":{"forecastReturn":-0.012,"result":"Polymarket Forecast: GLD | Horizon: 30 days | Grade: B+ (78/100)\\nWill gold finish May above $3,250?: 39% YES"}}',
      '### forecast_arbitrator(ticker=GLD)',
      '{"data":{"result":{"verdict":"NO_TRADE","preferredDirection":"neutral","shouldEnterNow":false}}}',
    ].join('\n');

    const prompt = buildIterationPrompt(
      'Provide a GOLD price forecast based on markov chain and polymarket for the next 30 days with trade direction',
      results,
    );

    expect(prompt).toContain('combined GOLD Markov + Polymarket workflow');
    expect(prompt).toContain('Keep the Markov and Polymarket sections separate');
    expect(prompt).toContain('Do NOT collapse them into one blended gold forecast');
    expect(prompt).toContain('forecast_arbitrator verdict comes after those evidence blocks');
  });

  it('does not inject canonical markov guard for non-markov tool output', () => {
    const results = '### get_market_data(query=BTC)\n{"data":{"ticker":"BTC-USD"}}';
    const prompt = buildIterationPrompt('BTC query', results);
    expect(prompt).not.toContain('markov_distribution results are present');
  });

  it('injects a forecast-lab routing hint when one is provided', () => {
    const prompt = buildIterationPrompt(
      'Optimize the BTC forecast tool',
      '',
      null,
      {
        recommendedProfileId: 'btc-markov-ultra-short-horizon',
        whyMatched: 'Matched improvement cues: optimize.',
        mutationAllowed: true,
        shouldInvokeSkill: true,
      },
    );
    expect(prompt).toContain('Forecast-Lab Routing Hint');
    expect(prompt).toContain('btc-markov-ultra-short-horizon');
    expect(prompt).toContain('Matched improvement cues: optimize.');
    expect(prompt).toContain('Mutation allowed: yes');
    expect(prompt).toContain('Invoke skill("forecast-lab"): yes');
    expect(prompt).toContain('bounded forecast-workflow improvement task');
    expect(prompt).toContain('Call skill("forecast-lab") before ordinary forecast/data tools');
    expect(prompt).toContain('Do NOT auto-run mutation or any tool');
  });
});

describe('injectForecastLabRoutingHint', () => {
  it('leaves prompts unchanged when no routing hint exists', () => {
    expect(injectForecastLabRoutingHint('Query: hello', null)).toBe('Query: hello');
  });
});

describe('buildSystemPrompt tool guidance', () => {
  it('prioritizes markov_distribution for terminal price distributions', async () => {
    const prompt = buildSystemPrompt('gpt-5.4', null, 'cli');
    expect(prompt).toContain('For terminal price probability distributions');
    expect(prompt).toContain('use markov_distribution FIRST');
    expect(prompt).toContain('only valid source of scenario bucket probabilities');
  });

  it('includes BTC/crypto forecast guidance with onchain and fixed income tools', () => {
    const prompt = buildSystemPrompt('gpt-5.4', null, 'cli');
    expect(prompt).toContain('BTC/crypto price forecasts');
    expect(prompt).toContain('get_onchain_crypto');
    expect(prompt).toContain('get_fixed_income');
    expect(prompt).toContain('markov_distribution');
    expect(prompt).toContain('trajectory=true');
  });

  it('requires detailed separate Markov and Polymarket evidence blocks for crypto forecasts', () => {
    const prompt = buildSystemPrompt('gpt-5.4', null, 'cli');
    expect(prompt).toContain('do **not** compress them into a single sentence');
    expect(prompt).toContain('**Markov block**');
    expect(prompt).toContain('**Polymarket block**');
    expect(prompt).toContain('quote 2–4 exact market questions with their YES probabilities');
    expect(prompt).toContain('**Trade decision**');
    expect(prompt).toContain('Keep terminal vs trajectory semantics separate');
    expect(prompt).toContain('Do not merge confidence fields');
  });

  it('mentions probability_assessment skill for full structured BTC/crypto forecast reports', () => {
    const prompt = buildSystemPrompt('gpt-5.4', null, 'cli');
    expect(prompt).toContain('probability_assessment');
  });
});

describe('buildGroupSection', () => {
  it('includes the group name when provided', () => {
    const section = buildGroupSection({
      groupName: 'Stock Traders',
      activationMode: 'mention',
    });
    expect(section).toContain('Stock Traders');
  });

  it('handles missing group name gracefully', () => {
    const section = buildGroupSection({ activationMode: 'mention' });
    expect(section).toContain('WhatsApp group chat');
    expect(section).not.toContain('undefined');
  });

  it('includes members list when provided', () => {
    const section = buildGroupSection({
      activationMode: 'mention',
      membersList: 'Alice, Bob, Charlie',
    });
    expect(section).toContain('Alice, Bob, Charlie');
    expect(section).toContain('Group members');
  });

  it('includes activation mode mention text', () => {
    const section = buildGroupSection({ activationMode: 'mention' });
    expect(section).toContain('@-mentioned');
  });

  it('always includes ## Group Chat header', () => {
    const section = buildGroupSection({ activationMode: 'mention' });
    expect(section).toContain('## Group Chat');
  });

  it('includes group behavior guidelines', () => {
    const section = buildGroupSection({ activationMode: 'mention' });
    expect(section).toContain('Group behavior');
  });
});
