---
name: dcf-valuation
description: Performs discounted cash flow (DCF) valuation analysis to estimate intrinsic value per share. Triggers when user asks for fair value, intrinsic value, DCF, valuation, "what is X worth", price target, undervalued/overvalued analysis, or wants to compare current price to fundamental value.
parameters:
  wacc:
    type: number
    description: "Weighted Average Cost of Capital override (e.g. 0.10 for 10%)"
    default: 0.10
    min: 0.03
    max: 0.30
  growth_rate:
    type: number
    description: "Near-term revenue growth rate assumption (e.g. 0.15 for 15%)"
    default: 0.15
    min: -0.20
    max: 2.00
  terminal_growth_rate:
    type: number
    description: "Long-term terminal growth rate (e.g. 0.025 for 2.5%)"
    default: 0.025
    min: 0.00
    max: 0.10
  years:
    type: number
    description: "DCF projection horizon in years"
    default: 5
    min: 1
    max: 20
---

# DCF Valuation Skill

This skill orchestrates typed valuation tools. **Do not compute DCF, RIM, or
sensitivity grids in prose.** Call the named tools; they return the numbers and
the validation signals. The skill's job is to gather inputs, route them to the
right tools, and present the results with caveats.

## Workflow Checklist

```
DCF Analysis Progress:
- [ ] Step 1: Gather financial data (get_financials + get_market_data)
- [ ] Step 2: Derive FCF growth rate (used as dcf_valuation input)
- [ ] Step 3: Estimate WACC via wacc_inputs (CAPM-based)
- [ ] Step 4: DCF valuation — call dcf_valuation tool
- [ ] Step 5: RIM cross-check — call rim_valuation tool
- [ ] Step 6: Reverse DCF — call reverse_dcf tool
- [ ] Step 7: Present sensitivity grid (returned by dcf_valuation)
- [ ] Step 8: Present validation warnings + EV-vs-reported sanity check
- [ ] Step 9: Persist assumptions via store_financial_insight
- [ ] Step 10: Output structured summary
```

## Step 1: Gather Financial Data

Call `get_financials` and `get_market_data` to assemble the inputs the typed
tools will consume. Capture all values before proceeding to Step 4 — the
`dcf_valuation` tool requires `base_fcf`, `net_debt`, `diluted_shares`,
`wacc`, `growth_rate`, `terminal_growth_rate`, `years`, and (optionally)
`exit_multiple`.

### 1.1 Cash Flow History

**Query:** `"[TICKER] annual cash flow statements for the last 5 years"`

**Extract:** `free_cash_flow`, `net_cash_flow_from_operations`, `capital_expenditure`

**Fallback:** If `free_cash_flow` is missing, calculate:
`free_cash_flow = net_cash_flow_from_operations - capital_expenditure`

### 1.2 Financial Metrics

**Query:** `"[TICKER] financial metrics snapshot"`

**Extract:** `market_cap`, `enterprise_value`, `free_cash_flow_growth`,
`revenue_growth`, `return_on_invested_capital`, `debt_to_equity`,
`free_cash_flow_per_share`

### 1.3 Balance Sheet

**Query:** `"[TICKER] latest balance sheet"`

**Extract:** `total_debt`, `cash_and_equivalents`, `current_investments`,
`outstanding_shares`

**Fallback:** If `current_investments` is missing, use 0.

### 1.4 Net Debt (exact formula — pass to `dcf_valuation`)

```
Net Debt = Total Debt
         + Operating Lease Liabilities     ← from balance sheet (IFRS 16 / ASC 842)
         − Cash and Cash Equivalents
         − Short-term Investments
         − Restricted Cash                 ← only if accessible for debt repayment
```

Lease-heavy companies (airlines, retailers, restaurants) routinely have
operating lease liabilities equal to 30–80% of total "debt" — always include
them. Preferred stock and convertible notes are out of scope for the typed
tool; flag them as caveats in Step 8 if material.

### 1.5 Analyst Estimates (cross-validation only)

**Query:** `"[TICKER] analyst estimates"`

**Extract:** forward `earnings_per_share` by fiscal year.

**Use:** Compare implied EPS growth to the FCF growth you derive in Step 2.
Not a tool input — for sanity-checking only.

### 1.6 Current Price

Call `get_market_data`:

**Query:** `"[TICKER] price snapshot"`

**Extract:** `price` (passes to `reverse_dcf` and to the upside/downside calc).

### 1.7 Company Facts

Call `get_financials`:

**Query:** `"[TICKER] company facts"`

**Extract:** `sector`, `industry`, `market_cap`

**Use:** Pick the appropriate WACC range from [sector-wacc.md](sector-wacc.md)
as a reasonableness check on the `wacc_inputs` result.

### 1.8 Effective Tax Rate

Call `get_financials`:

**Query:** `"[TICKER] effective tax rate income tax expense"`

**Extract:** `effective_tax_rate` or `income_tax_rate` (decimal, e.g. 0.21).

**Fallback sector medians:**

| Sector | Median Effective Rate |
|--------|-----------------------|
| Technology / Software | 18–22% |
| Healthcare | 20–23% |
| Industrials / Materials | 22–25% |
| Energy | 20–25% |
| Financials | 20–24% |
| Consumer Staples | 22–25% |
| Utilities | 24–26% |

**Default if still unavailable:** 21% (US statutory). Do **not** use 30% — it
overstates the tax shield and understates WACC.

## Step 2: Derive FCF Growth Rate

Calculate the {{years}}-year FCF CAGR from the cash flow history. This value
feeds the `dcf_valuation` tool as `growth_rate`.

**Cross-validate with:** `free_cash_flow_growth` (YoY), `revenue_growth`, and
analyst-implied EPS growth from Step 1.5.

**Selection rules:**
- Stable FCF history → use CAGR with a 10–20% haircut.
- Volatile FCF → weight analyst estimates more heavily.
- **Cap at 15%** — sustained higher growth is rare.
- **Active assumption:** `growth_rate = {{growth_rate}}`

If the user passed a `growth_rate` parameter, that value is the override and
wins over any value derived in this step. Pass it straight to `dcf_valuation`.

### FCF Consistency Check (before projection)

1. **Manual calc:** `FCF_manual = operating_cash_flow − capital_expenditure`.
2. Compare to reported `free_cash_flow`. If they differ by **>10%**, use the
   manual figure and note the discrepancy in Step 8 caveats.
3. **Stock-based compensation:** for software/tech names, check whether SBC
   is already excluded from FCF. If it is, add it back for comparability.
4. **One-time items:** exclude large non-recurring items (asset sales,
   litigation settlements) from the base FCF used for projection.

## Step 3: Estimate WACC via CAPM

**Always use the `wacc_inputs` tool — do not estimate WACC from the sector
table.** The tool returns `wacc`, `betaSource`, `ke`, `deRatio`,
`equityWeight`, `debtWeight`, `waccPct`, and a human-readable `note`.

```
wacc_inputs({
  ticker: "[TICKER]",
  cost_of_debt: [pre-tax interest rate from Step 1.3, default 0.055],
  tax_rate: [effective tax rate from Step 1.8, default 0.21],
  risk_free_rate: [10Y Treasury yield; default 0.043 if not yet fetched],
  equity_risk_premium: 0.055
})
```

**Override rules:**
- `{{wacc}}` parameter provided → use it as `wacc` directly in `dcf_valuation`
  and `reverse_dcf` (skip the tool-computed value). The `wacc_inputs` call is
  still made so the `betaSource` and `ke` fields are available for Step 8.

| Situation | Override |
|-----------|----------|
| You found a precise 10Y yield via `web_search` | `risk_free_rate: <decimal>` |
| Balance sheet shows a specific debt cost | `cost_of_debt: <pre-tax decimal>` |
| Income statement shows an exact effective tax rate | `tax_rate: <decimal>` |
| You computed D/E from the balance sheet directly | `debt_to_equity: <decimal>` |

**Reasonableness checks (run before proceeding):**
- WACC should be **2–4% below** `return_on_invested_capital` for
  value-creating companies. If WACC > ROIC, the company may be destroying
  value — note in Step 8 caveats.
- WACC should fall within the sector range from [sector-wacc.md](sector-wacc.md).
  If it does not, review the `betaSource` and D/E inputs.

## Step 4: DCF Valuation (typed tool)

Call `dcf_valuation`. **Do not project FCF, discount, or compute terminal
value in prose** — the tool does it and returns a `validation` block with
warnings/errors you must surface.

```
dcf_valuation({
  ticker: "[TICKER]",
  base_fcf: [latest annual FCF from Step 1.1, in currency units],
  net_debt: [Net Debt from Step 1.4],
  diluted_shares: [shares outstanding from Step 1.3, Step 1.5, or get_financials],
  wacc: [from Step 3 — {{wacc}} override or wacc_inputs.wacc],
  growth_rate: [from Step 2 — {{growth_rate}} override or derived],
  terminal_growth_rate: {{terminal_growth_rate}},
  years: {{years}},
  decay: 0.05,
  exit_multiple: [peer EV/EBITDA from peer-comparison skill if available, else 10]
})
```

**Tool returns:**
- `fairValuePerShare` — base-case intrinsic value (the headline number).
- `enterpriseValue`, `equityValue` — reconciliation fields.
- `sensitivityGrid` — full 2D WACC × terminal-growth table for Step 7.
- `validation.warnings` — e.g. WACC ≤ terminal growth rate, terminal value
  share too high, exit-multiple divergence vs. Gordon growth. Surface each
  in Step 8.
- `validation.errors` — hard failures (e.g. non-positive base FCF, WACC out
  of range). If any error fires, the headline number is not trustworthy —
  present the error, do not paraphrase away the warning.

## Step 5: RIM Cross-Check (typed tool)

Residual Income Model (RIM) is a useful independent cross-check on the DCF
because it uses **book value + projected earnings** rather than free cash
flow. When the two models agree, confidence in the headline fair value goes
up. When they diverge materially, surface it as a caveat (Lundholm &
O'Keefe 2001 — finite-horizon RIM is practically the same as DCF, but
inputs and accrual accounting noise can still cause divergence).

Call `rim_valuation`:

```
rim_valuation({
  ticker: "[TICKER]",
  cost_of_equity: [ke from wacc_inputs output, not WACC — RIM discounts at Ke],
  beginning_book_value: [book value of equity from latest balance sheet],
  projected_net_income: [5–{{years}} years of net income projections; use analyst EPS × diluted_shares, or extrapolate current net income at growth_rate with the same 5% annual decay used in Step 4],
  terminal_growth_rate: {{terminal_growth_rate}},
  diluted_shares: [from Step 4]
})
```

**Compare and report:**
- Compute `divergence_pct = |RIM_fairValue − DCF_fairValue| / DCF_fairValue × 100`.
- **Divergence ≤ 20%** → the two models agree; report RIM as confirmation.
- **Divergence > 20%** → surface as a caveat. Common causes:
  - One-time items distorting either net income or FCF.
  - Aggressive vs. conservative revenue recognition.
  - RIM's terminal-value dependence on book value (high-P/B names get
    penalized; deep-value names get rewarded).
  - Wrong `cost_of_equity` (use Ke, not WACC — RIM discounts equity cash
    flows, not enterprise cash flows).

## Step 6: Reverse DCF (typed tool)

Reverse DCF solves for the growth rate the market is currently pricing in.
Comparing that implied rate to your Step 2 derived rate is the cleanest way
to surface over/undervaluation.

Call `reverse_dcf`:

```
reverse_dcf({
  ticker: "[TICKER]",
  current_price: [from Step 1.6],
  wacc: [same value used in Step 4],
  terminal_growth_rate: {{terminal_growth_rate}},
  years: {{years}},
  decay: 0.05,
  base_fcf: [same base_fcf used in Step 4],
  net_debt: [same net_debt used in Step 4],
  diluted_shares: [same diluted_shares used in Step 4]
})
```

**Tool returns:** `impliedGrowthRate` — the constant growth rate over the
projection horizon that reconciles current price with the model.

**Interpret:**
- `impliedGrowth > growth_rate (Step 2)` → market is more optimistic than
  the fundamentals support. **Overvalued signal.** Compute
  `|implied − fundamental| / fundamental × 100` and report the gap.
- `impliedGrowth < growth_rate (Step 2)` → market is pricing in weaker
  growth than fundamentals. **Undervalued signal.** Same gap calc.
- `impliedGrowth ≈ growth_rate` → market and fundamentals agree. Confidence
  in the base-case fair value is high.

**Caveat:** the reverse-DCF implied growth is only as good as its WACC
assumption. A 1pp WACC error shifts implied growth by roughly 3–5pp at
typical inputs. Always report the comparison alongside the WACC used.

## Step 7: Present Sensitivity Grid

The `dcf_valuation` tool already computed the grid. **Do not recompute it in
prose.** Present the returned `sensitivityGrid` as-is, label the axes, and
highlight the base case.

Format: 3×3 minimum, varying WACC ({{wacc}} ±1pp) and terminal growth
({{terminal_growth_rate}} −0.5pp, base, +0.5pp). If the tool returned a
finer grid (e.g. 5×5), use that — more resolution is better.

```
                | Terminal -0.5pp | Terminal base | Terminal +0.5pp |
WACC -1pp       |                |               |                 |
WACC base       |                |  BASE CASE    |                 |
WACC +1pp       |                |               |                 |
```

## Step 8: Present Validation + Sanity Checks

Two distinct checks:

**A. Tool-returned validation (from Step 4):**
- Surface every entry in `validation.warnings` and `validation.errors`
  verbatim. Do not paraphrase a warning into "looks fine".
- Common warnings to expect:
  - `terminal_value_share_too_high` → growth rate likely too high.
  - `wacc_le_terminal_growth` → invalid; the tool will refuse a clean
    answer in that case.
  - `exit_multiple_divergence` → Gordon-growth TV and exit-multiple TV
    disagree by more than ~30%. Both methods are valid; report the gap.

**B. EV-vs-reported sanity check (do this in prose):**
- Compare `dcf_valuation` returned `enterpriseValue` to the reported
  `enterprise_value` from Step 1.2.
- If off by **>30%**, revisit WACC and growth assumptions before
  presenting. Likely culprits: stale share count, missing leases in net
  debt, beta sourced from a low-quality feed (check `betaSource`).
- Terminal value share of total EV: should be 50–80% for mature companies.
  <40% → near-term projections may be too aggressive. >90% → growth rate
  too high.
- Per-share cross-check: `free_cash_flow_per_share × 15–25` as a rough
  sanity range. Wide divergence is a sign to double-check inputs.

## Step 9: Persist Assumptions

Append a structured summary of the analysis to memory so the same
assumptions can be recalled (and re-used or challenged) in future sessions.

Call `store_financial_insight`:

```
store_financial_insight({
  ticker: "[TICKER]",
  namespace: "dcf",
  content: [structured markdown summary — see template below],
  tags: ["analysis:valuation", "sector:[SECTOR]", "exchange:[EX]"],
  sector: "[SECTOR from Step 1.7]"
})
```

**Content template** (one insight per analysis):

```markdown
## DCF — [TICKER] — [YYYY-MM-DD]

### Headline
- DCF fair value: $[X.XX] / share (vs. current $[X.XX], +/-X%)
- RIM fair value: $[X.XX] / share (divergence vs. DCF: +/-X%)
- Reverse-DCF implied growth: [X.X]% (vs. fundamental [X.X]% — [over/under]valued signal)

### WACC
- WACC: [X.X]% (override: [yes/no])
- Beta: [X.XX] (source: [betaSource from wacc_inputs])
- Ke: [X.X]%, Kd after-tax: [X.X]%, tax rate: [X.X]%
- D/E: [X.XX], equity weight: [X.X]%, debt weight: [X.X]%
- Rfr: [X.X]%, ERP: [X.X]%

### Growth
- FCF growth rate: [X.X]% (cap: 15%; method: [CAGR-with-haircut / analyst-weighted / override])
- Terminal growth rate: [X.X]%
- Years: [N]

### Cross-checks
- EV vs. reported EV: $[calculated] vs. $[reported] ([+/-X]% diff)
- Terminal share of EV: [X]%
- Validation warnings: [list each one or "none"]
- Validation errors: [list each one or "none"]
```

The `namespace: "dcf"` parameter scopes this to DCF analyses only — the
`short-thesis` and `peer-comparison` skills write to their own namespaces so
the three workflows don't pollute each other.

## Step 10: Output Format

Present a structured summary. Required fields:

1. **Valuation Summary:** current price vs. fair value (DCF), upside/downside %.
2. **Key Inputs Table** — must include every row below:

   | Field | Value | Source |
   |-------|-------|--------|
   | Ticker | [TICKER] | — |
   | Current price | $[X.XX] | get_market_data |
   | WACC | [X.X]% | wacc_inputs (override: [yes/no]) |
   | Beta (source) | [X.XX] ([betaSource]) | wacc_inputs |
   | Cost of equity (Ke) | [X.X]% | wacc_inputs |
   | Cost of debt (after-tax) | [X.X]% | wacc_inputs |
   | Tax rate | [X.X]% | Step 1.8 |
   | D/E ratio | [X.XX] | wacc_inputs |
   | FCF growth rate | [X.X]% | Step 2 (override: [yes/no]) |
   | Terminal growth rate | [X.X]% | parameter / sector table |
   | Projection years | [N] | parameter |
   | Net debt | $[X]M | Step 1.4 |
   | Diluted shares | [X]M | Step 1.3 |
   | Base FCF | $[X]M | Step 1.1 |
   | **DCF fair value** | **$[X.XX]** | dcf_valuation |
   | **RIM fair value** | **$[X.XX]** | rim_valuation |
   | **RIM vs. DCF divergence** | **[+/-X]%** | this skill |
   | **Reverse-DCF implied growth** | **[X.X]%** | reverse_dcf |
   | **Market vs. fundamentals** | **[over/under]valued by [X]%** | this skill |
   | Upside / downside | +/-[X]% | this skill |

3. **Projected FCF Table** — returned by `dcf_valuation` (do not recompute).
4. **Sensitivity Grid** — returned by `dcf_valuation` (Step 7).
5. **Validation** — every warning/error from Step 8A + the prose sanity
   checks from Step 8B.
6. **Caveats** — standard DCF limitations (single-point WACC, terminal
   value concentration, sensitivity to small input changes) plus any
   company-specific risks (e.g. RIM > 20% divergence, EV off by >30%,
   implied growth far from fundamentals).
