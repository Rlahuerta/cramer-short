import { computeFairValuePerShare, type DcfInputs } from './dcf.js';

export interface SensitivityGrid {
  waccValues: number[];
  growthValues: number[];
  values: number[][];
}

export interface Range {
  min: number;
  max: number;
  step: number;
}

function buildRange({ min, max, step }: Range): number[] {
  if (step <= 0) {
    throw new Error('step must be positive');
  }
  if (min > max) {
    throw new Error('min must be less than or equal to max');
  }
  const values: number[] = [];
  for (let v = min; v <= max + Number.EPSILON; v += step) {
    values.push(v);
  }
  return values;
}

export function sensitivityGrid(
  inputs: DcfInputs,
  waccRange: Range,
  growthRange: Range,
): SensitivityGrid {
  const waccValues = buildRange(waccRange);
  const growthValues = buildRange(growthRange);

  const values: number[][] = [];
  for (let i = 0; i < waccValues.length; i++) {
    const row: number[] = [];
    for (let j = 0; j < growthValues.length; j++) {
      const result = computeFairValuePerShare({
        ...inputs,
        wacc: waccValues[i],
        terminalGrowthRate: growthValues[j],
      });
      row.push(result.fairValuePerShare);
    }
    values.push(row);
  }

  return { waccValues, growthValues, values };
}

export function formatSensitivityGrid(grid: SensitivityGrid): string[][] {
  const header: string[] = [
    'WACC \\ Terminal Growth',
    ...grid.growthValues.map((g) => (g * 100).toFixed(2)),
  ];
  const rows: string[][] = [header];
  for (let i = 0; i < grid.waccValues.length; i++) {
    const row: string[] = [
      (grid.waccValues[i] * 100).toFixed(2) + '%',
      ...grid.values[i].map((v) => v.toFixed(2)),
    ];
    rows.push(row);
  }
  return rows;
}
