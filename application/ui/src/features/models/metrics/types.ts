export interface MetricsEntry {
    epoch: number | null;
    step: number;
    train_loss: number | null | undefined;
    val_loss: number | null | undefined;
    // Dynamic columns, e.g. `lr-AdamW_pg1` and `val_action_error_<joint>`.
    [key: string]: number | null | undefined;
}

export const hasMetricValue = (entry: MetricsEntry, getY: MetricLine['getY'], lines?: MetricLine[]): boolean =>
    lines ? lines.some((line) => line.getY(entry) != null) : getY(entry) != null;

export interface MetricLine {
    name: string;
    color: string;
    strokeWidth?: number;
    strokeDasharray?: string;
    primary?: boolean;
    getY: (metricsEntry: MetricsEntry) => number | null | undefined;
}
