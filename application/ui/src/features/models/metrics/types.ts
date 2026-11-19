export interface MetricsEntry {
    epoch: number | null;
    step: number;
    train_loss: number | null | undefined;
    'lr-AdamW': number | null | undefined;
    val_loss: number | null | undefined;
    // Policy-specific columns, e.g. `val_action_error_<joint>`.
    [key: string]: number | null | undefined;
}

export interface MetricLine {
    name: string;
    color: string;
    strokeWidth?: number;
    getY: (metricsEntry: MetricsEntry) => number | null | undefined;
}
