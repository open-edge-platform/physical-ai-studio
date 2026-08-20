export interface MetricsEntry {
    epoch: number | null;
    step: number;
    train_loss: number | null | undefined;
    train_loss_step: number | null | undefined;
    'lr-AdamW': number | null | undefined;
    val_loss: number | null | undefined;
}
