import type { ReactNode } from 'react';

import classes from './setup-wizard.module.scss';

const VARIANT_CLASS = {
    ok: classes.statusOk,
    error: classes.statusError,
    pending: classes.statusPending,
    scanning: classes.statusScanning,
} as const;

export type StatusBadgeVariant = keyof typeof VARIANT_CLASS;

interface StatusBadgeProps {
    variant: StatusBadgeVariant;
    children: ReactNode;
}

export const StatusBadge = ({ variant, children }: StatusBadgeProps) => (
    <span className={`${classes.statusBadge} ${VARIANT_CLASS[variant]}`}>{children}</span>
);
