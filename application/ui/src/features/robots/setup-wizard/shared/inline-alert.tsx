import type { ReactNode } from 'react';

import { Text } from '@geti/ui';

import classes from './setup-wizard.module.scss';

const VARIANT_CLASS = {
    error: classes.errorBox,
    warning: classes.warningBox,
    success: classes.successBox,
    info: classes.infoBox,
} as const;

export type InlineAlertVariant = keyof typeof VARIANT_CLASS;

interface InlineAlertProps {
    variant: InlineAlertVariant;
    children: ReactNode;
}

export const InlineAlert = ({ variant, children }: InlineAlertProps) => (
    <div className={VARIANT_CLASS[variant]}>
        <Text>{children}</Text>
    </div>
);
