import { Text } from '@geti/ui';

import classes from './setup-wizard.module.scss';

// ---------------------------------------------------------------------------
// Generic stepper — accepts data via props so it can be used by any wizard
// ---------------------------------------------------------------------------

interface StepperProps<S extends string> {
    steps: S[];
    currentStep: S;
    completedSteps: Set<S>;
    labels: Record<S, string>;
    onGoToStep: (step: S) => void;
}

/**
 * Horizontal step indicator bar showing progress through a wizard.
 * Fully generic — receives step data via props rather than pulling
 * from a specific wizard context.
 */
export const Stepper = <S extends string>({
    steps,
    currentStep,
    completedSteps,
    labels,
    onGoToStep,
}: StepperProps<S>) => {
    return (
        <div className={classes.stepper}>
            {steps.map((step, index) => {
                const isActive = step === currentStep;
                const isCompleted = completedSteps.has(step);
                const currentIndex = steps.indexOf(currentStep);
                const isClickable = index < currentIndex;

                return (
                    <div key={step} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        {index > 0 && <div className={classes.stepDivider} />}
                        <div
                            className={[
                                classes.step,
                                isActive ? classes.stepActive : '',
                                !isClickable && !isActive ? classes.stepDisabled : '',
                            ].join(' ')}
                            onClick={() => {
                                if (isClickable) onGoToStep(step);
                            }}
                            role='button'
                            tabIndex={isClickable ? 0 : -1}
                        >
                            <span
                                className={`${classes.stepNumber} ${
                                    isActive
                                        ? classes.stepNumberActive
                                        : isCompleted
                                          ? classes.stepNumberCompleted
                                          : classes.stepNumberDefault
                                }`}
                            >
                                {isCompleted ? '\u2713' : index + 1}
                            </span>
                            <Text UNSAFE_className={classes.stepLabel}>{labels[step]}</Text>
                        </div>
                    </div>
                );
            })}
        </div>
    );
};
