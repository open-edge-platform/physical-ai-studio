import { Text } from '@geti/ui';

import { STEP_LABELS, useSetupActions, useSetupState } from './wizard-provider';

import classes from './setup-wizard.module.scss';

/**
 * Horizontal step indicator bar showing progress through the wizard.
 */
export const Stepper = () => {
    const { currentStep, completedSteps } = useSetupState();
    const { visibleSteps, goToStep } = useSetupActions();

    return (
        <div className={classes.stepper}>
            {visibleSteps.map((step, index) => {
                const isActive = step === currentStep;
                const isCompleted = completedSteps.has(step);
                const currentIndex = visibleSteps.indexOf(currentStep);
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
                                if (isClickable) goToStep(step);
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
                            <Text UNSAFE_className={classes.stepLabel}>{STEP_LABELS[step]}</Text>
                        </div>
                    </div>
                );
            })}
        </div>
    );
};
