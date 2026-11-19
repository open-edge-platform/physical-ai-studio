import { ActionButton, Divider, Flex, Text, View } from '@geti-ui/ui';
import { clsx } from 'clsx';

import type { Step } from './get-reachable-steps';

import classes from './stepper.module.css';

const STEPS: { key: Step; label: string }[] = [
    { key: 'robot', label: 'Choose robot' },
    { key: 'variant', label: 'Choose variant' },
    { key: 'configure', label: 'Connect & configure' },
];

interface StepperProps {
    current: Step;
    reachable: Step[];
    onStep: (step: Step) => void;
}

export const Stepper = ({ current, reachable, onStep }: StepperProps) => (
    <Flex alignItems='center' gap='size-100'>
        {STEPS.map((step, index) => {
            const isCurrent = step.key === current;

            return (
                <Flex key={step.key} alignItems='center' gap='size-100'>
                    {index > 0 && (
                        <View width='size-500'>
                            <Divider size='S' />
                        </View>
                    )}
                    <ActionButton
                        isQuiet
                        isDisabled={!reachable.includes(step.key)}
                        aria-current={isCurrent ? 'step' : undefined}
                        onPress={() => onStep(step.key)}
                        UNSAFE_className={clsx({ [classes.stepCurrent]: isCurrent })}
                    >
                        <Flex alignItems='center' gap='size-100'>
                            <View UNSAFE_className={classes.stepNumber} flexShrink={0}>
                                {index + 1}
                            </View>
                            <Text UNSAFE_className={classes.stepLabel}>{step.label}</Text>
                        </Flex>
                    </ActionButton>
                </Flex>
            );
        })}
    </Flex>
);
