import { useState } from 'react';

import { Flex, View } from '@geti-ui/ui';

import { FormPreviewLayout } from '../../../../components/form-preview-layout';
import { Preview } from '../preview';
import { useRobotForm } from '../provider';
import { CreateRobotForm } from './create-form';
import { FamilyStep } from './family-step';
import { getReachableSteps, type Step } from './get-reachable-steps';
import { getFamilyKey, RobotFamily, useRobotFamilies } from './robot-families';
import { Stepper } from './stepper';
import { useSelectRobotType } from './use-select-robot-type';
import { VariantStep } from './variant-step';

export const CreateRobotWizard = () => {
    const families = useRobotFamilies();
    const { activeType } = useRobotForm();
    const selectType = useSelectRobotType();

    const [familyKey, setFamilyKey] = useState<string | null>(() =>
        activeType === undefined ? null : getFamilyKey(activeType)
    );
    const [requestedStep, setRequestedStep] = useState<Step>(() => (activeType === undefined ? 'robot' : 'configure'));

    const family = families.find(({ key }) => key === familyKey);
    const reachable = getReachableSteps(family, activeType);
    // Never render a step whose prerequisites are not met, whatever was requested.
    const step = reachable.includes(requestedStep) ? requestedStep : reachable[reachable.length - 1];

    const chooseFamily = (next: RobotFamily) => {
        setFamilyKey(next.key);

        // Single variant: preselect it, but still show the step so the user sees what they get.
        if (next.variants.length === 1) {
            selectType(next.variants[0].type);
        }

        setRequestedStep('variant');
    };

    const chooseVariant = (type: string) => {
        selectType(type);
        setRequestedStep('configure');
    };

    return (
        <Flex direction='column' height='100%' minHeight={0}>
            <View paddingX='size-400' paddingTop='size-300' paddingBottom='size-200'>
                <Stepper current={step} reachable={reachable} onStep={setRequestedStep} />
            </View>
            {step === 'configure' && family !== undefined ? (
                <View flex={1} minHeight={0} overflow='hidden'>
                    <FormPreviewLayout
                        preview={<Preview />}
                        form={<CreateRobotForm onChangeRobot={() => setRequestedStep('variant')} />}
                    />
                </View>
            ) : (
                <View flex={1} minHeight={0} overflow='auto' paddingX='size-400' paddingBottom='size-400'>
                    <Flex direction='column' gap='size-250'>
                        {step === 'robot' && (
                            <FamilyStep families={families} selectedKey={familyKey} onSelect={chooseFamily} />
                        )}
                        {step === 'variant' && family !== undefined && (
                            <VariantStep family={family} onSelect={chooseVariant} />
                        )}
                    </Flex>
                </View>
            )}
        </Flex>
    );
};
