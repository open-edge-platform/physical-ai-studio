import { View } from '@geti/ui';

import { RobotFormProvider } from '../../features/robots/robot-form/provider';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';
import { SetupWizardContent } from '../../features/robots/setup-wizard/setup-wizard';
import { SetupWizardProvider } from '../../features/robots/setup-wizard/wizard-provider';

export const New = () => {
    return (
        <RobotModelsProvider>
            <RobotFormProvider>
                <SetupWizardProvider>
                    <View
                        height='100%'
                        backgroundColor='gray-100'
                        padding='size-400'
                        UNSAFE_style={{ overflow: 'hidden' }}
                    >
                        <SetupWizardContent />
                    </View>
                </SetupWizardProvider>
            </RobotFormProvider>
        </RobotModelsProvider>
    );
};
