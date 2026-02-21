import { useMemo } from 'react';

import { Button, Divider, Grid, Heading, View } from '@geti/ui';
import { SchemaRobotType } from '../../../api/openapi-spec';

import { RobotForm } from '../robot-form/form';
import { useRobotForm } from '../robot-form/provider';
import { CalibrationStep } from './calibration-step';
import { DiagnosticsStep } from './diagnostics-step';
import { MotorSetupStep } from './motor-setup-step';
import { SetupRobotViewer } from './setup-robot-viewer';
import { Stepper } from './stepper';
import { useCenteringAnimation, useRangeOfMotionAnimation } from './use-calibration-animations';
import { JointHighlight } from './use-joint-highlight';
import { useSetupActions, useSetupState, WizardStep } from './wizard-provider';

import classes from './setup-wizard.module.scss';

// ---------------------------------------------------------------------------
// Motor setup order (gripper first) — matches lerobot's setup flow
// ---------------------------------------------------------------------------

const MOTOR_SETUP_ORDER = ['gripper', 'wrist_roll', 'wrist_flex', 'elbow_flex', 'shoulder_lift', 'shoulder_pan'];

/**
 * The "Begin Setup" button replaces the standard "Add Robot" submit button
 * in the initial robot info form step.
 */
const BeginSetupButton = () => {
    const robotForm = useRobotForm();
    const { goNext, markCompleted } = useSetupActions();

    const isDisabled =
        !robotForm.name || !robotForm.type || (!robotForm.serial_number && !robotForm.connection_string);

    const isSO101 = robotForm.type?.toLowerCase().startsWith('so101') ?? false;

    return (
        <Button
            variant='accent'
            isDisabled={isDisabled || !isSO101}
            onPress={() => {
                markCompleted(WizardStep.ROBOT_INFO);
                goNext();
            }}
        >
            {isSO101 ? 'Begin Setup' : 'Add Robot'}
        </Button>
    );
};

// ---------------------------------------------------------------------------
// Derive highlights from setup state (replaces the old useState + useEffect)
// ---------------------------------------------------------------------------

function useHighlights(): JointHighlight[] {
    const { currentStep, wsState, preVerifyProbeResult } = useSetupState();

    return useMemo(() => {
        if (currentStep !== WizardStep.MOTOR_SETUP) return [];

        const progress = wsState.motorSetupProgress;

        // Derive current motor index (same logic as motor-setup-step)
        const currentMotorIndex = (() => {
            const idx = MOTOR_SETUP_ORDER.findIndex(
                (motor) => progress[motor]?.status !== 'success',
            );
            return idx === -1 ? MOTOR_SETUP_ORDER.length : idx;
        })();

        const allDone = currentMotorIndex >= MOTOR_SETUP_ORDER.length;

        if (allDone) {
            // Derive reassembly state
            const hasTriggeredVerify = preVerifyProbeResult !== null;
            const newResultArrived = hasTriggeredVerify && wsState.probeResult !== preVerifyProbeResult;

            if (newResultArrived) {
                // After verification: color each motor based on probe result
                const probeMotors = wsState.probeResult?.motors ?? [];
                return probeMotors.map((m) => ({
                    joint: m.name,
                    color: m.found ? ('positive' as const) : ('negative' as const),
                }));
            }
            // Reassembly idle/verifying: highlight all motors in accent
            return MOTOR_SETUP_ORDER.map((j) => ({ joint: j, color: 'accent' as const }));
        }

        const currentMotor = MOTOR_SETUP_ORDER[currentMotorIndex];
        if (currentMotor) {
            return [{ joint: currentMotor, color: 'accent' as const }];
        }

        return [];
    }, [currentStep, wsState.motorSetupProgress, wsState.probeResult, preVerifyProbeResult]);
}

// ---------------------------------------------------------------------------
// Right-column viewer panel
// ---------------------------------------------------------------------------

/**
 * Right column: shows the 3D URDF viewer with contextual animations.
 *
 * - ROBOT_INFO: shows viewer when a robot type is selected
 * - DIAGNOSTICS / MOTOR_SETUP: idle viewer (with highlights during motor setup)
 * - CALIBRATION + homing phase: centering animation
 * - CALIBRATION + recording phase: range-of-motion animation
 * - VERIFICATION: live-synced viewer (sync is driven by VerificationStep)
 */
const ViewerPanel = () => {
    const robotForm = useRobotForm();
    const { currentStep, calibrationPhase } = useSetupState();
    const highlights = useHighlights();

    const robotType = robotForm.type || null;

    const isCentering =
        currentStep === WizardStep.CALIBRATION &&
        (calibrationPhase === 'instructions' || calibrationPhase === 'homing');

    const isRangeDemo = currentStep === WizardStep.CALIBRATION && calibrationPhase === 'recording';

    // Drive animations — these hooks are no-ops when `enabled` is false
    useCenteringAnimation(isCentering);
    useRangeOfMotionAnimation(isRangeDemo);

    if (!robotType) {
        return (
            <View
                height='100%'
                backgroundColor='gray-200'
                UNSAFE_style={{
                    borderRadius: 'var(--spectrum-alias-border-radius-regular)',
                    borderColor: 'var(--spectrum-global-color-gray-700)',
                    borderWidth: '1px',
                    borderStyle: 'dashed',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                }}
            >
                <Heading level={4} UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-500)' }}>
                    Select a robot type to preview
                </Heading>
            </View>
        );
    }

    return (
        <View
            height='100%'
            backgroundColor='gray-200'
            UNSAFE_style={{
                borderRadius: 'var(--spectrum-alias-border-radius-regular)',
                overflow: 'hidden',
            }}
        >
            <SetupRobotViewer
                robotType={robotType as SchemaRobotType}
                highlights={currentStep === WizardStep.MOTOR_SETUP ? highlights : []}
            />
        </View>
    );
};

// ---------------------------------------------------------------------------
// Main wizard content
// ---------------------------------------------------------------------------

/**
 * Main wizard content — two-column layout.
 * Left: stepper + current step form/content.
 * Right: 3D robot viewer with contextual animations.
 *
 * All setup state is provided by the SetupWizardProvider context —
 * this component simply renders the layout and switches between steps.
 */
export const SetupWizardContent = () => {
    const { currentStep } = useSetupState();

    return (
        <Grid
            areas={['stepper stepper', 'form viewer']}
            columns={['size-6000', '1fr']}
            rows={['auto', '1fr']}
            gap='size-400'
            height='100%'
            UNSAFE_className={classes.wizardGrid}
        >
            {/* Top row: stepper spans full width */}
            <View gridArea='stepper'>
                <Stepper />
                <Divider orientation='horizontal' size='S' marginTop='size-200' />
            </View>

            {/* Left column: current step content */}
            <View gridArea='form' UNSAFE_style={{ overflowY: 'auto' }} paddingBottom='size-400' minWidth={0}>
                {currentStep === WizardStep.ROBOT_INFO && (
                    <RobotForm heading='Add new robot' submitButton={<BeginSetupButton />} />
                )}

                {currentStep === WizardStep.DIAGNOSTICS && <DiagnosticsStep />}

                {currentStep === WizardStep.MOTOR_SETUP && <MotorSetupStep />}

                {currentStep === WizardStep.CALIBRATION && <CalibrationStep />}

                {currentStep === WizardStep.VERIFICATION && <VerificationStep />}
            </View>

            {/* Right column: 3D robot viewer */}
            <View gridArea='viewer'>
                <ViewerPanel />
            </View>
        </Grid>
    );
};

// Lazy import to avoid circular dependency (VerificationStep imports from wizard-provider)
import { VerificationStep } from './verification-step';
