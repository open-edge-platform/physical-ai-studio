import { useMemo } from 'react';

import { Divider, Grid, Heading, View } from '@geti/ui';

import { SchemaRobotType } from '../../../../api/openapi-spec';
import { useRobotForm } from '../../robot-form/provider';
import { SetupRobotViewer } from '../shared/setup-robot-viewer';
import { Stepper } from '../shared/stepper';
import { JointHighlight } from '../shared/use-joint-highlight';
import { CalibrationStep } from './calibration-step';
import { DiagnosticsStep } from './diagnostics-step';
import { MotorSetupStep } from './motor-setup-step';
import { useCenteringAnimation, useRangeOfMotionAnimation } from './use-calibration-animations';
// Lazy import to avoid circular dependency (VerificationStep imports from wizard-provider)
import { VerificationStep } from './verification-step';
import { STEP_LABELS, useSetupActions, useSetupState, WizardStep } from './wizard-provider';

import classes from '../shared/setup-wizard.module.scss';

// ---------------------------------------------------------------------------
// Motor setup order (gripper first) — matches lerobot's setup flow
// ---------------------------------------------------------------------------

const MOTOR_SETUP_ORDER = ['gripper', 'wrist_roll', 'wrist_flex', 'elbow_flex', 'shoulder_lift', 'shoulder_pan'];

// ---------------------------------------------------------------------------
// Derive highlights from setup state (replaces the old useState + useEffect)
// ---------------------------------------------------------------------------

function useHighlights(): JointHighlight[] {
    const { currentStep, wsState, preVerifyProbeResult } = useSetupState();

    return useMemo(() => {
        if (currentStep !== WizardStep.MOTOR_SETUP) {
            return [];
        }

        const progress = wsState.motorSetupProgress;

        // Derive current motor index (same logic as motor-setup-step)
        const currentMotorIndex = (() => {
            const idx = MOTOR_SETUP_ORDER.findIndex((motor) => progress[motor]?.status !== 'success');
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
    const { currentStep, completedSteps } = useSetupState();
    const { visibleSteps, goToStep } = useSetupActions();

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
                <Stepper
                    steps={visibleSteps}
                    currentStep={currentStep}
                    completedSteps={completedSteps}
                    labels={STEP_LABELS}
                    onGoToStep={goToStep}
                />
                <Divider orientation='horizontal' size='S' marginTop='size-200' />
            </View>

            {/* Left column: current step content */}
            <View gridArea='form' UNSAFE_style={{ overflowY: 'auto' }} paddingBottom='size-400' minWidth={0}>
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
