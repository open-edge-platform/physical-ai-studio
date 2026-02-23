import { createContext, ReactNode, useCallback, useContext, useMemo, useState } from 'react';

import { useProjectId } from '../../../projects/use-project';
import { useRobotForm } from '../../robot-form/provider';
import { MotorProbeResult, SetupWebSocketState, useSetupWebSocket } from './use-setup-websocket';

// ---------------------------------------------------------------------------
// Wizard step definitions
// ---------------------------------------------------------------------------

export enum WizardStep {
    /** Checking voltage + probing motors (auto-runs on websocket connect) */
    DIAGNOSTICS = 'diagnostics',
    /** Per-motor ID assignment (only if motors missing) */
    MOTOR_SETUP = 'motor_setup',
    /** Calibration: center robot + record range of motion */
    CALIBRATION = 'calibration',
    /** Final verification: 3D preview synced to live robot state */
    VERIFICATION = 'verification',
}

export const WIZARD_STEPS: WizardStep[] = [
    WizardStep.DIAGNOSTICS,
    WizardStep.MOTOR_SETUP,
    WizardStep.CALIBRATION,
    WizardStep.VERIFICATION,
];

export const STEP_LABELS: Record<WizardStep, string> = {
    [WizardStep.DIAGNOSTICS]: 'Diagnostics',
    [WizardStep.MOTOR_SETUP]: 'Motor Setup',
    [WizardStep.CALIBRATION]: 'Calibration',
    [WizardStep.VERIFICATION]: 'Verification',
};

export type CalibrationPhase = 'instructions' | 'homing' | 'recording' | 'done';

// ---------------------------------------------------------------------------
// Context shapes
// ---------------------------------------------------------------------------

interface SetupState {
    // Wizard navigation
    currentStep: WizardStep;
    completedSteps: Set<WizardStep>;
    skippedSteps: Set<WizardStep>;
    // WebSocket state
    wsState: SetupWebSocketState;
    // Domain state
    calibrationPhase: CalibrationPhase;
    preVerifyProbeResult: MotorProbeResult | null;
}

interface SetupActions {
    // Wizard navigation
    goToStep: (step: WizardStep) => void;
    goNext: () => void;
    goBack: () => void;
    markCompleted: (step: WizardStep) => void;
    markSkipped: (step: WizardStep) => void;
    unmarkSkipped: (step: WizardStep) => void;
    canGoNext: boolean;
    canGoBack: boolean;
    stepIndex: number;
    visibleSteps: WizardStep[];
    // WS commands
    commands: ReturnType<typeof useSetupWebSocket>['commands'];
    // Domain state setters
    setCalibrationPhase: (phase: CalibrationPhase) => void;
    setPreVerifyProbeResult: (result: MotorProbeResult | null) => void;
}

const SetupStateContext = createContext<SetupState | null>(null);
const SetupActionsContext = createContext<SetupActions | null>(null);

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export const SetupWizardProvider = ({ children }: { children: ReactNode }) => {
    const { project_id: projectId } = useProjectId();
    const robotForm = useRobotForm();

    // -----------------------------------------------------------------------
    // Wizard step state
    // -----------------------------------------------------------------------

    const [wizardState, setWizardState] = useState({
        currentStep: WizardStep.DIAGNOSTICS as WizardStep,
        completedSteps: new Set<WizardStep>(),
        skippedSteps: new Set<WizardStep>(),
    });

    const visibleSteps = useMemo(
        () => WIZARD_STEPS.filter((s) => !wizardState.skippedSteps.has(s)),
        [wizardState.skippedSteps]
    );

    const stepIndex = visibleSteps.indexOf(wizardState.currentStep);

    const goToStep = useCallback((step: WizardStep) => {
        setWizardState((prev) => ({ ...prev, currentStep: step }));
    }, []);

    const goNext = useCallback(() => {
        setWizardState((prev) => {
            const visible = WIZARD_STEPS.filter((s) => !prev.skippedSteps.has(s));
            const idx = visible.indexOf(prev.currentStep);
            if (idx < visible.length - 1) {
                return { ...prev, currentStep: visible[idx + 1] };
            }
            return prev;
        });
    }, []);

    const goBack = useCallback(() => {
        setWizardState((prev) => {
            const visible = WIZARD_STEPS.filter((s) => !prev.skippedSteps.has(s));
            const idx = visible.indexOf(prev.currentStep);
            if (idx > 0) {
                return { ...prev, currentStep: visible[idx - 1] };
            }
            return prev;
        });
    }, []);

    const markCompleted = useCallback((step: WizardStep) => {
        setWizardState((prev) => {
            const next = new Set(prev.completedSteps);
            next.add(step);
            return { ...prev, completedSteps: next };
        });
    }, []);

    const markSkipped = useCallback((step: WizardStep) => {
        setWizardState((prev) => {
            const next = new Set(prev.skippedSteps);
            next.add(step);
            return { ...prev, skippedSteps: next };
        });
    }, []);

    const unmarkSkipped = useCallback((step: WizardStep) => {
        setWizardState((prev) => {
            const next = new Set(prev.skippedSteps);
            next.delete(step);
            return { ...prev, skippedSteps: next };
        });
    }, []);

    // -----------------------------------------------------------------------
    // Domain state
    // -----------------------------------------------------------------------

    const [calibrationPhase, setCalibrationPhase] = useState<CalibrationPhase>('instructions');
    const [preVerifyProbeResult, setPreVerifyProbeResult] = useState<MotorProbeResult | null>(null);

    // -----------------------------------------------------------------------
    // WebSocket hook
    // -----------------------------------------------------------------------

    const serialNumber = robotForm.serial_number ?? '';
    const robotType = robotForm.type ?? '';
    const wsEnabled = !!serialNumber && !!robotType;

    const { state: wsState, commands } = useSetupWebSocket({
        projectId,
        robotType,
        serialNumber,
        enabled: wsEnabled,
    });

    // -----------------------------------------------------------------------
    // Auto-skip MOTOR_SETUP when all motors are OK (render-time adjustment)
    // -----------------------------------------------------------------------

    const [prevAllMotorsOk, setPrevAllMotorsOk] = useState<boolean | undefined>(undefined);
    const allMotorsOk = wsState.probeResult?.all_motors_ok;
    const motorSetupIdx = WIZARD_STEPS.indexOf(WizardStep.MOTOR_SETUP);
    const currentIdx = WIZARD_STEPS.indexOf(wizardState.currentStep);
    const beforeMotorSetup = currentIdx < motorSetupIdx;

    if (allMotorsOk !== prevAllMotorsOk) {
        setPrevAllMotorsOk(allMotorsOk);
        if (allMotorsOk === true && beforeMotorSetup) {
            markSkipped(WizardStep.MOTOR_SETUP);
        } else if (allMotorsOk === false) {
            unmarkSkipped(WizardStep.MOTOR_SETUP);
        }
    }

    // -----------------------------------------------------------------------
    // Context values
    // -----------------------------------------------------------------------

    const state: SetupState = {
        currentStep: wizardState.currentStep,
        completedSteps: wizardState.completedSteps,
        skippedSteps: wizardState.skippedSteps,
        wsState,
        calibrationPhase,
        preVerifyProbeResult,
    };

    const actions: SetupActions = useMemo(
        () => ({
            goToStep,
            goNext,
            goBack,
            markCompleted,
            markSkipped,
            unmarkSkipped,
            canGoNext: stepIndex < visibleSteps.length - 1,
            canGoBack: stepIndex > 0,
            stepIndex,
            visibleSteps,
            commands,
            setCalibrationPhase,
            setPreVerifyProbeResult,
        }),
        [goToStep, goNext, goBack, markCompleted, markSkipped, unmarkSkipped, stepIndex, visibleSteps, commands]
    );

    return (
        <SetupStateContext.Provider value={state}>
            <SetupActionsContext.Provider value={actions}>{children}</SetupActionsContext.Provider>
        </SetupStateContext.Provider>
    );
};

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

export const useSetupState = () => {
    const ctx = useContext(SetupStateContext);
    if (ctx === null) throw new Error('useSetupState must be used within SetupWizardProvider');
    return ctx;
};

export const useSetupActions = () => {
    const ctx = useContext(SetupActionsContext);
    if (ctx === null) throw new Error('useSetupActions must be used within SetupWizardProvider');
    return ctx;
};
