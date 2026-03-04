import { useMemo } from 'react';

import { Button, Flex, Heading, Text } from '@geti/ui';

import { InlineAlert } from '../shared/inline-alert';
import { StatusBadge, type StatusBadgeVariant } from '../shared/status-badge';
import { useSetupActions, useSetupState, WizardStep } from './wizard-provider';

import classes from '../shared/setup-wizard.module.scss';

/** The motors in reverse order (gripper first) — matches lerobot's setup flow */
const MOTOR_SETUP_ORDER = ['gripper', 'wrist_roll', 'wrist_flex', 'elbow_flex', 'shoulder_lift', 'shoulder_pan'];

type ReassemblyState = 'idle' | 'verifying' | 'success' | 'failed';

/**
 * Motor Setup step — guides the user through connecting motors one at a time
 * to assign correct IDs. After all motors are set up, enters a reassembly
 * phase where the user reattaches all motors and verifies they respond.
 */
export const MotorSetupStep = () => {
    const { wsState, preVerifyProbeResult } = useSetupState();
    const { goNext, goBack, markCompleted, commands, setPreVerifyProbeResult } = useSetupActions();

    const progress = wsState.motorSetupProgress;

    // Derive current motor index: first motor in MOTOR_SETUP_ORDER without 'success' status.
    // If all succeed, index goes past the end (triggering reassembly phase).
    // If a motor errors, index stays on it until the user retries and it succeeds.
    const currentMotorIndex = useMemo(() => {
        const idx = MOTOR_SETUP_ORDER.findIndex((motor) => progress[motor]?.status !== 'success');
        return idx === -1 ? MOTOR_SETUP_ORDER.length : idx;
    }, [progress]);

    const currentMotor = MOTOR_SETUP_ORDER[currentMotorIndex] ?? null;
    const allDone = currentMotorIndex >= MOTOR_SETUP_ORDER.length;
    const currentProgress = currentMotor ? progress[currentMotor] : null;
    const isScanning = currentProgress?.status === 'scanning';

    // Derive reassembly state from preVerifyProbeResult and wsState.probeResult.
    // - 'idle': verification hasn't been triggered (preVerifyProbeResult is null)
    // - 'verifying': triggered but no new probe result yet (same reference)
    // - 'success'/'failed': new probe result arrived with the verdict
    const reassemblyState = useMemo<ReassemblyState>(() => {
        if (!allDone || preVerifyProbeResult === null) {
            return 'idle';
        }
        // Still waiting — the current probeResult is the same reference as before verification
        if (wsState.probeResult === preVerifyProbeResult) {
            return 'verifying';
        }
        // New result arrived
        if (wsState.probeResult?.all_motors_ok) {
            return 'success';
        }
        return 'failed';
    }, [allDone, preVerifyProbeResult, wsState.probeResult]);

    // Compute which motors are missing (for the failed state)
    const missingMotors = useMemo(() => {
        if (!wsState.probeResult) {
            return [];
        }
        return wsState.probeResult.motors.filter((m) => !m.found).map((m) => m.name);
    }, [wsState.probeResult]);

    const handleConnectMotor = () => {
        if (currentMotor && !isScanning) {
            commands.motorConnected(currentMotor);
        }
    };

    const handleVerifyMotors = () => {
        setPreVerifyProbeResult(wsState.probeResult);
        commands.finishMotorSetup();
    };

    const handleContinue = () => {
        markCompleted(WizardStep.MOTOR_SETUP);
        goNext();
    };

    return (
        <Flex direction='column' gap='size-300'>
            <div className={classes.sectionCard}>
                <Flex direction='column' gap='size-200'>
                    <Heading level={4}>Motor Setup Progress</Heading>

                    {MOTOR_SETUP_ORDER.map((motor, index) => {
                        const motorProgress = progress[motor];
                        const isCurrentMotor = index === currentMotorIndex && !allDone;
                        const isPast = index < currentMotorIndex;

                        let statusVariant: StatusBadgeVariant = 'pending';
                        let statusText = 'Pending';

                        if (motorProgress) {
                            switch (motorProgress.status) {
                                case 'scanning':
                                    statusVariant = 'scanning';
                                    statusText = 'Scanning...';
                                    break;
                                case 'success':
                                    statusVariant = 'ok';
                                    statusText = 'Done';
                                    break;
                                case 'error':
                                    statusVariant = 'error';
                                    statusText = 'Error';
                                    break;
                            }
                        } else if (isPast) {
                            statusVariant = 'ok';
                            statusText = 'Done';
                        }

                        // In reassembly verification results, show per-motor found/missing status
                        if (allDone && reassemblyState !== 'idle' && wsState.probeResult) {
                            const probeEntry = wsState.probeResult.motors.find((m) => m.name === motor);
                            if (probeEntry) {
                                if (probeEntry.found) {
                                    statusVariant = 'ok';
                                    statusText = 'Found';
                                } else {
                                    statusVariant = 'error';
                                    statusText = 'Missing';
                                }
                            }
                        }

                        return (
                            <div
                                key={motor}
                                className={classes.motorRow}
                                style={{
                                    borderColor: isCurrentMotor
                                        ? 'var(--spectrum-accent-color-500, #1473e6)'
                                        : undefined,
                                    borderWidth: isCurrentMotor ? 2 : undefined,
                                }}
                            >
                                <span className={classes.motorName}>
                                    {isCurrentMotor ? '\u25B6 ' : ''}
                                    {motor}
                                </span>
                                <span className={classes.motorId}>
                                    ID {index < MOTOR_SETUP_ORDER.length ? MOTOR_SETUP_ORDER.length - index : '?'}
                                </span>
                                <StatusBadge variant={statusVariant}>{statusText}</StatusBadge>
                                {motorProgress?.status === 'error' && (
                                    <Text
                                        UNSAFE_style={{
                                            fontSize: 12,
                                            color: 'var(--spectrum-semantic-negative-color-default)',
                                        }}
                                    >
                                        {motorProgress.message}
                                    </Text>
                                )}
                            </div>
                        );
                    })}
                </Flex>
            </div>

            {/* Contextual instructions & status — positioned close to action buttons */}
            {!allDone && currentMotor && (
                <InlineAlert variant='warning'>
                    {currentMotorIndex === 0 ? (
                        <>
                            Connect <strong>only the &apos;{currentMotor}&apos;</strong> motor to the controller board.
                            When ready, click the button below.
                        </>
                    ) : (
                        <>
                            Disconnect the previous motor, then connect{' '}
                            <strong>only the &apos;{currentMotor}&apos;</strong> motor to the controller board. When
                            ready, click the button below.
                        </>
                    )}
                </InlineAlert>
            )}

            {allDone && reassemblyState === 'idle' && (
                <InlineAlert variant='info'>
                    All motor IDs have been assigned. Now <strong>reconnect all motors</strong> to the controller board
                    and reassemble the robot. When ready, click &ldquo;Verify Motors&rdquo; to confirm all motors are
                    responding.
                </InlineAlert>
            )}

            {allDone && reassemblyState === 'verifying' && (
                <InlineAlert variant='info'>Verifying all motors... Please wait.</InlineAlert>
            )}

            {allDone && reassemblyState === 'success' && (
                <InlineAlert variant='success'>
                    All motors verified successfully! The robot is fully assembled and ready for calibration.
                </InlineAlert>
            )}

            {allDone && reassemblyState === 'failed' && (
                <InlineAlert variant='warning'>
                    {missingMotors.length} motor{missingMotors.length !== 1 ? 's' : ''} not found:{' '}
                    <strong>{missingMotors.join(', ')}</strong>. Please check the connections and try again.
                </InlineAlert>
            )}

            {/* Action buttons */}
            <Flex gap='size-200' justifyContent='space-between'>
                <Button variant='secondary' onPress={goBack}>
                    Back
                </Button>
                <Flex gap='size-200'>
                    {/* Individual motor setup button */}
                    {!allDone && currentMotor && (
                        <Button variant='accent' isDisabled={isScanning} onPress={handleConnectMotor}>
                            {isScanning ? 'Scanning...' : `Assign ID for '${currentMotor}'`}
                        </Button>
                    )}

                    {/* Reassembly phase buttons */}
                    {allDone && reassemblyState !== 'success' && (
                        <Button
                            variant='accent'
                            isDisabled={reassemblyState === 'verifying'}
                            onPress={handleVerifyMotors}
                        >
                            {reassemblyState === 'verifying'
                                ? 'Verifying...'
                                : reassemblyState === 'failed'
                                  ? 'Retry Verification'
                                  : 'Verify Motors'}
                        </Button>
                    )}

                    {allDone && reassemblyState === 'success' && (
                        <Button variant='accent' onPress={handleContinue}>
                            Continue to Calibration
                        </Button>
                    )}
                </Flex>
            </Flex>
        </Flex>
    );
};
