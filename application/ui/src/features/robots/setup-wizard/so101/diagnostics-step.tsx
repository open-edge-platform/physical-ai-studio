import { ActionButton, Button, Flex, Heading, Icon, Loading, Text } from '@geti/ui';
import { Refresh } from '@geti/ui/icons';
import { useNavigate } from 'react-router';

import { DiagnosticSection } from '../shared/diagnostic-section';
import { InlineAlert } from '../shared/inline-alert';
import { StatusBadge } from '../shared/status-badge';
import { useSetupActions, useSetupState, WizardStep } from './wizard-provider';

import classes from '../shared/setup-wizard.module.scss';

/**
 * Diagnostics step — shows voltage check + motor probe results.
 * Sections auto-collapse on the happy path (voltage OK, all motors found)
 * to reduce noise. Calibration status stays expanded.
 */
export const DiagnosticsStep = () => {
    const { wsState } = useSetupState();
    const { goNext, markCompleted, markSkipped, goToStep, commands } = useSetupActions();
    const navigate = useNavigate();

    const { voltageResult, probeResult, error } = wsState;
    const isLoading = !voltageResult || !probeResult;

    if (error) {
        return (
            <Flex direction='column' gap='size-200'>
                <InlineAlert variant='error'>
                    <strong>Connection Error:</strong> {error}
                </InlineAlert>
            </Flex>
        );
    }

    if (isLoading) {
        return (
            <Flex direction='column' gap='size-300' alignItems='center' justifyContent='center' minHeight='size-3000'>
                <Loading mode='inline' />
                <Text>{wsState.statusMessage ?? 'Connecting to robot...'}</Text>
            </Flex>
        );
    }

    const allMotorsOk = probeResult.all_motors_ok;
    const voltageOk = voltageResult.voltage_ok;
    const voltageReadable = voltageResult.avg_voltage !== null;
    const alreadyCalibrated = probeResult.calibration?.all_calibrated ?? false;

    // When voltage is readable but mismatched, motor/calibration details are noise —
    // the user needs to fix the power supply first. When voltage is unreadable
    // (e.g. after a full reset where all motors share ID 1), we still show the
    // motor section so the user can proceed to motor setup.
    const showMotorSection = !voltageReadable || voltageOk;

    return (
        <Flex direction='column' gap='size-200'>
            {/* Header with refresh */}
            <Flex alignItems='center' justifyContent='space-between'>
                <Heading level={4} margin={0}>
                    Diagnostics
                </Heading>
                <ActionButton isQuiet onPress={commands.reProbe} aria-label='Re-check'>
                    <Icon>
                        <Refresh />
                    </Icon>
                </ActionButton>
            </Flex>

            {/* Voltage section */}
            <DiagnosticSection
                title='Supply Voltage'
                badge={
                    voltageReadable
                        ? voltageOk
                            ? { variant: 'ok', label: `${(voltageResult.avg_voltage ?? 0).toFixed(1)}V OK` }
                            : { variant: 'error', label: `${(voltageResult.avg_voltage ?? 0).toFixed(1)}V MISMATCH` }
                        : { variant: 'pending', label: 'Unreadable' }
                }
                defaultExpanded={!voltageOk || !voltageReadable}
            >
                <Flex direction='column' gap='size-100' marginTop='size-100'>
                    {voltageReadable ? (
                        <>
                            <InlineAlert variant={voltageOk ? 'success' : 'error'}>
                                Average: <strong>{(voltageResult.avg_voltage ?? 0).toFixed(1)}V</strong>
                                {' — '}
                                Expected: {voltageResult.expected_source}
                                {voltageOk ? ' (OK)' : ' (MISMATCH)'}
                            </InlineAlert>
                            {!voltageOk && (
                                <InlineAlert variant='warning'>
                                    The voltage does not match what is expected for a{' '}
                                    {voltageResult.robot_type.includes('Follower') ? 'Follower' : 'Leader'}. Please
                                    verify the robot type and power connections, then re-check.
                                </InlineAlert>
                            )}
                        </>
                    ) : (
                        <InlineAlert variant='warning'>
                            Could not read voltage from any motor. This is normal after a full reset (all motors share
                            the same ID). Proceed to motor setup to assign unique IDs, then re-check.
                        </InlineAlert>
                    )}
                </Flex>
            </DiagnosticSection>

            {/* Motor probe section — only shown when voltage is OK */}
            {showMotorSection && (
                <DiagnosticSection
                    title={`Motors (${probeResult.found_count}/${probeResult.total_count})`}
                    badge={
                        allMotorsOk
                            ? { variant: 'ok', label: 'All found' }
                            : {
                                  variant: 'error',
                                  label: `${probeResult.total_count - probeResult.found_count} missing`,
                              }
                    }
                    defaultExpanded={!allMotorsOk}
                >
                    <Flex direction='column' gap='size-150' marginTop='size-100'>
                        <div className={classes.diagnosticsGrid}>
                            {probeResult.motors.map((motor) => (
                                <div key={motor.name} className={classes.motorRow}>
                                    <span className={classes.motorName}>{motor.name}</span>
                                    <span className={classes.motorId}>ID {motor.motor_id}</span>
                                    {motor.found && motor.model_correct ? (
                                        <StatusBadge variant='ok'>Found</StatusBadge>
                                    ) : motor.found ? (
                                        <StatusBadge variant='error'>Wrong model ({motor.model_number})</StatusBadge>
                                    ) : (
                                        <StatusBadge variant='error'>Not found</StatusBadge>
                                    )}
                                </div>
                            ))}
                        </div>

                        {allMotorsOk ? (
                            <InlineAlert variant='success'>All motors detected and verified.</InlineAlert>
                        ) : (
                            <InlineAlert variant='error'>
                                Some motors are missing or have incorrect firmware. You can run motor setup to assign
                                motor IDs.
                            </InlineAlert>
                        )}
                    </Flex>
                </DiagnosticSection>
            )}

            {/* Calibration status (if motors are OK) — always expanded by default */}
            {showMotorSection && allMotorsOk && probeResult.calibration && (
                <DiagnosticSection
                    title='Calibration Status'
                    badge={
                        alreadyCalibrated
                            ? { variant: 'ok', label: 'Calibrated' }
                            : { variant: 'pending', label: 'Needs calibration' }
                    }
                    defaultExpanded
                >
                    <Flex direction='column' gap='size-150' marginTop='size-100'>
                        <div className={classes.diagnosticsGrid}>
                            {Object.entries(probeResult.calibration.motors).map(([name, cal]) => (
                                <div key={name} className={classes.motorRow}>
                                    <span className={classes.motorName}>{name}</span>
                                    <StatusBadge variant={cal.is_calibrated ? 'ok' : 'error'}>
                                        {cal.is_calibrated ? 'Calibrated' : 'Not calibrated'}
                                    </StatusBadge>
                                    <Text
                                        UNSAFE_style={{
                                            fontSize: 12,
                                            color: 'var(--spectrum-global-color-gray-600)',
                                        }}
                                    >
                                        offset={cal.homing_offset} min={cal.range_min} max={cal.range_max}
                                    </Text>
                                </div>
                            ))}
                        </div>
                        {alreadyCalibrated && (
                            <InlineAlert variant='info'>
                                All motors are already calibrated. You can skip to verification, or recalibrate if
                                needed.
                            </InlineAlert>
                        )}
                    </Flex>
                </DiagnosticSection>
            )}

            {/* Actions */}
            <Flex gap='size-200' justifyContent='space-between'>
                <Button variant='secondary' onPress={() => navigate(-1)}>
                    Back
                </Button>
                <Flex gap='size-200'>
                    {showMotorSection && !allMotorsOk && (
                        <Button
                            variant='accent'
                            onPress={() => {
                                markCompleted(WizardStep.DIAGNOSTICS);
                                commands.startMotorSetup();
                                goNext();
                            }}
                        >
                            Setup Motors
                        </Button>
                    )}
                    {showMotorSection && allMotorsOk && !alreadyCalibrated && (
                        <Button
                            variant='accent'
                            onPress={() => {
                                markCompleted(WizardStep.DIAGNOSTICS);
                                goNext();
                            }}
                        >
                            Calibrate
                        </Button>
                    )}
                    {showMotorSection && allMotorsOk && alreadyCalibrated && (
                        <>
                            <Button
                                variant='secondary'
                                onPress={() => {
                                    markCompleted(WizardStep.DIAGNOSTICS);
                                    goNext();
                                }}
                            >
                                Recalibrate
                            </Button>
                            <Button
                                variant='accent'
                                onPress={() => {
                                    markCompleted(WizardStep.DIAGNOSTICS);
                                    markSkipped(WizardStep.CALIBRATION);
                                    commands.enterVerification();
                                    goToStep(WizardStep.VERIFICATION);
                                }}
                            >
                                Skip to Verification
                            </Button>
                        </>
                    )}
                </Flex>
            </Flex>
        </Flex>
    );
};
