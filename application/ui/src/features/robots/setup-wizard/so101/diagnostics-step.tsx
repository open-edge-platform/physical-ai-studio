import {
    ActionButton,
    Button,
    Disclosure,
    DisclosurePanel,
    DisclosureTitle,
    Flex,
    Heading,
    Icon,
    Loading,
    Text,
} from '@geti/ui';
import { Refresh } from '@geti/ui/icons';
import { useNavigate } from 'react-router';

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
                <div className={classes.errorBox}>
                    <Text>
                        <strong>Connection Error:</strong> {error}
                    </Text>
                </div>
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
            <Disclosure defaultExpanded={!voltageOk || !voltageReadable} isQuiet>
                <DisclosureTitle UNSAFE_className={classes.disclosureHeader}>
                    <Flex alignItems='center' gap='size-100' width='100%'>
                        <Text UNSAFE_style={{ fontWeight: 600, fontSize: 14 }}>Supply Voltage</Text>
                        <Flex flex alignItems='center' justifyContent='end'>
                            {voltageReadable && voltageOk && (
                                <span className={`${classes.statusBadge} ${classes.statusOk}`}>
                                    {(voltageResult.avg_voltage ?? 0).toFixed(1)}V OK
                                </span>
                            )}
                            {voltageReadable && !voltageOk && (
                                <span className={`${classes.statusBadge} ${classes.statusError}`}>
                                    {(voltageResult.avg_voltage ?? 0).toFixed(1)}V MISMATCH
                                </span>
                            )}
                            {!voltageReadable && (
                                <span className={`${classes.statusBadge} ${classes.statusPending}`}>Unreadable</span>
                            )}
                        </Flex>
                    </Flex>
                </DisclosureTitle>
                <DisclosurePanel>
                    <Flex direction='column' gap='size-100' marginTop='size-100'>
                        {voltageReadable ? (
                            <>
                                <div className={voltageOk ? classes.successBox : classes.errorBox}>
                                    <Text>
                                        Average: <strong>{(voltageResult.avg_voltage ?? 0).toFixed(1)}V</strong>
                                        {' — '}
                                        Expected: {voltageResult.expected_source}
                                        {voltageOk ? ' (OK)' : ' (MISMATCH)'}
                                    </Text>
                                </div>
                                {!voltageOk && (
                                    <div className={classes.warningBox}>
                                        <Text>
                                            The voltage does not match what is expected for a{' '}
                                            {voltageResult.robot_type.includes('Follower') ? 'Follower' : 'Leader'}.
                                            Please verify the robot type and power connections, then re-check.
                                        </Text>
                                    </div>
                                )}
                            </>
                        ) : (
                            <div className={classes.warningBox}>
                                <Text>
                                    Could not read voltage from any motor. This is normal after a full reset (all motors
                                    share the same ID). Proceed to motor setup to assign unique IDs, then re-check.
                                </Text>
                            </div>
                        )}
                    </Flex>
                </DisclosurePanel>
            </Disclosure>

            {/* Motor probe section — only shown when voltage is OK */}
            {showMotorSection && (
                <Disclosure defaultExpanded={!allMotorsOk} isQuiet>
                    <DisclosureTitle UNSAFE_className={classes.disclosureHeader}>
                        <Flex alignItems='center' gap='size-100' width='100%'>
                            <Text UNSAFE_style={{ fontWeight: 600, fontSize: 14 }}>
                                Motors ({probeResult.found_count}/{probeResult.total_count})
                            </Text>
                            <Flex flex alignItems='center' justifyContent='end'>
                                {allMotorsOk ? (
                                    <span className={`${classes.statusBadge} ${classes.statusOk}`}>All found</span>
                                ) : (
                                    <span className={`${classes.statusBadge} ${classes.statusError}`}>
                                        {probeResult.total_count - probeResult.found_count} missing
                                    </span>
                                )}
                            </Flex>
                        </Flex>
                    </DisclosureTitle>
                    <DisclosurePanel>
                        <Flex direction='column' gap='size-150' marginTop='size-100'>
                            <div className={classes.diagnosticsGrid}>
                                {probeResult.motors.map((motor) => (
                                    <div key={motor.name} className={classes.motorRow}>
                                        <span className={classes.motorName}>{motor.name}</span>
                                        <span className={classes.motorId}>ID {motor.motor_id}</span>
                                        {motor.found && motor.model_correct ? (
                                            <span className={`${classes.statusBadge} ${classes.statusOk}`}>Found</span>
                                        ) : motor.found ? (
                                            <span className={`${classes.statusBadge} ${classes.statusError}`}>
                                                Wrong model ({motor.model_number})
                                            </span>
                                        ) : (
                                            <span className={`${classes.statusBadge} ${classes.statusError}`}>
                                                Not found
                                            </span>
                                        )}
                                    </div>
                                ))}
                            </div>

                            {allMotorsOk ? (
                                <div className={classes.successBox}>
                                    <Text>All motors detected and verified.</Text>
                                </div>
                            ) : (
                                <div className={classes.errorBox}>
                                    <Text>
                                        Some motors are missing or have incorrect firmware. You can run motor setup to
                                        assign motor IDs.
                                    </Text>
                                </div>
                            )}
                        </Flex>
                    </DisclosurePanel>
                </Disclosure>
            )}

            {/* Calibration status (if motors are OK) — always expanded by default */}
            {showMotorSection && allMotorsOk && probeResult.calibration && (
                <Disclosure defaultExpanded isQuiet>
                    <DisclosureTitle UNSAFE_className={classes.disclosureHeader}>
                        <Flex alignItems='center' gap='size-100' width='100%'>
                            <Text UNSAFE_style={{ fontWeight: 600, fontSize: 14 }}>Calibration Status</Text>
                            <Flex flex alignItems='center' justifyContent='end'>
                                {alreadyCalibrated ? (
                                    <span className={`${classes.statusBadge} ${classes.statusOk}`}>Calibrated</span>
                                ) : (
                                    <span className={`${classes.statusBadge} ${classes.statusPending}`}>
                                        Needs calibration
                                    </span>
                                )}
                            </Flex>
                        </Flex>
                    </DisclosureTitle>
                    <DisclosurePanel>
                        <Flex direction='column' gap='size-150' marginTop='size-100'>
                            <div className={classes.diagnosticsGrid}>
                                {Object.entries(probeResult.calibration.motors).map(([name, cal]) => (
                                    <div key={name} className={classes.motorRow}>
                                        <span className={classes.motorName}>{name}</span>
                                        <span
                                            className={`${classes.statusBadge} ${
                                                cal.is_calibrated ? classes.statusOk : classes.statusError
                                            }`}
                                        >
                                            {cal.is_calibrated ? 'Calibrated' : 'Not calibrated'}
                                        </span>
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
                                <div className={classes.infoBox}>
                                    <Text>
                                        All motors are already calibrated. You can skip to verification, or recalibrate
                                        if needed.
                                    </Text>
                                </div>
                            )}
                        </Flex>
                    </DisclosurePanel>
                </Disclosure>
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
