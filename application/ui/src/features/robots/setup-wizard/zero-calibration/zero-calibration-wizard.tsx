import { ReactNode, useMemo, useState } from 'react';

import { Button, Divider, Flex, Grid, Heading, Loading, minmax, Text, View } from '@geti-ui/ui';
import { useNavigate } from 'react-router';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../../api/client';
import { paths } from '../../../../router';
import { useProjectId } from '../../../projects/use-project';
import { buildRobotBody } from '../../robot-form/form-data';
import { useRobotForm } from '../../robot-form/provider';
import { SchemaRobotType } from '../../robot-types';
import { useSynchronizeModelJoints } from '../../use-joint-state';
import { InlineAlert } from '../shared/inline-alert';
import { SetupRobotViewer } from '../shared/setup-robot-viewer';
import { StatusBadge } from '../shared/status-badge';
import { Stepper } from '../shared/stepper';
import { useZeroCalibrationWebSocket, ZeroCalibrationResult } from './use-zero-calibration-websocket';

import classes from '../shared/setup-wizard.module.css';

type ZeroCalibrationStep = 'position' | 'verify';

const STEPS: ZeroCalibrationStep[] = ['position', 'verify'];
const STEP_LABELS: Record<ZeroCalibrationStep, string> = { position: 'Zero pose', verify: 'Verify' };

const jointLabel = (key: string) => (key.endsWith('.pos') ? key.slice(0, -4) : key);

// ---------------------------------------------------------------------------
// Right column: live 3D view
// ---------------------------------------------------------------------------

const LiveViewer = ({ robotType, joints }: { robotType: SchemaRobotType; joints: Record<string, number> | null }) => {
    const jointList = useMemo(() => Object.entries(joints ?? {}).map(([name, value]) => ({ name, value })), [joints]);
    useSynchronizeModelJoints(jointList, robotType);

    return (
        <View
            height='100%'
            backgroundColor='gray-200'
            UNSAFE_style={{ borderRadius: 'var(--spectrum-alias-border-radius-regular)', overflow: 'hidden' }}
        >
            <SetupRobotViewer robotType={robotType} />
        </View>
    );
};

// ---------------------------------------------------------------------------
// Joint readings after set-zero
// ---------------------------------------------------------------------------

const JointReadings = ({ result }: { result: ZeroCalibrationResult }) => (
    <div className={classes.sectionCard}>
        <table className={classes.rangeTable}>
            <thead>
                <tr>
                    <th>Joint</th>
                    <th>Reading</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {Object.entries(result.joints).map(([key, value]) => {
                    const isZero = Math.abs(value) <= result.tolerance_deg;
                    return (
                        <tr key={key}>
                            <td>{jointLabel(key)}</td>
                            <td>{value.toFixed(1)}°</td>
                            <td>
                                <StatusBadge variant={isZero ? 'ok' : 'error'}>
                                    {isZero ? 'Zero' : 'Off zero'}
                                </StatusBadge>
                            </td>
                        </tr>
                    );
                })}
            </tbody>
        </table>
    </div>
);

// ---------------------------------------------------------------------------
// Main wizard content
// ---------------------------------------------------------------------------

/**
 * Zero-pose calibration wizard for plugin robots whose catalog entry offers
 * calibration. Two-column layout like the SO101 setup wizard:
 * left — instructions, "Set zero", and the verification result;
 * right — the robot's 3D model, live-synced to the arm.
 *
 * The robot is only created once calibration succeeds.
 */
interface ZeroCalibrationWizardContentProps {
    /** Heading for the zero-pose step; robot-specific sections pass their own. */
    title?: string;
    /** Robot-specific guidance shown above the plugin's instructions. */
    tips?: ReactNode;
}

export const ZeroCalibrationWizardContent = ({
    title = 'Move the arm to its zero pose',
    tips,
}: ZeroCalibrationWizardContentProps) => {
    const navigate = useNavigate();
    const { project_id } = useProjectId();
    const { activeType, robotForm } = useRobotForm();

    const [robotId] = useState(() => uuidv4());
    // Snapshot the form once: the calibration session and the saved robot must use the same settings.
    const [robotBody] = useState(() =>
        activeType === undefined ? null : buildRobotBody(robotForm, activeType, robotId)
    );

    const { state, commands } = useZeroCalibrationWebSocket({ projectId: project_id, robot: robotBody });
    const { phase, statusMessage, joints, calibrationResult, isSettingZero, error, isConnected } = state;

    const [saving, setSaving] = useState(false);
    const [saveError, setSaveError] = useState<string | null>(null);

    const addRobotMutation = $api.useMutation('post', '/api/projects/{project_id}/robots', {
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/robots', { params: { path: { project_id } } }],
                ['get', '/api/projects/{project_id}/robots/online', { params: { path: { project_id } } }],
            ],
        },
    });

    const isCalibrated = calibrationResult?.success === true;
    const currentStep: ZeroCalibrationStep = isCalibrated ? 'verify' : 'position';
    const completedSteps = new Set<ZeroCalibrationStep>(isCalibrated ? ['position'] : []);

    const goBackToForm = () => navigate(paths.project.robots.new({ project_id }));

    const handleSave = async () => {
        if (robotBody === null) {
            return;
        }

        setSaving(true);
        setSaveError(null);
        try {
            const createdRobot = await addRobotMutation.mutateAsync({
                params: { path: { project_id } },
                body: robotBody,
            });
            navigate(paths.project.robots.show({ project_id, robot_id: createdRobot.id }));
        } catch (err) {
            setSaveError(err instanceof Error ? err.message : 'Failed to save robot');
        } finally {
            setSaving(false);
        }
    };

    if (robotBody === null || activeType === undefined) {
        return (
            <Flex direction='column' gap='size-200' maxWidth='size-6000'>
                <InlineAlert variant='warning'>Fill in the robot form before calibrating.</InlineAlert>
                <Button variant='secondary' onPress={goBackToForm} alignSelf='start'>
                    Back
                </Button>
            </Flex>
        );
    }

    const isReady = phase === 'positioning' || phase === 'verification';

    return (
        <Grid
            areas={['stepper stepper', 'form viewer']}
            columns={['size-6000', minmax(0, '1fr')]}
            rows={['auto', minmax(0, '1fr')]}
            gap='size-400'
            height='100%'
            minHeight={0}
            UNSAFE_className={classes.wizardGrid}
        >
            <View gridArea='stepper'>
                <Stepper
                    steps={STEPS}
                    currentStep={currentStep}
                    completedSteps={completedSteps}
                    labels={STEP_LABELS}
                    onGoToStep={() => undefined}
                />
                <Divider orientation='horizontal' size='S' marginTop='size-200' />
            </View>

            <View
                gridArea='form'
                UNSAFE_style={{ overflowY: 'auto' }}
                paddingBottom='size-400'
                minHeight={0}
                minWidth={0}
            >
                <Flex direction='column' gap='size-300'>
                    {currentStep === 'position' && (
                        <>
                            <Heading level={3}>{title}</Heading>
                            {tips}
                            {isReady ? (
                                <InlineAlert variant='info'>{statusMessage}</InlineAlert>
                            ) : (
                                !error && (
                                    <Flex alignItems='center' gap='size-150'>
                                        <Loading mode='inline' size='S' />
                                        <Text>Connecting to the robot...</Text>
                                    </Flex>
                                )
                            )}
                            {calibrationResult !== null && !calibrationResult.success && (
                                <>
                                    <InlineAlert variant='warning'>
                                        Some joints do not read zero. Adjust the arm and set zero again.
                                    </InlineAlert>
                                    <JointReadings result={calibrationResult} />
                                </>
                            )}
                        </>
                    )}

                    {currentStep === 'verify' && calibrationResult !== null && (
                        <>
                            <InlineAlert variant='success'>
                                Zero position set. Move the arm to check the 3D view follows it, then save.
                            </InlineAlert>
                            <JointReadings result={calibrationResult} />
                        </>
                    )}

                    {!isConnected && isReady && (
                        <InlineAlert variant='warning'>
                            WebSocket disconnected — 3D preview is not updating.
                        </InlineAlert>
                    )}
                    {(error || saveError) && <InlineAlert variant='error'>{saveError ?? error}</InlineAlert>}

                    <Flex gap='size-200' justifyContent='space-between'>
                        <Button variant='secondary' onPress={goBackToForm}>
                            Back
                        </Button>
                        {currentStep === 'position' ? (
                            <Flex gap='size-200'>
                                {/* The zero lives on the motors, so an arm calibrated before needs no new zero. */}
                                <Button variant='secondary' isPending={saving} onPress={handleSave}>
                                    Skip calibration
                                </Button>
                                <Button
                                    variant='accent'
                                    isPending={isSettingZero}
                                    isDisabled={!isReady || !isConnected}
                                    onPress={commands.setZero}
                                >
                                    Set zero
                                </Button>
                            </Flex>
                        ) : (
                            <Button variant='accent' isPending={saving} onPress={handleSave}>
                                Save Robot
                            </Button>
                        )}
                    </Flex>
                </Flex>
            </View>

            <View gridArea='viewer' minHeight={0} minWidth={0} overflow='hidden'>
                <LiveViewer robotType={activeType as SchemaRobotType} joints={joints} />
            </View>
        </Grid>
    );
};
