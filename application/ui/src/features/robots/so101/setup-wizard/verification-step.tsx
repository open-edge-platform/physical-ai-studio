import { useEffect, useState } from 'react';

import { Button, Flex, Heading, Text } from '@geti/ui';
import { useNavigate } from 'react-router';
import { degToRad } from 'three/src/math/MathUtils.js';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../../api/client';
import { SchemaCalibration, SchemaRobot } from '../../../../api/openapi-spec';
import { paths } from '../../../../router';
import { useProjectId } from '../../../projects/use-project';
import { useRobotForm } from '../../robot-form/provider';
import { useRobotModels } from '../../robot-models-context';
import { CalibrationResult } from './use-setup-websocket';
import { useSetupActions, useSetupState } from './wizard-provider';

import classes from './setup-wizard.module.scss';

// ---------------------------------------------------------------------------
// Hook: sync normalized joint state from the setup websocket to the URDF model
// ---------------------------------------------------------------------------

/**
 * Syncs joint positions (from `state_was_updated` events) to the loaded URDF
 * model via the shared `RobotModelsProvider` context.
 *
 * Uses the same pattern as `robot-cell.tsx`'s `useSynchronizeModelJoints`.
 * Values are normalized (-100..100 for body, 0..100 for gripper) and treated
 * as degrees by the viewer — `degToRad()` maps them to the URDF's radian range.
 */
const useSyncJointState = (jointState: Record<string, number> | null) => {
    const { models } = useRobotModels();

    useEffect(() => {
        if (!jointState) {
            return;
        }

        models.forEach((model) => {
            for (const [key, value] of Object.entries(jointState)) {
                const name = key.endsWith('.pos') ? key.slice(0, -4) : key;

                if (name === 'gripper' && model.robotName === 'wxai') {
                    model.setJointValue('left_carriage_joint', value);
                    continue;
                }

                if (model.joints[name]) {
                    model.setJointValue(name, degToRad(value));
                }
            }
        });
    }, [models, jointState]);
};

// ---------------------------------------------------------------------------
// Helper: build a Calibration body from the websocket calibration_result event
// ---------------------------------------------------------------------------

function buildCalibrationBody(
    calibrationResult: CalibrationResult,
    robotId: string,
    calibrationId: string
): SchemaCalibration {
    const values: SchemaCalibration['values'] = {};
    for (const [jointName, cal] of Object.entries(calibrationResult.calibration)) {
        values[jointName] = {
            id: cal.id,
            joint_name: jointName,
            drive_mode: cal.drive_mode,
            homing_offset: cal.homing_offset,
            range_min: cal.range_min,
            range_max: cal.range_max,
        };
    }

    return {
        id: calibrationId,
        robot_id: robotId,
        // file_path is set server-side by _save_calibration_to_disk; send a
        // placeholder since the schema requires it.
        file_path: '',
        values,
    };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * Verification step — streams live joint positions from the setup websocket
 * and syncs them to the 3D URDF model so the user can visually verify the
 * robot moves correctly. Shows "Save Robot" button that persists both the
 * robot and its calibration to the DB.
 */
export const VerificationStep = () => {
    const navigate = useNavigate();
    const { project_id } = useProjectId();
    const { goBack } = useSetupActions();
    const { wsState } = useSetupState();
    const robotForm = useRobotForm();

    const serialNumber = robotForm.serial_number ?? '';
    const robotType = robotForm.type ?? '';

    const [robotId] = useState(() => uuidv4());
    const [calibrationId] = useState(() => uuidv4());
    const [saving, setSaving] = useState(false);
    const [saveError, setSaveError] = useState<string | null>(null);

    // Sync live joint positions to the 3D model
    useSyncJointState(wsState.jointState);

    const addRobotMutation = $api.useMutation('post', '/api/projects/{project_id}/robots');
    const saveCalibrationMutation = $api.useMutation(
        'post',
        '/api/projects/{project_id}/robots/{robot_id}/calibrations'
    );
    const updateRobotMutation = $api.useMutation('put', '/api/projects/{project_id}/robots/{robot_id}');

    const robotBody: SchemaRobot | null =
        robotForm.type !== null && robotForm.name
            ? {
                  id: robotId,
                  name: robotForm.name,
                  type: robotForm.type,
                  connection_string: robotForm.connection_string ?? '',
                  serial_number: robotForm.serial_number ?? '',
                  active_calibration_id: null,
              }
            : null;

    const hasCalibration = wsState.calibrationResult !== null;

    const handleSave = async () => {
        if (robotBody === null) {
            return;
        }

        setSaving(true);
        setSaveError(null);

        try {
            // 1. Create the robot
            const createdRobot = await addRobotMutation.mutateAsync({
                params: { path: { project_id } },
                body: robotBody,
            });

            // 2. Save calibration if we have it
            if (wsState.calibrationResult) {
                const calibrationBody = buildCalibrationBody(wsState.calibrationResult, createdRobot.id, calibrationId);

                await saveCalibrationMutation.mutateAsync({
                    params: { path: { project_id, robot_id: createdRobot.id } },
                    body: calibrationBody,
                });

                // 3. Update robot with active_calibration_id
                await updateRobotMutation.mutateAsync({
                    params: { path: { project_id, robot_id: createdRobot.id } },
                    body: {
                        ...createdRobot,
                        active_calibration_id: calibrationId,
                    },
                });
            }

            // Navigate to the robot page
            navigate(paths.project.robots.show({ project_id, robot_id: createdRobot.id }));
        } catch (err) {
            setSaveError(err instanceof Error ? err.message : 'Failed to save robot');
        } finally {
            setSaving(false);
        }
    };

    return (
        <Flex direction='column' gap='size-300'>
            <div className={classes.successBox}>
                <Text>
                    Robot setup is complete. Move the robot arm to verify that the 3D visualization matches the physical
                    robot, then save.
                </Text>
            </div>

            {!wsState.isConnected && (
                <div className={classes.warningBox}>
                    <Text>WebSocket disconnected — 3D preview is not updating.</Text>
                </div>
            )}

            <div className={classes.sectionCard}>
                <Flex direction='column' gap='size-100'>
                    <Heading level={4}>Robot Details</Heading>
                    <Flex direction='column' gap='size-50'>
                        <Text>
                            <strong>Name:</strong> {robotForm.name}
                        </Text>
                        <Text>
                            <strong>Type:</strong> {robotType}
                        </Text>
                        <Text>
                            <strong>Serial:</strong> {serialNumber}
                        </Text>
                        <Text>
                            <strong>Calibration:</strong>{' '}
                            {hasCalibration ? 'Available' : 'Not available — robot may not function correctly'}
                        </Text>
                    </Flex>
                </Flex>
            </div>

            {(wsState.error || saveError) && (
                <div className={classes.warningBox}>
                    <Text>{saveError ?? wsState.error}</Text>
                </div>
            )}

            <Flex gap='size-200' justifyContent='space-between'>
                <Button variant='secondary' onPress={goBack}>
                    Back
                </Button>
                <Button variant='accent' isPending={saving} isDisabled={robotBody === null} onPress={handleSave}>
                    Save Robot
                </Button>
            </Flex>
        </Flex>
    );
};
