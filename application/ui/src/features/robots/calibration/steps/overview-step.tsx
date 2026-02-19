import { ChangeEvent, RefObject, useRef } from 'react';

import {
    Button,
    ButtonGroup,
    Cell,
    Column,
    Content,
    Flex,
    Heading,
    InlineAlert,
    Item,
    Menu,
    MenuTrigger,
    Row,
    TableBody,
    TableHeader,
    TableView,
    toast,
    View,
} from '@geti/ui';
import { v4 as uuidv4 } from 'uuid';

import { $api } from '../../../../api/client';
import { components } from '../../../../api/openapi-spec';
import { useRobot, useRobotId } from '../../use-robot';

const downloadJson = (data: Object, fileName: string) => {
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    a.click();
    URL.revokeObjectURL(url);
};

const useMotorCalibration = () => {
    const { project_id, robot_id } = useRobotId();
    const robot = useRobot();

    const motorCalibrationQuery = $api.useSuspenseQuery(
        'get',
        '/api/projects/{project_id}/robots/{robot_id}/calibrations/motor',
        {
            params: { path: { project_id, robot_id } },
        }
    );

    return motorCalibrationQuery.data;
};

const useActiveCalibration = () => {
    const { project_id, robot_id } = useRobotId();
    const robot = useRobot();
    return $api.useQuery(
        'get',
        '/api/projects/{project_id}/robots/{robot_id}/calibrations/{calibration_id}',
        {
            params: {
                path: {
                    project_id,
                    robot_id,
                    calibration_id: robot.active_calibration_id ?? '',
                },
            },
        },
        { enabled: robot.active_calibration_id !== null }
    );
};

const CalibrationTable = () => {
    const { project_id, robot_id } = useRobotId();
    const robot = useRobot();

    const motorCalibration = useMotorCalibration();
    const activeCalibrationQuery = useActiveCalibration();
    const joints = Object.keys(motorCalibration);

    return (
        <TableView aria-label='Calibration comparison table' gridArea='table' selectionMode='none'>
            <TableHeader>
                <Column isRowHeader showDivider>
                    Joint
                </Column>
                <Column isRowHeader title='Motor'>
                    <Column>Min</Column>
                    <Column>Offset</Column>
                    <Column showDivider>Max</Column>
                </Column>
                <Column isRowHeader title='Geti Action'>
                    <Column>Min</Column>
                    <Column>Offset</Column>
                    <Column showDivider>Max</Column>
                </Column>
            </TableHeader>
            <TableBody>
                {joints.map((jointName) => {
                    const motor = motorCalibration[jointName] as {
                        range_min: number;
                        range_max: number;
                        homing_offset: number;
                    };
                    const database = activeCalibrationQuery.data?.values[jointName];

                    if (database === undefined) {
                        return (
                            <Row key={jointName}>
                                <Cell>{jointName}</Cell>

                                <Cell>{motor.range_min}</Cell>
                                <Cell>{motor.range_max}</Cell>
                                <Cell>{motor.homing_offset}</Cell>
                                <Cell colSpan={3}>No data</Cell>
                            </Row>
                        );
                    }

                    return (
                        <Row key={jointName}>
                            <Cell>{jointName}</Cell>

                            <Cell>{motor.range_min}</Cell>
                            <Cell>{motor.range_max}</Cell>
                            <Cell>{motor.homing_offset}</Cell>

                            <Cell>{database?.range_min}</Cell>
                            <Cell>{database?.range_max}</Cell>
                            <Cell>{database?.homing_offset}</Cell>
                        </Row>
                    );
                })}
            </TableBody>
        </TableView>
    );
};

const ExportCalibrationMenu = () => {
    const { project_id, robot_id } = useRobotId();
    const robot = useRobot();

    const motorCalibration = useMotorCalibration();
    const activeCalibrationQuery = useActiveCalibration();

    const getExportData = (source: 'from-robot' | 'from-action') => {
        if (source === 'from-robot') {
            return motorCalibration;
        }

        const data = activeCalibrationQuery.data?.values;
        if (data === undefined) {
            return;
        }

        return Object.fromEntries(
            Object.keys(data).map((jointName) => {
                // Remove `joint_name` from the exported data
                const { joint_name: _, ...jointData } = data[jointName];

                return [jointName, jointData];
            })
        );
    };

    const handleExport = (source: 'from-robot' | 'from-action') => {
        const data = getExportData(source);
        if (data === undefined) {
            return;
        }

        downloadJson(data, `geti-action-${robot.name}-calibration-${source}.json`);
    };
    return (
        <MenuTrigger>
            <Button>Export</Button>
            <Menu
                onAction={(key) => handleExport(key as 'from-robot' | 'from-action')}
                disabledKeys={robot.active_calibration_id === null ? ['from-action'] : []}
            >
                <Item key='from-robot'>From robot</Item>
                <Item key='from-action'>From Geti Action</Item>
            </Menu>
        </MenuTrigger>
    );
};

const useImportCalibration = () => {
    const { project_id, robot_id } = useRobotId();
    const submitCalibrationMutation = $api.useMutation(
        'post',
        '/api/projects/{project_id}/robots/{robot_id}/calibrations'
    );

    return async (calibration: { [key: string]: components['schemas']['CalibrationValue'] }) => {
        const calibrationId = uuidv4();
        await submitCalibrationMutation.mutateAsync({
            params: { path: { project_id, robot_id } },
            body: {
                id: calibrationId,
                file_path: '',
                robot_id,
                values: calibration,
            },
        });
    };
};

const getCalibrationValues = (calibration: { [key: string]: components['schemas']['MotorCalibration'] }) => {
    // Build calibration object with proper typing for the API
    return Object.fromEntries(
        Object.entries(calibration).map(([key, value]) => {
            const motorValue = value as {
                id: number;
                drive_mode: number;
                homing_offset: number;
                range_min: number;
                range_max: number;
            };

            return [key, { ...motorValue, joint_name: key }];
        })
    ) satisfies { [key: string]: components['schemas']['CalibrationValue'] };
};

const FileImportInput = ({ fileInputRef }: { fileInputRef: RefObject<HTMLInputElement | null> }) => {
    const importCalibration = useImportCalibration();

    const handleFileImport = async (e: ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];

        if (file === undefined) {
            return;
        }

        try {
            const json = await file.text();
            // TODO: Implement proper file import logic
            // - Validate calibration JSON schema
            const calibration = getCalibrationValues(JSON.parse(json));
            await importCalibration(calibration);
        } catch (error) {
            toast({ message: `Failed to to import calibration file`, type: 'error' });
        }
    };

    return (
        <input ref={fileInputRef} type='file' accept='.json' onChange={handleFileImport} style={{ display: 'none' }} />
    );
};

const ImportCalibrationMenu = () => {
    const { project_id, robot_id } = useRobotId();
    const robot = useRobot();

    const importCalibration = useImportCalibration();
    const motorCalibration = useMotorCalibration();
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleImportFromRobot = async () => {
        const calibration = getCalibrationValues(motorCalibration);

        await importCalibration(calibration);
    };

    return (
        <>
            <FileImportInput fileInputRef={fileInputRef} />
            <MenuTrigger>
                <Button>Import</Button>
                <Menu
                    onAction={async (key) => {
                        if (key === 'from_robot') {
                            await handleImportFromRobot();
                        } else {
                            fileInputRef.current?.click();
                        }
                    }}
                >
                    <Item key='from_robot'>From robot</Item>
                    <Item key='from_file'>From file</Item>
                </Menu>
            </MenuTrigger>
        </>
    );
};

export const OverviewStep = () => {
    const robot = useRobot();

    return (
        <>
            <View gridArea='controls'>
                <Flex direction='column' gap='size-200'>
                    <Flex justifyContent={'end'}>
                        <ButtonGroup>
                            <ImportCalibrationMenu />
                            <ExportCalibrationMenu />
                        </ButtonGroup>
                    </Flex>
                    {robot.active_calibration_id === null && (
                        <InlineAlert variant='notice'>
                            <Heading>Calibration required</Heading>
                            <Content>
                                Calibrate your robot using Geti Action. If you&apos;ve recently calibrated your robot
                                with an external tool then you may import your calibration settings from the robot.
                                Otherwise start the calibration process.
                            </Content>
                        </InlineAlert>
                    )}
                </Flex>
            </View>
            <CalibrationTable />
        </>
    );
};
