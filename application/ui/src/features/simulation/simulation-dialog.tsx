// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState, type ReactNode } from 'react';

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogContainer,
    Disclosure,
    DisclosurePanel,
    DisclosureTitle,
    Divider,
    Flex,
    Heading,
    Item,
    NumberField,
    Picker,
    ProgressCircle,
    Radio,
    RadioGroup,
    Text,
    View,
} from '@geti-ui/ui';

import { getApiErrorMessage } from '../../api/errors';
import { useProjectId } from '../projects/use-project';
import { useRobotCatalogQuery } from '../robots/robot-catalog.hooks';
import { InlineAlert } from '../robots/setup-wizard/shared/inline-alert';
import { rememberedScene, rememberScene, RobotSimulationLaunch, useRobotSimulationLaunch } from './robot-simulation';
import {
    isActiveSimulation,
    Simulation,
    useSimulation,
    useSimulationLogs,
    useStartSimulation,
    useStopSimulation,
} from './use-simulation';

import classes from './simulation.module.css';

interface SimulatedRobot {
    id: string;
    name: string;
    type: string;
}

const robotTypeLabel = (type: string | null | undefined) => (type ?? 'robot').replaceAll('_', ' ');

const StartForm = ({
    robot,
    project_id,
    launch,
    simulation,
    close,
    onStarted,
}: {
    robot: SimulatedRobot;
    project_id: string;
    launch: RobotSimulationLaunch;
    simulation: Simulation | undefined;
    close: () => void;
    onStarted: () => void;
}) => {
    const [scene, setScene] = useState(() => rememberedScene(robot.id, launch));
    const [seedMode, setSeedMode] = useState<'random' | 'fixed'>('random');
    const [seed, setSeed] = useState(0);
    const startMutation = useStartSimulation(project_id);

    const other = isActiveSimulation(simulation) && simulation.robot_id !== robot.id ? simulation : undefined;
    const { data: catalog } = useRobotCatalogQuery();
    const displayName = catalog.find(({ type }) => type === robot.type)?.display_name ?? robotTypeLabel(robot.type);
    const sceneDescription = launch.scenes.find(({ id }) => id === scene)?.description;

    const start = () => {
        rememberScene(robot.id, scene);
        startMutation.mutate(
            {
                body: {
                    project_id,
                    robot_id: robot.id,
                    scene,
                    seed: launch.max_seed !== null && seedMode === 'fixed' ? seed : null,
                    // Also this robot's own simulation, e.g. one still stopping after a failed start.
                    replace: isActiveSimulation(simulation),
                },
            },
            { onSuccess: onStarted }
        );
    };

    const startError =
        startMutation.error === null
            ? undefined
            : (getApiErrorMessage(startMutation.error) ?? 'Could not start the simulation.');

    return (
        <Dialog width='size-6000'>
            <Heading>Start simulation</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-200'>
                    <Flex direction='column' gap='size-50'>
                        <Text UNSAFE_className={classes.robotName}>{robot.name}</Text>
                        {/* The type only when the user named the robot otherwise: a default name is the type. */}
                        {robot.name !== displayName ? (
                            <Text UNSAFE_className={classes.detail}>{displayName}</Text>
                        ) : null}
                    </Flex>
                    <Picker
                        label='Scene'
                        width='100%'
                        selectedKey={scene}
                        onSelectionChange={(key) => key !== null && setScene(String(key))}
                        description={sceneDescription}
                    >
                        {launch.scenes.map((option) => (
                            <Item key={option.id}>{option.display_name}</Item>
                        ))}
                    </Picker>
                    {launch.max_seed !== null ? (
                        <Disclosure isQuiet>
                            <DisclosureTitle>Advanced</DisclosureTitle>
                            <DisclosurePanel>
                                <Flex gap='size-200' alignItems='end'>
                                    <RadioGroup
                                        label='Seed'
                                        orientation='horizontal'
                                        value={seedMode}
                                        onChange={(value) => setSeedMode(value === 'fixed' ? 'fixed' : 'random')}
                                    >
                                        <Radio value='random'>Random</Radio>
                                        <Radio value='fixed'>Fixed</Radio>
                                    </RadioGroup>
                                    {seedMode === 'fixed' ? (
                                        <NumberField
                                            aria-label='Seed value'
                                            minValue={0}
                                            maxValue={launch.max_seed}
                                            step={1}
                                            value={seed}
                                            onChange={(value) => setSeed(Number.isNaN(value) ? 0 : value)}
                                        />
                                    ) : null}
                                </Flex>
                            </DisclosurePanel>
                        </Disclosure>
                    ) : null}
                    {other !== undefined ? (
                        <InlineAlert variant='warning'>
                            <strong>Another simulation is running</strong>
                            <br />
                            Studio runs one simulation at a time. {robotTypeLabel(other.robot_type)} · {other.scene} is
                            running; it will be stopped, and so will the sessions that use it.
                        </InlineAlert>
                    ) : null}
                    {startError !== undefined ? <InlineAlert variant='error'>{startError}</InlineAlert> : null}
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Cancel
                </Button>
                <Button variant='accent' onPress={start} isPending={startMutation.isPending}>
                    {other !== undefined ? 'Stop it and start this one' : 'Start simulation'}
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

type StepState = 'done' | 'active' | 'pending' | 'failed';

const STEPS = ['Robot model', 'Physics and viewer', 'Cameras', 'Ready'] as const;

const currentStep = (simulation: Simulation): number => {
    if (simulation.status === 'running') {
        return STEPS.length;
    }
    switch (simulation.phase ?? simulation.error?.phase) {
        case 'connect':
        case 'load':
            return 1;
        case 'cameras':
            return 2;
        default:
            return 0;
    }
};

const formatMegabytes = (bytes: number) => `${(bytes / 1_000_000).toFixed(1)} MB`;

const StepIcon = ({ state }: { state: StepState }) => {
    if (state === 'active') {
        return <ProgressCircle size='S' isIndeterminate aria-label='In progress' />;
    }

    return (
        <span className={classes.stepIcon} aria-hidden>
            {state === 'done' ? '✓' : state === 'failed' ? '✕' : '○'}
        </span>
    );
};

const Step = ({ state, children }: { state: StepState; children: ReactNode }) => (
    <Flex gap='size-100' alignItems='center' UNSAFE_className={classes.step} data-state={state}>
        <StepIcon state={state} />
        {children}
    </Flex>
);

const SimulationProgress = ({
    simulation,
    project_id,
    close,
    onBack,
}: {
    simulation: Simulation;
    project_id: string;
    close: () => void;
    onBack: () => void;
}) => {
    const [showLog, setShowLog] = useState(false);
    const stopMutation = useStopSimulation(project_id);
    const { data: logs } = useSimulationLogs(showLog);

    const step = currentStep(simulation);
    const failed = simulation.status === 'failed';
    const stepState = (index: number): StepState =>
        index < step ? 'done' : index > step ? 'pending' : failed ? 'failed' : 'active';
    const download = simulation.download;

    const heading =
        simulation.status === 'running'
            ? 'Simulation running'
            : failed
              ? 'Simulation failed'
              : simulation.status === 'stopping'
                ? 'Stopping simulation'
                : 'Starting simulation';

    return (
        <Dialog width='size-6000'>
            <Heading>{heading}</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-150'>
                    <Text>
                        {robotTypeLabel(simulation.robot_type)} · {simulation.scene}
                    </Text>
                    {STEPS.map((label, index) => (
                        <Step key={label} state={stepState(index)}>
                            <Text>{label}</Text>
                            {index === 0 && download != null ? (
                                <Text UNSAFE_className={classes.detail}>
                                    downloading {formatMegabytes(download.bytes)}
                                    {download.total != null ? ` of ${formatMegabytes(download.total)}` : ''}
                                </Text>
                            ) : null}
                            {index === STEPS.length - 1 && simulation.owner_name != null ? (
                                <Text UNSAFE_className={classes.detail}>as {simulation.owner_name}</Text>
                            ) : null}
                        </Step>
                    ))}
                    {failed && simulation.error != null ? (
                        <InlineAlert variant='error'>{simulation.error.message}</InlineAlert>
                    ) : null}
                    {simulation.status === 'running' && (simulation.cameras ?? []).length === 0 ? (
                        <InlineAlert variant='warning'>
                            The simulation runs without cameras: rendering is not available on this machine.
                        </InlineAlert>
                    ) : null}
                    {showLog ? (
                        <View UNSAFE_className={classes.log}>
                            <pre>{(logs?.lines ?? []).join('\n') || 'No output yet.'}</pre>
                        </View>
                    ) : null}
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={() => setShowLog((value) => !value)}>
                    {showLog ? 'Hide log' : 'Show log'}
                </Button>
                {simulation.status === 'starting' ? (
                    <Button
                        variant='secondary'
                        onPress={() => stopMutation.mutate({}, { onSuccess: close })}
                        isPending={stopMutation.isPending}
                    >
                        Cancel
                    </Button>
                ) : null}
                {failed ? (
                    <Button variant='secondary' onPress={() => stopMutation.mutate({}, { onSuccess: onBack })}>
                        Back
                    </Button>
                ) : null}
                <Button variant='accent' onPress={close} isDisabled={simulation.status === 'starting'}>
                    Close
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

/**
 * Starts a simulated robot's simulation and follows it until it runs or fails. Opens on the
 * progress view when the robot's simulation is already starting.
 */
export const SimulationDialog = ({
    robot,
    project_id,
    launch,
    close,
    onRunning,
}: {
    robot: SimulatedRobot;
    project_id: string;
    launch: RobotSimulationLaunch;
    close: () => void;
    onRunning?: () => void;
}) => {
    const { data: simulation } = useSimulation();
    const isThisRobot = simulation?.robot_id === robot.id && simulation.status !== 'idle';
    const [showForm, setShowForm] = useState(!isThisRobot);
    // Undefined until the state has loaded, so a simulation already running on open is not news.
    const wasRunning = useRef<boolean | undefined>(undefined);

    useEffect(() => {
        if (isThisRobot) {
            setShowForm(false);
        }
    }, [isThisRobot]);

    useEffect(() => {
        if (simulation === undefined) {
            return;
        }
        const isRunning = isThisRobot && simulation.status === 'running';
        if (isRunning && wasRunning.current === false) {
            onRunning?.();
        }
        wasRunning.current = isRunning;
    }, [isThisRobot, simulation, onRunning]);

    if (!showForm && isThisRobot && simulation !== undefined) {
        return (
            <SimulationProgress
                simulation={simulation}
                project_id={project_id}
                close={close}
                onBack={() => setShowForm(true)}
            />
        );
    }

    return (
        <StartForm
            robot={robot}
            project_id={project_id}
            launch={launch}
            simulation={simulation}
            close={close}
            onStarted={() => setShowForm(false)}
        />
    );
};

/**
 * The start dialog of a robot's simulation, with a function that opens it (undefined when the
 * robot type is not simulated). Render `dialog` where it stays mounted while the simulation starts:
 * views that show "not running" go away once it runs, and the dialog must outlive them.
 */
export const useSimulationDialog = (robot: SimulatedRobot, onRunning?: () => void) => {
    const { project_id } = useProjectId();
    const launch = useRobotSimulationLaunch(robot.type);
    const [isOpen, setIsOpen] = useState(false);

    const dialog = (
        <DialogContainer onDismiss={() => setIsOpen(false)}>
            {isOpen && launch !== null ? (
                <SimulationDialog
                    robot={robot}
                    project_id={project_id}
                    launch={launch}
                    close={() => setIsOpen(false)}
                    onRunning={onRunning}
                />
            ) : null}
        </DialogContainer>
    );

    return { open: launch !== null ? () => setIsOpen(true) : undefined, dialog };
};
