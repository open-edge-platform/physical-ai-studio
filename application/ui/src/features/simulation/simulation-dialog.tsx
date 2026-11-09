// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState, type ReactNode } from 'react';

import {
    ActionButton,
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
    Icon,
    Item,
    NumberField,
    Picker,
    ProgressCircle,
    Radio,
    RadioGroup,
    Text,
    toast,
    View,
} from '@geti-ui/ui';
import { Checkmark, Copy } from '@geti-ui/ui/icons';

import { getApiErrorMessage } from '../../api/errors';
import { useProjectId } from '../projects/use-project';
import { useRobotCatalogQuery } from '../robots/robot-catalog.hooks';
import { InlineAlert } from '../robots/setup-wizard/shared/inline-alert';
import { rememberedScene, rememberScene, RobotSimulationLaunch, useRobotSimulationLaunch } from './robot-simulation';
import { useRobotSimulation, useSimulationCommand } from './use-robot-simulation';
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

/** "<robot type> · <scene>" with the catalog's display names, e.g. "MuJoCo SO-101 Follower · Yahtzee". */
const useSimulationTitle = (robotType: string | null | undefined, scene: string | null | undefined): string => {
    const { data: catalog } = useRobotCatalogQuery();
    const entry = catalog.find(({ type }) => type === robotType);
    const robotLabel = entry?.display_name ?? robotTypeLabel(robotType);
    if (scene == null) {
        return robotLabel;
    }

    return `${robotLabel} · ${entry?.simulation?.scenes.find(({ id }) => id === scene)?.display_name ?? scene}`;
};

/** An argument as `shlex.join` quotes it, so the shown command reads like the copied one. */
const shellQuote = (argument: string): string =>
    /^[\w@%+=:,./-]+$/.test(argument) ? argument : `'${argument.replaceAll("'", `'"'"'`)}'`;

/** The port of a URL, or undefined when it has none or is not a URL. */
const urlPort = (url: string | null | undefined): string | undefined => {
    try {
        return url == null ? undefined : new URL(url).port || undefined;
    } catch {
        return undefined;
    }
};

/**
 * The command that starts the chosen scene in a terminal instead, which Studio attaches to as a
 * simulation started outside Studio. Hidden when the robot type's plugin offers none.
 */
const RunItYourself = ({
    project_id,
    robot_id,
    scene,
    seed,
}: {
    project_id: string;
    robot_id: string;
    scene: string;
    seed: number | null;
}) => {
    const { data } = useSimulationCommand(project_id, robot_id, scene, seed);
    const [copied, setCopied] = useState(false);

    if (data === undefined) {
        return null;
    }

    const copy = async () => {
        try {
            await navigator.clipboard.writeText(data.command);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch {
            toast.negative('Could not copy the command.');
        }
    };

    return (
        <Disclosure isQuiet>
            <DisclosureTitle>Run it yourself</DisclosureTitle>
            <DisclosurePanel>
                <Flex direction='column' gap='size-100'>
                    <Text UNSAFE_className={classes.detail}>
                        Start this simulation in a terminal instead. Studio connects to it, scripts can too, and it
                        keeps running when Studio stops.
                    </Text>
                    <Flex gap='size-100' alignItems='start'>
                        <View flex UNSAFE_className={classes.command}>
                            {/* One unbreakable span per argument: lines break only between arguments. */}
                            <pre>
                                {data.argv.map((argument, index) => (
                                    <span key={index}>
                                        {index > 0 ? ' ' : null}
                                        <span className={classes.argument}>{shellQuote(argument)}</span>
                                    </span>
                                ))}
                            </pre>
                        </View>
                        {/* Wide enough for both labels, so copying does not move the command box. */}
                        <ActionButton aria-label='Copy command' onPress={copy} width='size-1600'>
                            <Icon>{copied ? <Checkmark /> : <Copy />}</Icon>
                            <Text>{copied ? 'Copied' : 'Copy'}</Text>
                        </ActionButton>
                    </Flex>
                </Flex>
            </DisclosurePanel>
        </Disclosure>
    );
};

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
    const otherTitle = useSimulationTitle(other?.robot_type, other?.scene);
    const { data: catalog } = useRobotCatalogQuery();
    const displayName = catalog.find(({ type }) => type === robot.type)?.display_name ?? robotTypeLabel(robot.type);
    const sceneDescription = launch.scenes.find(({ id }) => id === scene)?.description;
    const chosenSeed = launch.max_seed !== null && seedMode === 'fixed' ? seed : null;

    const start = () => {
        rememberScene(robot.id, scene);
        startMutation.mutate(
            {
                body: {
                    project_id,
                    robot_id: robot.id,
                    scene,
                    seed: chosenSeed,
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
                    <RunItYourself project_id={project_id} robot_id={robot.id} scene={scene} seed={chosenSeed} />
                    {other !== undefined ? (
                        <InlineAlert variant='warning'>
                            <strong>Another simulation is running</strong>
                            <br />
                            Studio runs one simulation at a time. {otherTitle} is running; it will be stopped, and so
                            will the sessions that use it.
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

    const title = useSimulationTitle(simulation.robot_type, simulation.scene);
    const step = currentStep(simulation);
    const failed = simulation.status === 'failed';
    // The simulation reuses the robot's port while it is free; Studio moved the robot and its
    // cameras to the port it took instead.
    const askedPort = urlPort(simulation.robot_http_url);
    const runningPort = urlPort(simulation.http_url);
    const movedPort =
        simulation.status === 'running' &&
        askedPort !== undefined &&
        runningPort !== undefined &&
        askedPort !== runningPort;
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
                    <Text>{title}</Text>
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
                    {movedPort ? (
                        <InlineAlert variant='info'>
                            Port {askedPort} was in use; the simulation runs on {runningPort}, and the robot and its
                            cameras now use it.
                        </InlineAlert>
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
 * The robot's simulation runs, started outside Studio (e.g. with the "Run it yourself" command
 * while the dialog was open): Studio is connected to it, so there is nothing left to start.
 */
const ExternalSimulationRunning = ({
    robot,
    scene,
    close,
}: {
    robot: SimulatedRobot;
    scene: string | null | undefined;
    close: () => void;
}) => {
    const title = useSimulationTitle(robot.type, scene);

    return (
        <Dialog width='size-6000'>
            <Heading>Simulation running</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-150'>
                    <Text>{title}</Text>
                    <Step state='done'>
                        <Text>Started outside Studio</Text>
                        <Text UNSAFE_className={classes.detail}>Studio is connected to it</Text>
                    </Step>
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='accent' onPress={close}>
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
    // Probed while the form shows: a simulation started in a terminal is only known from probes.
    const { data: robotSimulation } = useRobotSimulation(project_id, robot.id, showForm && !isThisRobot);
    const runsOutsideStudio = robotSimulation?.running === true && robotSimulation.started_by === 'external';
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

    if (showForm && !isThisRobot && runsOutsideStudio) {
        return <ExternalSimulationRunning robot={robot} scene={robotSimulation?.scene} close={close} />;
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
