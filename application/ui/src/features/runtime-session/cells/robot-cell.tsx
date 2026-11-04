import { useCallback, useEffect, useRef } from 'react';

import { Flex, ProgressCircle, Switch, toast, View } from '@geti-ui/ui';
import { useMutation, useQueryClient } from '@tanstack/react-query';

import { $api } from '../../../api/client';
import { getRobotConnectionErrorTitle } from '../../../api/errors';
import { SchemaEnvironmentDraft, SchemaErrorMessage, SchemaTeleopControlConfig } from '../../../api/openapi-spec';
import { useProjectId } from '../../projects/use-project';
import { RobotViewer, UnavailableRobotViewer } from '../../robots/controller/robot-viewer';
import { RobotModelsProvider } from '../../robots/robot-models-context';
import { AvailableSchemaRobot, isUnavailableRobot } from '../../robots/robot-types';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';
import { useSynchronizeModelJoints } from '../../robots/use-synchronize-model-joints';
import { useRobotSimulationLaunch } from '../../simulation/robot-simulation';
import { SimulationViewer } from '../../simulation/simulation-viewer';
import { useRobotSimulation } from '../../simulation/use-robot-simulation';
import { ERROR_TOAST, failureOf, useRuntimeSession, useSessionJoints } from '../runtime-session-provider';

/**
 * Teleoperation as the UI starts it. The leader is read at 100 Hz, each reading one command, and on every
 * start the follower takes 3 s to reach the leader instead of jumping to it.
 */
export const TELEOP_CONTROL = { type: 'teleop', hz: 100, grace_s: 2 } as const satisfies SchemaTeleopControlConfig;

/** Drive the follower from the leader, or hold it where it is. */
export const TeleopSwitch = () => {
    const { state, statuses, send } = useRuntimeSession();
    const { mutate: setTeleop, isPending } = useMutation({
        mutationFn: async (teleoperate: boolean) => {
            const ack = await send({
                command: 'set_control',
                control: teleoperate ? TELEOP_CONTROL : null,
            });
            if (!ack.ok) {
                toast.negative(ack.error?.message ?? 'Could not switch teleoperation.', ERROR_TOAST);
            }
        },
    });

    return (
        <Switch
            isEmphasized
            isSelected={state?.control?.type === 'teleop'}
            // A failed environment stays loaded until it is loaded again, with nothing to drive;
            // switching the control mid-episode discards the episode.
            isDisabled={isPending || failureOf(statuses, 'environment') !== null || state?.dataset?.is_recording}
            onChange={setTeleop}
        >
            Teleoperate
        </Switch>
    );
};

const LiveRobot = ({ robot }: { robot: AvailableSchemaRobot }) => {
    const joints = useSessionJoints();
    useSynchronizeModelJoints(joints, robot);

    return <RobotViewer robot={robot} />;
};

const ConnectionError = ({ error }: { error: SchemaErrorMessage }) => (
    <Flex width='100%' height='100%' justifyContent='center' alignItems='center'>
        <InlineAlert variant='error'>
            <strong>{getRobotConnectionErrorTitle(error.error_code)}</strong>
            <br />
            {error.message}
        </InlineAlert>
    </Flex>
);

/**
 * Load the environment that is loaded now again. A new load resolves it again, which points a
 * leader at the simulation's port; a reload would reconnect to the addresses it was loaded with.
 * The workspace may have swapped the page's environment for a saved one, so a saved environment is
 * loaded by its id; an unsaved one is the page's own, which ``retry`` loads. Nothing is sent if,
 * by the time the environment is fetched, another one is loaded or it no longer failed.
 */
const useLoadEnvironmentAgain = (isFailed: boolean) => {
    const { project_id } = useProjectId();
    const queryClient = useQueryClient();
    const { state, send, retry } = useRuntimeSession();
    const environment_id = state?.environment?.environment_id ?? null;
    const latest = useRef({ environment_id, isFailed, isMounted: true });
    latest.current = { ...latest.current, environment_id, isFailed };
    useEffect(() => {
        latest.current.isMounted = true;
        return () => {
            latest.current.isMounted = false;
        };
    }, []);

    return useCallback(async () => {
        if (environment_id === null) {
            retry();
            return;
        }
        const environments = await queryClient.fetchQuery(
            $api.queryOptions('get', '/api/projects/{project_id}/environments', {
                params: { path: { project_id } },
            })
        );
        const now = latest.current;
        const saved = environments.find((candidate) => candidate.id === environment_id);
        if (saved === undefined || !now.isMounted || now.environment_id !== environment_id || !now.isFailed) {
            return;
        }
        const draft: SchemaEnvironmentDraft = {
            id: saved.id,
            name: saved.name,
            robots: saved.robots ?? [],
            cameras: saved.cameras ?? [],
        };
        await send({ command: 'load_environment', environment: draft, unload_incompatible: false });
    }, [environment_id, project_id, queryClient, retry, send]);
};

/**
 * When each robot's environment was last loaded again, outside the cell, which a load in progress may
 * unmount and mount again. Simulations have no run id, so instead of once per run it is once while it
 * runs, and at most every ``RECOVER_AGAIN_MS`` should the cell miss the simulation stopping.
 */
const recoveredAt = new Map<string, number>();
const RECOVER_AGAIN_MS = 30_000;

/**
 * Load the environment again once the follower's simulation runs, if its follower or leader failed: a
 * session loaded before the simulation started has nothing to connect to, and the simulation may have
 * started on another port. After the load in flight settles, and not again until the simulation stops
 * or the environment comes up, so a failure for another reason is not retried in a loop.
 */
const useRetryWhenSimulationRuns = (robot: AvailableSchemaRobot) => {
    const { project_id } = useProjectId();
    const { statuses, isLoading } = useRuntimeSession();
    const { data: simulation } = useRobotSimulation(project_id, robot.id);
    const isRunning = simulation?.running === true;
    const failed = failureOf(statuses, 'environment') !== null || failureOf(statuses, 'environment/leader') !== null;
    // Up, not merely not failed: the load again passes through connecting on its way.
    const isUp = statuses.environment?.state === 'running' && !failed;
    const loadAgain = useLoadEnvironmentAgain(failed);

    useEffect(() => {
        if (!isRunning || isUp) {
            recoveredAt.delete(robot.id);
            return;
        }
        const last = recoveredAt.get(robot.id);
        if (failed && !isLoading && (last === undefined || Date.now() - last > RECOVER_AGAIN_MS)) {
            recoveredAt.set(robot.id, Date.now());
            void loadAgain();
        }
    }, [isRunning, isUp, failed, isLoading, loadAgain, robot.id]);

    return isRunning;
};

/** Forget when environments were loaded again; for tests. */
export const resetRecoveredSimulationRuns = () => recoveredAt.clear();

/** A simulated follower shows the simulation's own viewer, which offers to start it while it does not run. */
const SimulatedRobot = ({ robot, showTeleop }: { robot: AvailableSchemaRobot; showTeleop: boolean }) => {
    const { state, error, isLoading } = useRuntimeSession();
    const isRunning = useRetryWhenSimulationRuns(robot);

    if (isRunning && error !== null && !isLoading) {
        return <ConnectionError error={error} />;
    }

    const hasLeader = state?.environment?.robot_id === robot.id && state.environment.leader_id !== null;

    return <SimulationViewer robot={robot} actions={showTeleop && hasLeader ? <TeleopSwitch /> : undefined} />;
};

/** The session's follower. ``showTeleop`` puts the Teleoperate switch on it, for pages without a side bar. */
export const RobotCell = ({ follower_id, showTeleop = true }: { follower_id: string; showTeleop?: boolean }) => {
    const { project_id } = useProjectId();
    const { state, error, isLoading } = useRuntimeSession();

    const { data: robot } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots/{robot_id}', {
        params: { path: { project_id, robot_id: follower_id } },
    });
    const isSimulated = useRobotSimulationLaunch(robot.type) !== null;
    if (isUnavailableRobot(robot)) {
        return <UnavailableRobotViewer robotType={robot.type} />;
    }

    if (isSimulated) {
        return <SimulatedRobot robot={robot} showTeleop={showTeleop} />;
    }

    if (error !== null && !isLoading) {
        return <ConnectionError error={error} />;
    }

    if (state?.environment?.robot_id !== follower_id) {
        return (
            <Flex width='100%' height='100%' justifyContent='center' alignItems='center'>
                <ProgressCircle isIndeterminate aria-label='Connecting robot' />
            </Flex>
        );
    }

    return (
        <RobotModelsProvider>
            <View
                minWidth='size-4000'
                minHeight='size-4000'
                width='100%'
                height='100%'
                backgroundColor={'gray-600'}
                position={'relative'}
            >
                <LiveRobot robot={robot} />
                {showTeleop && state.environment.leader_id !== null && (
                    <View position={'absolute'} right={0} top={0} padding='size-100'>
                        <TeleopSwitch />
                    </View>
                )}
            </View>
        </RobotModelsProvider>
    );
};
