import { Flex, Heading, StatusLight, Text, View } from '@geti-ui/ui';

import { SchemaRemoteTrainer, SchemaRemoteTrainerHealth } from '../../../api/openapi-spec';
import {
    CheckState,
    deviceTypes,
    formatStorage,
    getCapabilityState,
    getStorageState,
    healthDescription,
    healthLabel,
    healthVariant,
} from '../remote-trainer-health-utils';

import classes from '../remote-trainers-page.module.css';

type HealthCheckRowProps = {
    label: string;
    detail: string;
    state: CheckState;
    status: string;
};

export const HealthCheckRow = ({ label, detail, state, status }: HealthCheckRowProps) => (
    <div className={classes.checkRow}>
        <span className={`${classes.checkIcon} ${classes[state]}`} aria-hidden='true'>
            {state === 'positive' ? '✓' : state === 'negative' ? '×' : state === 'yellow' ? '!' : '–'}
        </span>
        <span className={classes.checkContent}>
            <Text UNSAFE_className={classes.checkLabel}>{label}</Text>
            <Text UNSAFE_className={classes.checkDetail}>{detail}</Text>
        </span>
        <StatusLight variant={state}>{status}</StatusLight>
    </div>
);

type RemoteTrainerDetailProps = {
    remoteTrainer: SchemaRemoteTrainer;
    health?: SchemaRemoteTrainerHealth;
    isChecking: boolean;
};

export const RemoteTrainerDetail = ({ remoteTrainer, health, isChecking }: RemoteTrainerDetailProps) => {
    const devices = health?.devices ?? [];
    const types = deviceTypes(health);
    const state = healthVariant(health, isChecking);
    const deviceReportIsInvalid = health?.reason_code === 'invalid_devices_response';
    const trainerHealthState = deviceReportIsInvalid ? 'positive' : state;
    const capabilityState = getCapabilityState(health, isChecking);
    const storageState = getStorageState(health, isChecking);
    const lastChecked = health ? new Date(health.checked_at).toLocaleString() : 'Not checked';

    return (
        <View backgroundColor={'gray-75'} padding={'size-300'} borderColor={'gray-300'} borderWidth={'thin'}>
            <Flex gap={'size-300'}>
                <View UNSAFE_className={classes.detailSection}>
                    <Heading level={3} UNSAFE_className={classes.sectionHeading}>
                        Health &amp; capability
                    </Heading>
                    <div className={classes.checkList}>
                        <HealthCheckRow
                            label='Trainer health endpoint'
                            detail={
                                isChecking
                                    ? 'connection check in progress'
                                    : health?.status === 'healthy' || deviceReportIsInvalid
                                      ? health?.latency_ms != null
                                          ? `responded in ${health.latency_ms} ms and is ready for training requests`
                                          : 'ready for training requests'
                                      : health?.status === 'degraded'
                                        ? 'responded with a degraded status'
                                        : healthDescription(health)
                            }
                            state={trainerHealthState}
                            status={deviceReportIsInvalid ? 'Healthy' : healthLabel(health, isChecking)}
                        />
                        <HealthCheckRow
                            label='Compute capability'
                            detail={
                                devices.length > 0
                                    ? devices
                                          .map((device) => `${device.type.toUpperCase()} · ${device.name}`)
                                          .join(', ')
                                    : health === undefined || isChecking
                                      ? 'awaiting device report'
                                      : 'no compute device reported'
                            }
                            state={capabilityState}
                            status={devices.length > 0 ? 'Available' : 'Unknown'}
                        />
                        <HealthCheckRow
                            label='Storage capacity'
                            detail={
                                formatStorage(health?.storage) ??
                                (health === undefined || isChecking ? 'awaiting storage report' : 'no storage reported')
                            }
                            state={storageState}
                            status={health?.storage ? 'Available' : 'Unknown'}
                        />
                    </div>
                </View>

                <View UNSAFE_className={classes.detailSection}>
                    <Heading level={3} UNSAFE_className={classes.sectionHeading}>
                        Connection
                    </Heading>
                    <dl className={classes.definitionList}>
                        <dt>Connection type</dt>
                        <dd>Direct trainer URL</dd>
                        <dt>Trainer URL</dt>
                        <dd>{remoteTrainer.url}</dd>
                        <dt>Device type</dt>
                        <dd>{types.join(', ') || 'Not reported'}</dd>
                        <dt>Available storage</dt>
                        <dd>{formatStorage(health?.storage) ?? 'Not reported'}</dd>
                        <dt>Health status</dt>
                        <dd>{healthLabel(health, isChecking)}</dd>
                        <dt>Added</dt>
                        <dd>
                            {remoteTrainer.created_at ? new Date(remoteTrainer.created_at).toLocaleString() : 'Unknown'}
                        </dd>
                        <dt>Last checked</dt>
                        <dd>{lastChecked}</dd>
                    </dl>
                </View>
            </Flex>
        </View>
    );
};
