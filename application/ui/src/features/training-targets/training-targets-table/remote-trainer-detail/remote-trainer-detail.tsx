import { Badge, Grid, Heading, StatusLight, Text, View } from '@geti-ui/ui';

import { SchemaRemoteTrainer, SchemaRemoteTrainerHealth } from '../../../../api/openapi-spec';
import {
    capabilityDetail,
    CheckState,
    deviceTypes,
    formatStorage,
    getCapabilityState,
    getStorageState,
    healthLabel,
    healthVariant,
    storageDetail,
    trainerHealthDetail,
} from '../../remote-trainer-health-utils';

import classes from './remote-trainer-detail.module.css';

type HealthCheckRowProps = {
    label: string;
    detail: string;
    state: CheckState;
    status: string;
    rowClassName?: string;
    contentClassName?: string;
};

export const HealthCheckRow = ({
    label,
    detail,
    state,
    status,
    rowClassName,
    contentClassName,
}: HealthCheckRowProps) => (
    <Grid
        columns={'subgrid'}
        gridColumn={'1/-1'}
        alignItems={'center'}
        minHeight={'size-400'}
        UNSAFE_className={`${classes.checkRow} ${rowClassName ?? ''}`}
    >
        <span className={`${classes.checkIcon} ${classes[state]}`} aria-hidden='true'>
            {state === 'positive' ? '✓' : state === 'negative' ? '×' : state === 'yellow' ? '!' : '–'}
        </span>
        <span className={`${classes.checkContent} ${contentClassName ?? ''}`}>
            <Text UNSAFE_className={classes.checkLabel}>{label}</Text>
            <Text UNSAFE_className={classes.checkDetail}>{detail}</Text>
        </span>
        {/*
         * A passing check is already unambiguous from the green tick, so the
         * per-row status label is only rendered when it carries new information.
         * This keeps the eye on the rows that actually need attention.
         */}
        {state === 'positive' ? (
            <span className={classes.visuallyHidden}>{status}</span>
        ) : (
            <StatusLight variant={state} justifySelf={'start'}>
                {status}
            </StatusLight>
        )}
    </Grid>
);

type RemoteTrainerDetailProps = {
    remoteTrainer: SchemaRemoteTrainer;
    health?: SchemaRemoteTrainerHealth;
    isChecking: boolean;
};

export const RemoteTrainerDetail = ({ remoteTrainer, health, isChecking }: RemoteTrainerDetailProps) => {
    const types = deviceTypes(health);
    const state = healthVariant(health, isChecking);
    const deviceReportIsInvalid = health?.reason_code === 'invalid_devices_response';
    const trainerHealthState = deviceReportIsInvalid ? 'positive' : state;
    const capabilityState = getCapabilityState(health, isChecking);
    const storageState = getStorageState(health, isChecking);
    const lastChecked = health ? new Date(health.checked_at).toLocaleString() : 'Not checked';
    const devicesReported = (health?.devices?.length ?? 0) > 0;

    return (
        <View backgroundColor={'gray-75'} padding={'size-300'} borderColor={'gray-300'} borderWidth={'thin'}>
            <div className={`${classes.detailGrid} ${classes.twoColumns}`}>
                <View UNSAFE_className={classes.detailSection}>
                    <div className={classes.sectionHeader}>
                        <Heading level={3} UNSAFE_className={classes.sectionHeading}>
                            Health &amp; capability
                        </Heading>
                        <StatusLight variant={state}>{healthLabel(health, isChecking)}</StatusLight>
                    </div>
                    <div className={classes.checkList}>
                        <HealthCheckRow
                            label='Trainer health endpoint'
                            detail={trainerHealthDetail(health, isChecking, deviceReportIsInvalid)}
                            state={trainerHealthState}
                            status={deviceReportIsInvalid ? 'Healthy' : healthLabel(health, isChecking)}
                        />
                        <HealthCheckRow
                            label='Compute capability'
                            detail={capabilityDetail(health, isChecking)}
                            state={capabilityState}
                            status={devicesReported ? 'Available' : 'Unknown'}
                        />
                        <HealthCheckRow
                            label='Storage capacity'
                            detail={storageDetail(health, isChecking)}
                            state={storageState}
                            status={health?.storage ? 'Available' : 'Unknown'}
                        />
                    </div>
                </View>

                <View UNSAFE_className={classes.detailSection}>
                    <div className={classes.sectionHeader}>
                        <Heading level={3} UNSAFE_className={classes.sectionHeading}>
                            Connection
                        </Heading>
                    </div>
                    <dl className={classes.definitionList}>
                        <dt>Connection type</dt>
                        <dd>
                            <Badge variant='neutral' UNSAFE_className={classes.connectionTypeBadge}>
                                Direct trainer URL
                            </Badge>
                        </dd>
                        <dt>Trainer URL</dt>
                        <dd className={classes.definitionListMono}>{remoteTrainer.url}</dd>
                        <dt>Device type</dt>
                        <dd className={classes.definitionListMono}>{types.join(', ') || 'Not reported'}</dd>
                        <dt>Available storage</dt>
                        <dd>{formatStorage(health?.storage) ?? 'Not reported'}</dd>
                        <dt>Added</dt>
                        <dd className={classes.definitionListTabular}>
                            {remoteTrainer.created_at ? new Date(remoteTrainer.created_at).toLocaleString() : 'Unknown'}
                        </dd>
                        <dt>Last checked</dt>
                        <dd className={classes.definitionListTabular}>{lastChecked}</dd>
                    </dl>
                </View>
            </div>
        </View>
    );
};
