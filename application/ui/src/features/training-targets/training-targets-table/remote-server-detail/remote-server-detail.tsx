import { Button, Flex, Heading, StatusLight, Text, View } from '@geti-ui/ui';

import { SchemaPreflightResult, SchemaRemoteServer, SchemaRemoteServerStatus } from '../../../../api/openapi-spec';
import {
    checkLabel,
    checksForTier,
    checkStateForCheck,
    checkStatusLabel,
    remoteServerStatusLabel,
    remoteServerStatusVariant,
} from '../../remote-server-status-utils';
import { HealthCheckRow } from '../remote-trainer-detail/remote-trainer-detail';

import detailClasses from '../remote-trainer-detail/remote-trainer-detail.module.css';
import classes from './remote-server-detail.module.css';

type RemoteServerDetailProps = {
    remoteServer: SchemaRemoteServer;
    status?: SchemaRemoteServerStatus;
    isChecking: boolean;
    tier2Result?: SchemaPreflightResult;
    tier2CheckedAt?: string;
    isRunningTier2: boolean;
    onTestConnection: () => void;
};

export const RemoteServerDetail = ({
    remoteServer,
    status,
    isChecking,
    tier2Result,
    tier2CheckedAt,
    isRunningTier2,
    onTestConnection,
}: RemoteServerDetailProps) => {
    const tier1Checks = status ? checksForTier(status.checks, 1) : [];
    const tier2Checks = tier2Result ? checksForTier(tier2Result.checks, 2) : [];
    const rollupVariant = remoteServerStatusVariant(status, isChecking);
    const rollupLabel = remoteServerStatusLabel(status, isChecking);

    return (
        <View backgroundColor={'gray-75'} padding={'size-300'} borderColor={'gray-300'} borderWidth={'thin'}>
            <Flex gap={'size-300'} wrap>
                <View UNSAFE_className={detailClasses.detailSection}>
                    <Flex justifyContent={'space-between'} alignItems={'center'}>
                        <Heading level={3} UNSAFE_className={detailClasses.sectionHeading}>
                            Health &amp; preflight — Tier 1
                        </Heading>
                        <StatusLight variant={rollupVariant}>{rollupLabel}</StatusLight>
                    </Flex>
                    <div className={detailClasses.checkList}>
                        {tier1Checks.map((check) => (
                            <HealthCheckRow
                                key={check.key}
                                label={checkLabel[check.key] ?? check.key}
                                detail={check.detail ?? (check.method ? `via ${check.method}` : '')}
                                state={checkStateForCheck(check)}
                                status={checkStatusLabel(check.outcome)}
                            />
                        ))}
                        {tier1Checks.length === 0 && (
                            <Text UNSAFE_className={`${detailClasses.checkDetail} ${classes.emptyCheckList}`}>
                                Not checked yet.
                            </Text>
                        )}
                    </div>
                </View>

                <View UNSAFE_className={detailClasses.detailSection}>
                    <Flex justifyContent={'space-between'} alignItems={'center'}>
                        <Heading level={3} UNSAFE_className={detailClasses.sectionHeading}>
                            Deep verification — Tier 2
                        </Heading>
                        <Button variant='secondary' onPress={onTestConnection} isPending={isRunningTier2}>
                            Test connection
                        </Button>
                    </Flex>
                    <div className={detailClasses.checkList}>
                        {tier2Checks.map((check) => (
                            <HealthCheckRow
                                key={check.key}
                                label={checkLabel[check.key] ?? check.key}
                                detail={check.detail ?? (check.method ? `via ${check.method}` : '')}
                                state={checkStateForCheck(check)}
                                status={checkStatusLabel(check.outcome)}
                            />
                        ))}
                        {tier2Checks.length === 0 && (
                            <Text UNSAFE_className={`${detailClasses.checkDetail} ${classes.emptyCheckList}`}>
                                Not verified yet. Test connection pulls the trainer image and probes the device. It
                                never runs automatically.
                            </Text>
                        )}
                    </div>
                    {tier2CheckedAt !== undefined && (
                        <Text UNSAFE_className={detailClasses.checkDetail}>
                            Last verified {new Date(tier2CheckedAt).toLocaleString()}
                        </Text>
                    )}
                </View>

                <View UNSAFE_className={detailClasses.detailSection}>
                    <Heading level={3} UNSAFE_className={detailClasses.sectionHeading}>
                        Connection
                    </Heading>
                    <dl className={detailClasses.definitionList}>
                        <dt>Connection type</dt>
                        <dd>SSH provisioned</dd>
                        <dt>SSH host alias</dt>
                        <dd>{remoteServer.ssh_host_alias}</dd>
                        <dt>Device type</dt>
                        <dd>{remoteServer.device_type.toUpperCase()}</dd>
                        <dt>Added</dt>
                        <dd>
                            {remoteServer.created_at ? new Date(remoteServer.created_at).toLocaleString() : 'Unknown'}
                        </dd>
                        <dt>Last checked</dt>
                        <dd>
                            {remoteServer.last_check_at
                                ? new Date(remoteServer.last_check_at).toLocaleString()
                                : 'Not checked'}
                        </dd>
                    </dl>
                </View>
            </Flex>
        </View>
    );
};
