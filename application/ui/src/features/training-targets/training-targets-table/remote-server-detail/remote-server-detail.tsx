import { useState } from 'react';

import { ActionButton, Badge, Button, Heading, StatusLight, Text, Tooltip, TooltipTrigger, View } from '@geti-ui/ui';

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

    const tier1Passing = tier1Checks.filter((check) => checkStateForCheck(check) === 'positive').length;
    const tier1AllPassing = tier1Checks.length > 0 && tier1Passing === tier1Checks.length;
    // An all-green list carries no actionable information, so it starts collapsed.
    const [showTier1Checks, setShowTier1Checks] = useState(false);
    const tier1Expanded = showTier1Checks || !tier1AllPassing;

    // Pulling/verifying the image requires an SSH connection to the host, so the
    // action is only meaningful once the "reachable" tier 1 check has passed.
    const reachableCheck = tier1Checks.find((check) => check.key === 'reachable');
    const isHostUnreachable = reachableCheck !== undefined && checkStateForCheck(reachableCheck) !== 'positive';
    const isTestConnectionDisabled = isHostUnreachable || isRunningTier2;

    return (
        <View backgroundColor={'gray-75'} padding={'size-300'} borderColor={'gray-300'} borderWidth={'thin'}>
            <div className={detailClasses.detailGrid}>
                <View UNSAFE_className={detailClasses.detailSection}>
                    <div className={detailClasses.sectionHeader}>
                        <Heading level={3} UNSAFE_className={detailClasses.sectionHeading}>
                            Health &amp; preflight
                        </Heading>
                        <StatusLight variant={rollupVariant}>{rollupLabel}</StatusLight>
                    </div>
                    {tier1AllPassing && (
                        <div className={classes.summaryRow}>
                            <Text UNSAFE_className={detailClasses.checkDetail}>
                                {`All ${tier1Checks.length} preflight checks passed`}
                            </Text>
                            <ActionButton isQuiet onPress={() => setShowTier1Checks((shown) => !shown)}>
                                {showTier1Checks ? 'Hide checks' : 'Show checks'}
                            </ActionButton>
                        </div>
                    )}
                    {tier1Expanded && (
                        <div className={detailClasses.checkList}>
                            {tier1Checks.map((check) => (
                                <HealthCheckRow
                                    key={check.key}
                                    label={checkLabel[check.key] ?? check.key}
                                    detail={check.detail ?? (check.method ? `via ${check.method}` : '')}
                                    state={checkStateForCheck(check)}
                                    status={checkStatusLabel(check)}
                                />
                            ))}
                            {tier1Checks.length === 0 && (
                                <Text UNSAFE_className={`${detailClasses.checkDetail} ${classes.emptyCheckList}`}>
                                    Not checked yet.
                                </Text>
                            )}
                        </div>
                    )}
                </View>

                <View UNSAFE_className={detailClasses.detailSection}>
                    <div className={detailClasses.sectionHeader}>
                        <Heading level={3} UNSAFE_className={detailClasses.sectionHeading}>
                            Image pull &amp; verification
                        </Heading>
                        <TooltipTrigger delay={300}>
                            <Button
                                variant='secondary'
                                onPress={onTestConnection}
                                isPending={isRunningTier2}
                                isDisabled={isTestConnectionDisabled}
                            >
                                Pull &amp; verify image
                            </Button>
                            <Tooltip>
                                {isHostUnreachable
                                    ? 'SSH host is not reachable. Resolve connectivity before pulling ' +
                                      'and verifying the image.'
                                    : 'Pull and verify the trainer image'}
                            </Tooltip>
                        </TooltipTrigger>
                    </div>
                    <div className={detailClasses.checkList}>
                        {tier2Checks.map((check) => (
                            <HealthCheckRow
                                key={check.key}
                                label={checkLabel[check.key] ?? check.key}
                                detail={check.detail ?? (check.method ? `via ${check.method}` : '')}
                                state={checkStateForCheck(check)}
                                status={checkStatusLabel(check)}
                                rowClassName={classes.tier2CheckRow}
                                contentClassName={classes.tier2CheckContent}
                            />
                        ))}
                    </div>
                    {tier2Checks.length === 0 && (
                        <div className={classes.emptyState}>
                            <Text UNSAFE_className={classes.emptyStateTitle}>Not verified yet</Text>
                            <Text UNSAFE_className={`${detailClasses.checkDetail} ${classes.emptyStateBody}`}>
                                Pull &amp; verify image the trainer image.
                            </Text>
                        </div>
                    )}
                    {tier2CheckedAt !== undefined && (
                        <Text UNSAFE_className={`${detailClasses.checkDetail} ${detailClasses.sectionNote}`}>
                            Last verified {new Date(tier2CheckedAt).toLocaleString()}
                        </Text>
                    )}
                </View>
                <View UNSAFE_className={detailClasses.detailSection}>
                    <div className={detailClasses.sectionHeader}>
                        <Heading level={3} UNSAFE_className={detailClasses.sectionHeading}>
                            Connection
                        </Heading>
                    </div>
                    <dl className={detailClasses.definitionList}>
                        <dt>Connection type</dt>
                        <dd>
                            <Badge variant='neutral' UNSAFE_className={detailClasses.connectionTypeBadge}>
                                SSH provisioned
                            </Badge>
                        </dd>
                        <dt>SSH host alias</dt>
                        <dd className={detailClasses.definitionListMono}>{remoteServer.ssh_host_alias}</dd>
                        <dt>Device type</dt>
                        <dd className={detailClasses.definitionListMono}>{remoteServer.device_type.toUpperCase()}</dd>
                        <dt>Added</dt>
                        <dd className={detailClasses.definitionListTabular}>
                            {remoteServer.created_at ? new Date(remoteServer.created_at).toLocaleString() : 'Unknown'}
                        </dd>
                        <dt>Last checked</dt>
                        <dd className={detailClasses.definitionListTabular}>
                            {remoteServer.last_check_at
                                ? new Date(remoteServer.last_check_at).toLocaleString()
                                : 'Not checked'}
                        </dd>
                    </dl>
                </View>
            </div>
        </View>
    );
};
