import { useMemo } from 'react';

import { Flex, StatusLight } from '@geti-ui/ui';

import { SchemaRemoteServer, SchemaRemoteTrainerHealth } from '../../../api/openapi-spec';
import { formatBytes } from './policies';
import { TrainingTargetKind } from './train-model-dialog';
import { pickBestDevice, useBestTrainingDevice } from './use-training-devices';

interface TrainingDeviceInfoProps {
    targetKind: TrainingTargetKind;
    remoteHealth: SchemaRemoteTrainerHealth | null;
    isCheckingRemote: boolean;
    sshServer: SchemaRemoteServer | null;
    /** Exact GPU/XPU name reported by the SSH server's `driver_present` health check, if known. */
    sshComputeDetail?: string;
}

export const TrainingDeviceInfo = ({
    targetKind,
    remoteHealth,
    isCheckingRemote,
    sshServer,
    sshComputeDetail,
}: TrainingDeviceInfoProps) => {
    const bestDevice = useBestTrainingDevice();
    const bestRemoteDevice = useMemo(() => pickBestDevice(remoteHealth?.devices ?? []), [remoteHealth]);

    if (targetKind === 'ssh') {
        return (
            <Flex UNSAFE_style={{ textAlign: 'right' }} direction='column' gap='size-75'>
                {sshServer === null ? (
                    <StatusLight variant='neutral'>Remote server selected</StatusLight>
                ) : sshServer.last_check_status === 'unknown' ? (
                    <StatusLight variant='notice'>Not verified yet</StatusLight>
                ) : sshServer.last_check_status !== 'healthy' ? (
                    <StatusLight variant='negative'>
                        Remote server not ready ({sshServer.last_check_status})
                    </StatusLight>
                ) : (
                    <StatusLight variant='positive'>
                        {sshComputeDetail ?? sshServer.device_type.toUpperCase()}
                    </StatusLight>
                )}
            </Flex>
        );
    }

    return (
        <Flex UNSAFE_style={{ textAlign: 'right' }} direction='column' gap='size-75'>
            {targetKind === 'trainer' ? (
                remoteHealth?.status === 'unreachable' ? (
                    <StatusLight variant='negative'>Remote trainer unavailable</StatusLight>
                ) : bestRemoteDevice ? (
                    <StatusLight variant='positive'>
                        {bestRemoteDevice.name}, {formatBytes(bestRemoteDevice.memory!)} VRAM
                    </StatusLight>
                ) : isCheckingRemote && remoteHealth === null ? (
                    <StatusLight variant='neutral'>Checking remote trainer…</StatusLight>
                ) : (
                    <StatusLight variant='neutral'>Remote trainer selected</StatusLight>
                )
            ) : bestDevice ? (
                <StatusLight variant='positive'>
                    {bestDevice.name}, {formatBytes(bestDevice.memory!)} VRAM
                </StatusLight>
            ) : (
                <StatusLight variant='neutral'>CPU only (no GPU detected)</StatusLight>
            )}
        </Flex>
    );
};
