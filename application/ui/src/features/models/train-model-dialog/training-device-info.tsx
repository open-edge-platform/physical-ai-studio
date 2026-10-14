import { Flex, StatusLight } from '@geti-ui/ui';

import { SchemaDeviceInfo, SchemaRemoteTrainerHealth } from '../../../api/openapi-spec';
import { formatBytes } from './policies';
import { TrainingTargetKind } from './train-model-dialog';

interface TrainingDeviceInfoProps {
    targetKind: TrainingTargetKind;
    remoteHealth: SchemaRemoteTrainerHealth | null;
    localDevice: SchemaDeviceInfo | null;
    remoteDevice: SchemaDeviceInfo | null;
    isRemoteDeviceBusy: boolean;
    isCheckingRemote: boolean;
}

export const TrainingDeviceInfo = ({
    targetKind,
    remoteHealth,
    localDevice,
    remoteDevice,
    isRemoteDeviceBusy,
    isCheckingRemote,
}: TrainingDeviceInfoProps) => {
    return (
        <Flex UNSAFE_style={{ textAlign: 'right' }} direction='column' gap='size-75'>
            {targetKind === 'trainer' ? (
                remoteHealth?.status === 'unreachable' ? (
                    <StatusLight variant='negative'>Remote trainer unavailable</StatusLight>
                ) : remoteHealth?.status === 'starting' ? (
                    <StatusLight variant='neutral'>Starting trainer container…</StatusLight>
                ) : remoteDevice ? (
                    <StatusLight variant={isRemoteDeviceBusy ? 'yellow' : 'positive'}>
                        {remoteDevice.name}, {formatBytes(remoteDevice.memory!)} VRAM
                    </StatusLight>
                ) : isCheckingRemote && remoteHealth === null ? (
                    <StatusLight variant='neutral'>Checking remote trainer…</StatusLight>
                ) : (
                    <StatusLight variant='neutral'>Remote trainer selected</StatusLight>
                )
            ) : localDevice ? (
                <StatusLight variant='positive'>
                    {localDevice.name}, {formatBytes(localDevice.memory!)} VRAM
                </StatusLight>
            ) : (
                <StatusLight variant='neutral'>CPU only (no GPU detected)</StatusLight>
            )}
        </Flex>
    );
};
