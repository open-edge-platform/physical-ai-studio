import { useMemo } from 'react';

import { Flex, StatusLight } from '@geti-ui/ui';

import { SchemaRemoteTrainerHealth } from '../../../api/openapi-spec';
import { formatBytes } from './policies';
import { TrainingTargetKind } from './train-model-dialog';
import { pickBestDevice, useBestTrainingDevice } from './use-training-devices';

interface TrainingDeviceInfoProps {
    targetKind: TrainingTargetKind;
    remoteHealth: SchemaRemoteTrainerHealth | null;
    isCheckingRemote: boolean;
}

export const TrainingDeviceInfo = ({ targetKind, remoteHealth, isCheckingRemote }: TrainingDeviceInfoProps) => {
    const bestDevice = useBestTrainingDevice();
    const bestRemoteDevice = useMemo(() => pickBestDevice(remoteHealth?.devices ?? []), [remoteHealth]);

    return (
        <Flex UNSAFE_style={{ textAlign: 'right' }} direction='column' gap='size-75'>
            {targetKind === 'trainer' ? (
                remoteHealth?.status === 'unreachable' ? (
                    <StatusLight variant='negative'>Remote trainer unavailable</StatusLight>
                ) : remoteHealth?.status === 'starting' ? (
                    <StatusLight variant='neutral'>Starting trainer container…</StatusLight>
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
