import { useState } from 'react';

import { NumberField } from '@geti-ui/ui';

import { SchemaSettingsUpdate, SchemaSshProvisioningSettings } from '../../../api/openapi-spec';
import { SettingsSection } from './settings-section';
import { useSettingsPatch } from './use-settings-patch';

type SshSettingsFormProps = { ssh: SchemaSshProvisioningSettings };

const BYTES_PER_GIB = 1024 ** 3;

export const SshSettingsForm = ({ ssh }: SshSettingsFormProps) => {
    const patchMutation = useSettingsPatch();

    const [connectTimeoutS, setConnectTimeoutS] = useState(ssh.connect_timeout_s);
    const [commandTimeoutS, setCommandTimeoutS] = useState(ssh.command_timeout_s);
    const [preflightTimeoutS, setPreflightTimeoutS] = useState(ssh.preflight_timeout_s);
    const [imagePullTimeoutS, setImagePullTimeoutS] = useState(ssh.image_pull_timeout_s);
    const [readinessTimeoutS, setReadinessTimeoutS] = useState(ssh.readiness_timeout_s);
    const [gpuWaitGiveupS, setGpuWaitGiveupS] = useState(ssh.gpu_wait_giveup_s);
    const [minFreeDiskGib, setMinFreeDiskGib] = useState(ssh.min_free_disk_bytes / BYTES_PER_GIB);
    const [dirty, setDirty] = useState(false);
    const [saved, setSaved] = useState(false);

    const update = <T,>(setValue: (value: T) => void, value: T) => {
        if (typeof value === 'number' && Number.isNaN(value)) {
            return;
        }
        setValue(value);
        setDirty(true);
        setSaved(false);
    };

    const save = () => {
        const body: SchemaSettingsUpdate = {
            ssh: {
                connect_timeout_s: connectTimeoutS,
                command_timeout_s: commandTimeoutS,
                preflight_timeout_s: preflightTimeoutS,
                image_pull_timeout_s: imagePullTimeoutS,
                readiness_timeout_s: readinessTimeoutS,
                gpu_wait_giveup_s: gpuWaitGiveupS,
                min_free_disk_bytes: Math.round(minFreeDiskGib * BYTES_PER_GIB),
            },
        };
        patchMutation.mutate(
            { body },
            {
                onSuccess: () => {
                    setDirty(false);
                    setSaved(true);
                },
            }
        );
    };

    return (
        <SettingsSection
            title='SSH-Provisioned Training'
            description='Connect to a remote GPU server over SSH and run training jobs on it.'
            isDirty={dirty}
            isPending={patchMutation.isPending}
            saved={saved}
            error={patchMutation.error}
            onSave={save}
        >
            <NumberField
                label='Connect timeout (s)'
                value={connectTimeoutS}
                onChange={(value) => update(setConnectTimeoutS, value)}
                minValue={0.1}
                width='100%'
            />
            <NumberField
                label='Command timeout (s)'
                value={commandTimeoutS}
                onChange={(value) => update(setCommandTimeoutS, value)}
                minValue={0.1}
                width='100%'
            />
            <NumberField
                label='Preflight timeout (s)'
                value={preflightTimeoutS}
                onChange={(value) => update(setPreflightTimeoutS, value)}
                minValue={0.1}
                width='100%'
            />
            <NumberField
                label='Image pull timeout (s)'
                value={imagePullTimeoutS}
                onChange={(value) => update(setImagePullTimeoutS, value)}
                minValue={0.1}
                width='100%'
            />
            <NumberField
                label='Container readiness timeout (s)'
                value={readinessTimeoutS}
                onChange={(value) => update(setReadinessTimeoutS, value)}
                minValue={0.1}
                width='100%'
            />
            <NumberField
                label='GPU wait give-up budget (s)'
                value={gpuWaitGiveupS}
                onChange={(value) => update(setGpuWaitGiveupS, value)}
                minValue={0.1}
                width='100%'
            />
            <NumberField
                label='Minimum free disk space (GiB)'
                value={minFreeDiskGib}
                onChange={(value) => update(setMinFreeDiskGib, value)}
                minValue={0}
                width='100%'
            />
        </SettingsSection>
    );
};
