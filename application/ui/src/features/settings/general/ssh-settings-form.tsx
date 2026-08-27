import { useState } from 'react';

import { Divider, NumberField, Switch } from '@geti-ui/ui';

import { SchemaSettingsUpdate, SchemaSshProvisioningSettings } from '../../../api/openapi-spec';
import { InlineAlert } from '../../robots/setup-wizard/shared/inline-alert';
import { SettingsSection } from './settings-section';
import { useSettingsPatch } from './use-settings-patch';

type SshSettingsFormProps = { ssh: SchemaSshProvisioningSettings };

const BYTES_PER_GIB = 1024 ** 3;

export const SshSettingsForm = ({ ssh }: SshSettingsFormProps) => {
    const patchMutation = useSettingsPatch();

    const [enabled, setEnabled] = useState(ssh.enabled);
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
                enabled,
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
            title='SSH-provisioned training'
            description='Run training jobs on a GPU server you reach over SSH.'
            isDirty={dirty}
            isPending={patchMutation.isPending}
            saved={saved}
            error={patchMutation.error}
            onSave={save}
        >
            <Switch isEmphasized isSelected={enabled} onChange={(value) => update(setEnabled, value)}>
                Enable SSH-provisioned training
            </Switch>
            <InlineAlert variant='warning'>
                <strong>Security warning:</strong> This feature has no built-in authentication. Anyone who can reach the
                Physical AI Studio backend can run arbitrary code as root on any server you register. Enable it only on
                a single-user, localhost-only workstation with no other network access.
            </InlineAlert>
            <InlineAlert variant='info'>
                Saving this setting restarts the backend. You&apos;ll be reconnected automatically once it&apos;s back
                up, usually within a few seconds.
            </InlineAlert>

            <Divider size='S' marginY='size-100' />

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
