import { useState } from 'react';

import { NumberField, Text } from '@geti-ui/ui';

import { SchemaSettingsUpdate, SchemaSshProvisioningSettings } from '../../../api/openapi-spec';
import { SettingsSection } from './settings-section';
import { useSettingsPatch } from './use-settings-patch';

type SshSettingsFormProps = { ssh: SchemaSshProvisioningSettings };

export const SshSettingsForm = ({ ssh }: SshSettingsFormProps) => {
    const patchMutation = useSettingsPatch();

    const [connectTimeoutS, setConnectTimeoutS] = useState(ssh.connect_timeout_s);
    const [commandTimeoutS, setCommandTimeoutS] = useState(ssh.command_timeout_s);
    const [preflightTimeoutS, setPreflightTimeoutS] = useState(ssh.preflight_timeout_s);
    const [imagePullTimeoutS, setImagePullTimeoutS] = useState(ssh.image_pull_timeout_s);
    const [trainerShmSizeGb, setTrainerShmSizeGb] = useState(ssh.trainer_shm_size_gb);
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
                trainer_shm_size_gb: trainerShmSizeGb,
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
            title='Managed SSH Training'
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
                label='SSH host verification timeout (s)'
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
                label='Trainer shared memory (GiB)'
                value={trainerShmSizeGb}
                onChange={(value) => update(setTrainerShmSizeGb, value)}
                minValue={1}
                step={1}
                width='100%'
            />
            <Text>
                Applies only to new containers. A running container keeps its current size. After jobs finish, stop the
                container and save its training target to recreate it.
            </Text>
        </SettingsSection>
    );
};
