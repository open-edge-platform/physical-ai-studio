import { useState } from 'react';

import { NumberField } from '@geti-ui/ui';

import { SchemaSettingsUpdate, SchemaTrainerClientSettings } from '../../../api/openapi-spec';
import { SettingsSection } from './settings-section';
import { useSettingsPatch } from './use-settings-patch';

type TrainerSettingsFormProps = { trainer: SchemaTrainerClientSettings };

export const TrainerSettingsForm = ({ trainer }: TrainerSettingsFormProps) => {
    const patchMutation = useSettingsPatch();
    const [requestTimeoutS, setRequestTimeoutS] = useState(trainer.request_timeout_s);
    const [downloadReadTimeoutS, setDownloadReadTimeoutS] = useState(trainer.download_read_timeout_s);
    const [streamReconnectMaxS, setStreamReconnectMaxS] = useState(trainer.stream_reconnect_max_s);
    const [streamReconnectBackoffMaxS, setStreamReconnectBackoffMaxS] = useState(
        trainer.stream_reconnect_backoff_max_s
    );
    const [dirty, setDirty] = useState(false);
    const [saved, setSaved] = useState(false);

    const update = (setValue: (value: number) => void, value: number) => {
        if (!Number.isNaN(value)) {
            setValue(value);
            setDirty(true);
            setSaved(false);
        }
    };

    const save = () => {
        const body: SchemaSettingsUpdate = {
            trainer: {
                request_timeout_s: requestTimeoutS,
                download_read_timeout_s: downloadReadTimeoutS,
                stream_reconnect_max_s: streamReconnectMaxS,
                stream_reconnect_backoff_max_s: streamReconnectBackoffMaxS,
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
            title='Trainer'
            description='Client timeouts for talking to a remote trainer service.'
            isDirty={dirty}
            isPending={patchMutation.isPending}
            saved={saved}
            error={patchMutation.error}
            onSave={save}
        >
            <NumberField
                label='Request timeout (s)'
                value={requestTimeoutS}
                onChange={(value) => update(setRequestTimeoutS, value)}
                width='100%'
            />
            <NumberField
                label='Artifact download read timeout (s)'
                value={downloadReadTimeoutS}
                onChange={(value) => update(setDownloadReadTimeoutS, value)}
                width='100%'
            />
            <NumberField
                label='Stream reconnect budget (s)'
                value={streamReconnectMaxS}
                onChange={(value) => update(setStreamReconnectMaxS, value)}
                width='100%'
            />
            <NumberField
                label='Stream reconnect backoff max (s)'
                value={streamReconnectBackoffMaxS}
                onChange={(value) => update(setStreamReconnectBackoffMaxS, value)}
                width='100%'
            />
        </SettingsSection>
    );
};
