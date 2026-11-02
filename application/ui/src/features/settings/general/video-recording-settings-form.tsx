import { useState } from 'react';

import { Item, Picker, Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaSettingsUpdate, SchemaVideoRecordingSettings } from '../../../api/openapi-spec';
import { SettingsSection } from './settings-section';
import { useSettingsPatch } from './use-settings-patch';

const AUTOMATIC_PRESET_ID = 'automatic';

type VideoRecordingSettingsFormProps = { videoRecording: SchemaVideoRecordingSettings };

export const VideoRecordingSettingsForm = ({ videoRecording }: VideoRecordingSettingsFormProps) => {
    const patchMutation = useSettingsPatch();
    const { data: videoRecordingPresets } = $api.useSuspenseQuery('get', '/api/system/video-recording-presets');
    const [presetId, setPresetId] = useState(videoRecording.preset ?? AUTOMATIC_PRESET_ID);
    const [dirty, setDirty] = useState(false);
    const [saved, setSaved] = useState(false);
    const presets = [
        {
            id: AUTOMATIC_PRESET_ID,
            name: 'Automatic',
            description: 'Automatically selects the best available video encoder for this computer.',
        },
        ...videoRecordingPresets.presets,
    ];
    const selectedPreset = presets.find((preset) => preset.id === presetId);

    const save = () => {
        const body: SchemaSettingsUpdate = {
            video_recording: { preset: presetId === AUTOMATIC_PRESET_ID ? null : presetId },
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
            title='Video recording'
            description='Choose how videos are encoded when recording new datasets.'
            isDirty={dirty}
            isPending={patchMutation.isPending}
            saved={saved}
            error={patchMutation.error}
            onSave={save}
        >
            <Picker
                label='Video encoder'
                width='100%'
                selectedKey={presetId}
                items={presets}
                onSelectionChange={(key) => {
                    setPresetId(String(key));
                    setDirty(true);
                    setSaved(false);
                }}
            >
                {(preset) => (
                    <Item key={preset.id} textValue={preset.name}>
                        <Text>{preset.name}</Text>
                        <Text slot='description'>{preset.description}</Text>
                    </Item>
                )}
            </Picker>
            {selectedPreset !== undefined && <Text>{selectedPreset.description}</Text>}
        </SettingsSection>
    );
};
