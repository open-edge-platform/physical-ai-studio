import { Heading, Text, View } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { AboutSettings } from './about-settings';
import { HuggingFaceSettingsForm } from './huggingface-settings-form';
import { SshSettingsForm } from './ssh-settings-form';
import { TrainerSettingsForm } from './trainer-settings-form';
import { VideoRecordingSettingsForm } from './video-recording-settings-form';

export const GeneralSettings = () => {
    const { data: settings } = $api.useSuspenseQuery('get', '/api/settings');

    return (
        <View padding='size-400' height='100%' maxWidth='240ch' marginX='auto'>
            <Heading level={1}>General</Heading>
            <Text>Configure video recording, trainer, and Hugging Face defaults.</Text>
            <AboutSettings />
            <HuggingFaceSettingsForm huggingface={settings.huggingface} />
            <VideoRecordingSettingsForm videoRecording={settings.video_recording} />
            <TrainerSettingsForm trainer={settings.trainer} />
            <SshSettingsForm ssh={settings.ssh} />
        </View>
    );
};
