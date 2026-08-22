import { useState } from 'react';

import { ActionButton, Flex, Icon, TextField } from '@geti-ui/ui';
import { Close } from '@geti-ui/ui/icons';

import { SchemaHuggingFaceSettings, SchemaSettingsUpdate } from '../../../api/openapi-spec';
import { SettingsSection } from './settings-section';
import { useSettingsPatch } from './use-settings-patch';

type HuggingFaceSettingsFormProps = { huggingface: SchemaHuggingFaceSettings };

export const HuggingFaceSettingsForm = ({ huggingface }: HuggingFaceSettingsFormProps) => {
    const patchMutation = useSettingsPatch();
    const [token, setToken] = useState('');
    const [saved, setSaved] = useState(false);
    const isSet = huggingface.hf_token != null;

    const save = () => {
        const body: SchemaSettingsUpdate = {
            huggingface: { hf_token: token },
        };
        patchMutation.mutate(
            { body },
            {
                onSuccess: () => {
                    setToken('');
                    setSaved(true);
                },
            }
        );
    };

    const clear = () => {
        patchMutation.mutate({ body: { huggingface: { hf_token: null } } }, { onSuccess: () => setSaved(true) });
    };

    return (
        <SettingsSection
            title='Hugging Face'
            description='Token used to authenticate downloads of pretrained training assets.'
            isDirty={!isSet && token !== ''}
            isPending={patchMutation.isPending}
            saved={saved}
            error={patchMutation.error}
            onSave={save}
        >
            {isSet ? (
                <Flex alignItems='end' gap='size-100'>
                    <TextField
                        label='Hugging Face token'
                        value={'hf_**********************************'}
                        isDisabled
                        width='100%'
                    />
                    <ActionButton
                        aria-label='Clear Hugging Face token'
                        isQuiet
                        onPress={clear}
                        isDisabled={patchMutation.isPending}
                    >
                        <Icon>
                            <Close />
                        </Icon>
                        <span
                            style={{
                                paddingLeft: 'var(--spectrum-global-dimension-size-100)',
                                paddingRight: 'var(--spectrum-global-dimension-size-200)',
                            }}
                        >
                            Clear
                        </span>
                    </ActionButton>
                </Flex>
            ) : (
                <TextField
                    label='Hugging Face token'
                    type='password'
                    value={token}
                    onChange={setToken}
                    onKeyDown={(event) => {
                        if (event.key === 'Enter') {
                            event.preventDefault();
                            save();
                        }
                    }}
                    width='100%'
                />
            )}
        </SettingsSection>
    );
};
