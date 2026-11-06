import { useState } from 'react';

import { Checkbox, Item, NumberField, Picker, Text, TextField } from '@geti-ui/ui';
import { isEqual } from 'lodash-es';

import { SchemaRerunSettings } from '../../../api/openapi-spec';
import { SettingsSection } from './settings-section';
import { useSettingsPatch } from './use-settings-patch';

type RerunSettingsFormProps = { rerun: SchemaRerunSettings };

export const RerunSettingsForm = ({ rerun }: RerunSettingsFormProps) => {
    const patchMutation = useSettingsPatch();
    const [values, setValues] = useState(rerun);
    const [savedValues, setSavedValues] = useState(rerun);
    const [saved, setSaved] = useState(false);
    const dirty = !isEqual(values, savedValues);

    const update = (patch: Partial<SchemaRerunSettings>) => {
        setValues((current) => ({ ...current, ...patch }));
        setSaved(false);
    };

    const save = () => {
        patchMutation.mutate(
            { body: { rerun: values } },
            {
                onSuccess: () => {
                    setSavedValues(values);
                    setSaved(true);
                },
            }
        );
    };

    return (
        <SettingsSection
            title='Rerun visualization'
            description='Log model runs from Play until Stop. Changes apply to new runs; active runs are unchanged.'
            isDirty={dirty}
            isPending={patchMutation.isPending}
            saved={saved}
            error={patchMutation.error}
            onSave={save}
        >
            <Checkbox isSelected={values.enabled} onChange={(enabled) => update({ enabled })}>
                Enable Rerun logging
            </Checkbox>
            <Picker
                label='Rerun mode'
                selectedKey={values.mode}
                isDisabled={!values.enabled}
                onSelectionChange={(key) => {
                    if (key === 'spawn' || key === 'connect') update({ mode: key });
                }}
                width='100%'
            >
                <Item key='spawn'>Start locally</Item>
                <Item key='connect'>Connect to an existing server</Item>
            </Picker>
            {values.mode === 'spawn' ? (
                <>
                    <Checkbox
                        isSelected={values.open_viewer}
                        isDisabled={!values.enabled}
                        onChange={(open_viewer) => update({ open_viewer })}
                    >
                        Open the native viewer when a model run starts
                    </Checkbox>
                    <Text>
                        By default, Rerun runs in the background on the backend host without opening a window. Open a
                        Rerun viewer manually and connect to rerun+http://127.0.0.1:{values.spawn_port}/proxy.
                    </Text>
                    <NumberField
                        label='Local Rerun port'
                        value={values.spawn_port}
                        minValue={1}
                        maxValue={65535}
                        step={1}
                        isDisabled={!values.enabled}
                        onChange={(spawn_port) => {
                            if (Number.isInteger(spawn_port)) update({ spawn_port });
                        }}
                        width='100%'
                    />
                </>
            ) : (
                <TextField
                    label='Rerun connection address'
                    description={
                        'Use host:port or rerun+http(s)://host:port/proxy. The backend host connects to this address.'
                    }
                    value={values.connect_addr}
                    isDisabled={!values.enabled}
                    onChange={(connect_addr) => update({ connect_addr })}
                    width='100%'
                />
            )}
        </SettingsSection>
    );
};
