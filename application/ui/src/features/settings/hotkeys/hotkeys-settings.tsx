import { useState } from 'react';

import { ActionButton, Flex, Heading, Text, View } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SchemaSettingsUpdate } from '../../../api/openapi-spec';
import { HOTKEY_ACTION_IDS, HOTKEY_ACTIONS, HotkeyActionId } from '../../hotkeys/hotkey-actions';
import { getEffectiveBindings } from '../../hotkeys/key-combo';
import { SettingsSection } from '../general/settings-section';
import { useSettingsPatch } from '../general/use-settings-patch';
import { HotkeyCaptureButton } from './hotkey-capture-button';

const SCOPES = [...new Set(HOTKEY_ACTION_IDS.map((actionId) => HOTKEY_ACTIONS[actionId].scope))];

export const HotkeysSettings = () => {
    const { data: settings } = $api.useSuspenseQuery('get', '/api/settings');
    const patchMutation = useSettingsPatch();
    const [bindings, setBindings] = useState(() => getEffectiveBindings(settings.hotkeys.bindings));
    const [dirty, setDirty] = useState(false);
    const [saved, setSaved] = useState(false);

    const updateBinding = (actionId: HotkeyActionId, combo: string) => {
        setBindings((prev) => ({ ...prev, [actionId]: combo }));
        setDirty(true);
        setSaved(false);
    };

    const resetBinding = (actionId: HotkeyActionId) => {
        updateBinding(actionId, HOTKEY_ACTIONS[actionId].defaultCombo);
    };

    const save = () => {
        const body: SchemaSettingsUpdate = { hotkeys: { bindings } };
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
        <View padding='size-400' height='100%' maxWidth='240ch' marginX='auto'>
            <Heading level={1}>Hotkeys</Heading>
            <Text>Configure keyboard shortcuts for app actions.</Text>
            {SCOPES.map((scope) => (
                <SettingsSection
                    key={scope}
                    title={scope}
                    description={`Keyboard shortcuts for ${scope.toLowerCase()}.`}
                    isDirty={dirty}
                    isPending={patchMutation.isPending}
                    saved={saved}
                    error={patchMutation.error}
                    onSave={save}
                >
                    {HOTKEY_ACTION_IDS.filter((actionId) => HOTKEY_ACTIONS[actionId].scope === scope).map(
                        (actionId) => (
                            <Flex key={actionId} alignItems='center' justifyContent='space-between' gap='size-200'>
                                <Text>{HOTKEY_ACTIONS[actionId].label}</Text>
                                <Flex alignItems='center' gap='size-100'>
                                    <HotkeyCaptureButton
                                        combo={bindings[actionId]}
                                        onChange={(combo) => updateBinding(actionId, combo)}
                                    />
                                    <ActionButton
                                        isQuiet
                                        aria-label={`Reset ${HOTKEY_ACTIONS[actionId].label} to default`}
                                        onPress={() => resetBinding(actionId)}
                                        isDisabled={bindings[actionId] === HOTKEY_ACTIONS[actionId].defaultCombo}
                                    >
                                        <Text>Reset</Text>
                                    </ActionButton>
                                </Flex>
                            </Flex>
                        )
                    )}
                </SettingsSection>
            ))}
        </View>
    );
};
