export type HotkeyActionId = 'recording.start_episode' | 'recording.accept_episode' | 'recording.discard_episode';

type HotkeyActionDefinition = {
    label: string;
    scope: string;
    defaultCombo: string;
};

export const HOTKEY_ACTIONS: Record<HotkeyActionId, HotkeyActionDefinition> = {
    'recording.start_episode': {
        label: 'Start episode',
        scope: 'Recording',
        defaultCombo: ' ',
    },
    'recording.accept_episode': {
        label: 'Save episode',
        scope: 'Recording',
        defaultCombo: 'Enter',
    },
    'recording.discard_episode': {
        label: 'Discard episode',
        scope: 'Recording',
        defaultCombo: 'Backspace',
    },
};

export const HOTKEY_ACTION_IDS = Object.keys(HOTKEY_ACTIONS) as HotkeyActionId[];
