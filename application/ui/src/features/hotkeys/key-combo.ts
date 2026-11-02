import { HOTKEY_ACTION_IDS, HOTKEY_ACTIONS, HotkeyActionId } from './hotkey-actions';

const MODIFIER_KEYS = new Set(['Control', 'Alt', 'Shift', 'Meta']);

const DISPLAY_KEY_LABELS: Record<string, string> = {
    ArrowRight: '→',
    ArrowLeft: '←',
    ArrowUp: '↑',
    ArrowDown: '↓',
    ' ': 'Space',
};

const modifierPrefix = (e: Pick<KeyboardEvent, 'ctrlKey' | 'altKey' | 'shiftKey' | 'metaKey'>): string[] => {
    const modifiers: string[] = [];
    if (e.ctrlKey) modifiers.push('Ctrl');
    if (e.altKey) modifiers.push('Alt');
    if (e.shiftKey) modifiers.push('Shift');
    if (e.metaKey) modifiers.push('Meta');
    return modifiers;
};

// Modifiers come from the boolean event flags, never from `e.key`'s casing -
// Caps Lock alone also produces an uppercase `e.key` with `shiftKey: false`, so
// trusting casing would make that indistinguishable from an actual Shift press.
const normalizeBaseKey = (key: string): string => (key.length === 1 ? key.toUpperCase() : key);

/** Canonical combo string for a keydown event, or null for a bare modifier press. */
export const keyboardEventToCombo = (e: KeyboardEvent): string | null => {
    if (MODIFIER_KEYS.has(e.key)) {
        return null;
    }

    return [...modifierPrefix(e), normalizeBaseKey(e.key)].join('+');
};

/** Human-readable form of a combo string, e.g. "Shift+ArrowLeft" -> "Shift+←". */
export const formatKeyCombo = (combo: string): string =>
    combo
        .split('+')
        .map((part) => DISPLAY_KEY_LABELS[part] ?? part)
        .join('+');

/** Merges registry defaults with any stored per-action overrides. */
export const getEffectiveBindings = (bindings: Record<string, string> = {}): Record<HotkeyActionId, string> => {
    const effective = {} as Record<HotkeyActionId, string>;

    for (const actionId of HOTKEY_ACTION_IDS) {
        effective[actionId] = bindings[actionId] ?? HOTKEY_ACTIONS[actionId].defaultCombo;
    }

    return effective;
};
