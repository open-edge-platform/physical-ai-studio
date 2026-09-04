import { useEffect } from 'react';

import { keyboardEventToCombo } from './key-combo';

const isTypingTarget = (target: EventTarget | null): boolean => {
    if (!(target instanceof HTMLElement)) {
        return false;
    }

    return (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        target.isContentEditable
    );
};

/**
 * Fires `handler` when `combo` (a canonical string from `keyboardEventToCombo`) is
 * pressed anywhere in the document, while `enabled`. Ignores keystrokes into text
 * inputs/content-editable elements and key-repeat events.
 */
export const useHotkey = (combo: string, handler: () => void, enabled = true): void => {
    useEffect(() => {
        if (!enabled) {
            return;
        }

        const onKeyDown = (e: KeyboardEvent) => {
            if (e.repeat || isTypingTarget(e.target)) {
                return;
            }

            if (keyboardEventToCombo(e) === combo) {
                e.preventDefault();
                handler();
            }
        };

        window.addEventListener('keydown', onKeyDown);
        return () => window.removeEventListener('keydown', onKeyDown);
    }, [combo, enabled, handler]);
};
