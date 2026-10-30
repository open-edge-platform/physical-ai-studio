import { useEffect, useRef } from 'react';

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

/**
 * Calls `onPress` when `combo` goes down and `onRelease` when its key comes up again, while `enabled`.
 * Losing focus counts as letting go, as the key-up would otherwise never arrive.
 */
export const useHeldHotkey = (combo: string, onPress: () => void, onRelease: () => void, enabled = true): void => {
    // The latest handlers, so a re-render while the key is held does not lose the hold.
    const handlers = useRef({ onPress, onRelease });
    handlers.current = { onPress, onRelease };

    useEffect(() => {
        if (!enabled) {
            return;
        }

        let held = false;
        // The key itself, not the combo: its modifiers may come up first.
        const baseKey = combo.split('+').at(-1);

        const release = () => {
            if (held) {
                held = false;
                handlers.current.onRelease();
            }
        };
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.repeat || held || isTypingTarget(e.target) || keyboardEventToCombo(e) !== combo) {
                return;
            }
            e.preventDefault();
            held = true;
            handlers.current.onPress();
        };
        const onKeyUp = (e: KeyboardEvent) => {
            if (keyboardEventToCombo(e)?.split('+').at(-1) === baseKey) {
                release();
            }
        };

        window.addEventListener('keydown', onKeyDown);
        window.addEventListener('keyup', onKeyUp);
        window.addEventListener('blur', release);
        return () => {
            window.removeEventListener('keydown', onKeyDown);
            window.removeEventListener('keyup', onKeyUp);
            window.removeEventListener('blur', release);
            release();
        };
    }, [combo, enabled]);
};
