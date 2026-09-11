import { useEffect, useState } from 'react';

import { ActionButton, Keyboard, Text } from '@geti-ui/ui';

import { formatKeyCombo, keyboardEventToCombo } from '../../hotkeys/key-combo';

type HotkeyCaptureButtonProps = {
    combo: string;
    onChange: (combo: string) => void;
};

export const HotkeyCaptureButton = ({ combo, onChange }: HotkeyCaptureButtonProps) => {
    const [isListening, setIsListening] = useState(false);

    useEffect(() => {
        if (!isListening) {
            return;
        }

        const onKeyDown = (e: KeyboardEvent) => {
            e.preventDefault();

            if (e.key === 'Escape') {
                setIsListening(false);
                return;
            }

            const nextCombo = keyboardEventToCombo(e);
            if (nextCombo !== null) {
                onChange(nextCombo);
                setIsListening(false);
            }
        };

        window.addEventListener('keydown', onKeyDown);
        return () => window.removeEventListener('keydown', onKeyDown);
    }, [isListening, onChange]);

    return (
        <ActionButton onPress={() => setIsListening(true)} width='size-1600'>
            {isListening ? <Text>Press a key…</Text> : <Keyboard>{formatKeyCombo(combo)}</Keyboard>}
        </ActionButton>
    );
};
