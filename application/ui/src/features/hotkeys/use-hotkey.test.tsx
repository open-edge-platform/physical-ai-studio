import { rs } from '@rstest/core';
import { fireEvent } from '@testing-library/react';

import { renderHook } from '../../test-utils/render';
import { useHeldHotkey } from './use-hotkey';

describe('useHeldHotkey', () => {
    it('presses once on key-down, ignoring repeats, and releases on key-up or losing focus', () => {
        const onPress = rs.fn();
        const onRelease = rs.fn();
        renderHook(() => useHeldHotkey('ArrowDown', onPress, onRelease));

        fireEvent.keyDown(window, { key: 'ArrowDown' });
        fireEvent.keyDown(window, { key: 'ArrowDown', repeat: true });
        expect(onPress).toHaveBeenCalledTimes(1);

        fireEvent.keyUp(window, { key: 'ArrowDown' });
        expect(onRelease).toHaveBeenCalledTimes(1);

        fireEvent.keyDown(window, { key: 'ArrowDown' });
        fireEvent.blur(window);
        expect(onPress).toHaveBeenCalledTimes(2);
        expect(onRelease).toHaveBeenCalledTimes(2);
    });
});
