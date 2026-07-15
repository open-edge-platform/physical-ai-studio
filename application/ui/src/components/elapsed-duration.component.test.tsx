import { act, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { render } from '../test-utils/render';
import { ElapsedDuration } from './elapsed-duration.component';

describe('ElapsedDuration', () => {
    beforeEach(() => {
        vi.useFakeTimers();
        vi.setSystemTime(new Date('2026-07-14T12:00:00Z'));
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('updates the displayed duration every second', () => {
        render(<ElapsedDuration date='2026-07-14T11:59:58Z' />);

        expect(screen.getByText('2s')).toBeInTheDocument();

        act(() => vi.advanceTimersByTime(1000));

        expect(screen.getByText('3s')).toBeInTheDocument();
    });
});
