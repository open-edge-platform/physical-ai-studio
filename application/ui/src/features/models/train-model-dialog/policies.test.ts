import { describe, expect, it } from 'vitest';

import { formatBytes } from './policies';

describe('formatBytes', () => {
    it('shows whole advertised GB labels for reported GPU memory', () => {
        expect(formatBytes(40960 * 1024 ** 2)).toBe('40 GB');
        expect(formatBytes(39.4 * 1024 ** 3)).toBe('40 GB');
        expect(formatBytes(39 * 1024 ** 3)).toBe('39 GB');
    });
});
