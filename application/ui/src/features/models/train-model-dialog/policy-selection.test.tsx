import { screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { render } from '../../../test-utils/render';
import { PolicySelection } from './policy-selection';

const renderWithVram = (gib: number) =>
    render(
        <PolicySelection
            selectedPolicy='pi05'
            onSelectionChange={vi.fn()}
            trainingDevice={{ type: 'cuda', name: 'A100', index: 0, memory: gib * 1024 ** 3 }}
        />
    );

describe('PolicySelection VRAM warning', () => {
    it('does not warn when reported VRAM and the policy card both display 40 GB', () => {
        renderWithVram(39.4);
        expect(screen.getAllByText('≥ 40 GB VRAM')).toHaveLength(2);
        expect(screen.queryByText(/Training may fail/)).not.toBeInTheDocument();
    });

    it('warns when displayed VRAM is below the policy requirement', () => {
        renderWithVram(38.4);
        expect(screen.getByText(/Training may fail/)).toBeInTheDocument();
    });
});
