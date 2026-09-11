import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import { render } from '../../../../../test-utils/render';
import { FieldContextualHelp } from './field-contextual-help';

describe('FieldContextualHelp', () => {
    it('renders nothing when info is not provided', () => {
        render(<FieldContextualHelp />);

        expect(screen.queryByRole('button')).not.toBeInTheDocument();
    });

    it('renders help content with title and description', async () => {
        const user = userEvent.setup();

        render(
            <FieldContextualHelp
                info={{
                    title: 'Calibration guidance',
                    description: 'Upload the calibration values exported from your control board.',
                    variant: 'help',
                }}
            />
        );

        await user.click(screen.getByRole('button', { name: /Help$/ }));

        expect(await screen.findByRole('heading', { name: 'Calibration guidance' })).toBeVisible();
        expect(screen.getByText('Upload the calibration values exported from your control board.')).toBeVisible();
    });

    it('renders a Learn more link when link_url is set', async () => {
        const user = userEvent.setup();

        render(
            <FieldContextualHelp
                info={{
                    title: 'Robot plugin docs',
                    description: 'Read the plugin documentation for payload and calibration details.',
                    link_url: 'https://github.com/open-edge-platform/physical-ai-studio/tree/main/application/docs',
                    variant: 'info',
                }}
            />
        );

        await user.click(screen.getByRole('button', { name: /Information$/ }));

        const learnMore = await screen.findByRole('link', { name: 'Learn more' });
        expect(learnMore).toHaveAttribute(
            'href',
            'https://github.com/open-edge-platform/physical-ai-studio/tree/main/application/docs'
        );
        expect(learnMore).toHaveAttribute('target', '_blank');
        expect(learnMore).toHaveAttribute('rel', 'noopener noreferrer');
    });
});
