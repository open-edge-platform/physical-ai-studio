import { screen } from '@testing-library/react';

import { render } from '../../../test-utils/render';
import { OpenRerunButton } from './open-rerun-button';

describe('OpenRerunButton', () => {
    it('opens the session viewer link in a new browser tab without navigating away', () => {
        const url = 'https://app.rerun.io/version/0.26.2/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9876%2Fproxy';
        render(<OpenRerunButton url={url} />);

        const link = screen.getByRole('button', { name: 'Open Rerun' });
        expect(link).toHaveAttribute('href', url);
        expect(link).toHaveAttribute('target', '_blank');
        expect(link).toHaveAttribute('rel', 'noopener noreferrer');
        expect(screen.getByText(/Local streams require a browser on the backend machine/)).toBeInTheDocument();
    });

    it.each([null, undefined])('disables the button when no viewer is available (%s)', (url) => {
        render(<OpenRerunButton url={url} />);
        const button = screen.getByRole('button', { name: 'Open Rerun' });
        expect(button).toBeDisabled();
        expect(button).not.toHaveAttribute('href');
    });
});
