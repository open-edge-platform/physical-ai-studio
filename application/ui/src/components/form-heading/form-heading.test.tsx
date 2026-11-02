import { screen } from '@testing-library/react';

import { render } from '../../test-utils/render';
import { FormHeading } from './form-heading';

describe('FormHeading', () => {
    it('renders the heading text', () => {
        render(<FormHeading heading='Add new robot' backTo='/projects/p1/robots' backLabel='Back to robots' />);

        expect(screen.getByRole('heading', { name: 'Add new robot' })).toBeInTheDocument();
    });

    it('renders an accessible back link pointing to backTo', () => {
        render(<FormHeading heading='Add new robot' backTo='/projects/p1/robots' backLabel='Back to robots' />);

        const link = screen.getByRole('link', { name: 'Back to robots' });
        expect(link).toHaveAttribute('href', '/projects/p1/robots');
    });
});
