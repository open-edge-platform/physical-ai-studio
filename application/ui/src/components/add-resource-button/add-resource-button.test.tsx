import { screen } from '@testing-library/react';

import { render } from '../../test-utils/render';
import { AddResourceButton } from './add-resource-button';

describe('AddResourceButton', () => {
    it('renders the children as the link content', () => {
        render(<AddResourceButton to='/projects/p1/robots/new'>Add new robot</AddResourceButton>);

        expect(screen.getByRole('link', { name: /add new robot/i })).toBeInTheDocument();
    });

    it('links to the given destination', () => {
        render(<AddResourceButton to='/projects/p1/robots/new'>Add new robot</AddResourceButton>);

        expect(screen.getByRole('link', { name: /add new robot/i })).toHaveAttribute('href', '/projects/p1/robots/new');
    });
});
