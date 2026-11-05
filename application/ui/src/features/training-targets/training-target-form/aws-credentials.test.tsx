import { useState } from 'react';

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedAwsAuthCapabilities } from '../../../test-utils/mocks/mock-aws-auth';
import { render } from '../../../test-utils/render';
import { AwsCredentials } from './aws-credentials';

const Wrapper = () => {
    const [profileName, setProfileName] = useState('');
    return <AwsCredentials profileName={profileName} onProfileChange={setProfileName} isDisabled={false} />;
};

describe('AWS credentials', () => {
    beforeEach(() => {
        server.use(
            http.get('/api/remote-trainers/providers/aws/auth', () =>
                HttpResponse.json(getMockedAwsAuthCapabilities({ profiles: ['default', 'studio'] }))
            )
        );
    });

    it('selects a named profile without a separate authentication action', async () => {
        const user = userEvent.setup();
        render(<Wrapper />);
        expect(screen.queryByRole('button', { name: 'Check connection' })).not.toBeInTheDocument();
        expect(screen.queryByRole('radio')).not.toBeInTheDocument();
        const picker = screen.getByRole('button', { name: /AWS profile/ });
        await waitFor(() => expect(picker).toBeEnabled());
        await user.click(picker);
        expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual(['default', 'studio']);
        await user.click(screen.getByRole('option', { name: 'studio' }));
        expect(picker).toHaveTextContent('studio');
        await user.click(picker);
        await user.click(screen.getByRole('option', { name: 'default' }));
        expect(picker).toHaveTextContent('default');
    });

    it('explains backend setup, renewal, Docker access, and permissions in the information hint', async () => {
        const user = userEvent.setup();
        render(<Wrapper />);
        expect(screen.queryByText(/Configure an AWS named profile/)).not.toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Named profile help' }));
        expect(screen.getByText(/machine running the Studio backend/)).toBeInTheDocument();
        expect(screen.getByText('aws configure sso --profile studio')).toBeInTheDocument();
        expect(screen.getByText('aws sso login --profile studio')).toBeInTheDocument();
        expect(screen.getByText('aws configure --profile studio')).toBeInTheDocument();
        expect(screen.getByText(/backend container user/)).toBeInTheDocument();
        expect(screen.getByText(/When credentials expire/)).toBeInTheDocument();
        expect(screen.getByText(/role must trust this identity/)).toBeInTheDocument();
        expect(screen.getByRole('link', { name: 'Configure a named profile' })).toBeInTheDocument();
    });

    it('guides setup when no profiles exist', async () => {
        server.use(
            http.get('/api/remote-trainers/providers/aws/auth', () =>
                HttpResponse.json(getMockedAwsAuthCapabilities({ profiles: [] }))
            )
        );
        render(<Wrapper />);
        expect(await screen.findByRole('status')).toHaveTextContent('No AWS profiles found');
    });
});
