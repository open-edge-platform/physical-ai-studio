import { rs } from '@rstest/core';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { getMockedRobotCatalogDefinition } from '../../../../test-utils/mocks/mock-robot-catalog-definition';
import { render } from '../../../../test-utils/render';
import { RobotFormProvider } from '../provider';
import { CreateRobotWizard } from './create-robot-wizard';

// The 3D thumbnails and preview need WebGL.
rs.mock('./robot-thumbnail', () => ({
    RobotThumbnail: () => <div>thumbnail</div>,
}));
rs.mock('../preview', () => ({
    Preview: () => <div>preview</div>,
}));

const CATALOG = [
    getMockedRobotCatalogDefinition({ type: 'SO101_Follower', display_name: 'SO101 Follower' }),
    getMockedRobotCatalogDefinition({ type: 'SO101_Leader', display_name: 'SO101 Leader', role: 'leader' }),
    getMockedRobotCatalogDefinition({ type: 'Solo_Follower', display_name: 'Solo Arm' }),
];

const renderWizard = (robot?: { type: string; name: string; payload: unknown }) => {
    server.use(
        http.get('/api/robots/catalog', () => HttpResponse.json(CATALOG)),
        http.get('/api/robots/catalog/{robot_type}/schema', () =>
            HttpResponse.json({ type: 'object', properties: {}, required: [] })
        )
    );

    return render(
        <RobotFormProvider robot={robot}>
            <CreateRobotWizard />
        </RobotFormProvider>,
        { route: '/projects/project-1/robots/new', path: '/projects/:project_id/robots/new' }
    );
};

const getStep = (name: RegExp) => screen.getByRole('button', { name });

describe('CreateRobotWizard', () => {
    it('blocks the later steps until a robot is chosen', async () => {
        renderWizard();

        expect(await screen.findByRole('heading', { name: /Which robot are you adding/ })).toBeInTheDocument();
        expect(getStep(/Choose robot/)).toHaveAttribute('aria-current', 'step');
        expect(getStep(/Choose variant/)).toBeDisabled();
        expect(getStep(/Connect & configure/)).toBeDisabled();
        expect(screen.queryByRole('textbox', { name: /Robot name/ })).not.toBeInTheDocument();
    });

    it('unlocks the variant step but keeps configure blocked until a variant is chosen', async () => {
        const user = userEvent.setup();
        renderWizard();

        await user.click(await screen.findByRole('button', { name: /SO101/ }));

        expect(await screen.findByRole('heading', { name: /which variant/ })).toBeInTheDocument();
        expect(getStep(/Choose robot/)).toBeEnabled();
        expect(getStep(/Choose variant/)).toHaveAttribute('aria-current', 'step');
        expect(getStep(/Connect & configure/)).toBeDisabled();
        expect(screen.queryByRole('textbox', { name: /Robot name/ })).not.toBeInTheDocument();
    });

    it('opens the form once a variant is chosen and prefills the name', async () => {
        const user = userEvent.setup();
        renderWizard();

        await user.click(await screen.findByRole('button', { name: /SO101/ }));
        await user.click(await screen.findByRole('button', { name: /Leader/ }));

        expect(await screen.findByRole('textbox', { name: /Robot name/ })).toHaveValue('SO101 Leader');
        expect(getStep(/Connect & configure/)).toHaveAttribute('aria-current', 'step');
    });

    it('shows the variant step with the only variant preselected for single-variant robots', async () => {
        const user = userEvent.setup();
        renderWizard();

        await user.click(await screen.findByRole('button', { name: /Solo Arm/ }));

        expect(await screen.findByRole('heading', { name: /which variant/ })).toBeInTheDocument();
        expect(getStep(/Choose variant/)).toHaveAttribute('aria-current', 'step');
        expect(screen.getByRole('button', { name: /Solo Arm/, pressed: true })).toBeInTheDocument();
        expect(getStep(/Connect & configure/)).toBeEnabled();

        await user.click(getStep(/Connect & configure/));

        expect(await screen.findByRole('textbox', { name: /Robot name/ })).toHaveValue('Solo Arm');
    });

    it('blocks configure again when another robot is picked without choosing its variant', async () => {
        const user = userEvent.setup();
        renderWizard({ type: 'Solo_Follower', name: 'My solo', payload: {} });

        expect(await screen.findByRole('textbox', { name: /Robot name/ })).toBeInTheDocument();

        await user.click(getStep(/Choose robot/));
        await user.click(await screen.findByRole('button', { name: /SO101/ }));

        expect(await screen.findByRole('heading', { name: /which variant/ })).toBeInTheDocument();
        expect(getStep(/Connect & configure/)).toBeDisabled();
    });

    it('starts on the form when a robot type is already selected', async () => {
        renderWizard({ type: 'SO101_Follower', name: 'My arm', payload: {} });

        expect(await screen.findByRole('textbox', { name: /Robot name/ })).toHaveValue('My arm');
        expect(getStep(/Choose robot/)).toBeEnabled();
        expect(getStep(/Choose variant/)).toBeEnabled();
    });

    it('keeps the chosen robot and variant highlighted when navigating between steps', async () => {
        const user = userEvent.setup();
        renderWizard();

        await user.click(await screen.findByRole('button', { name: /SO101/ }));
        await user.click(await screen.findByRole('button', { name: /Leader/ }));
        expect(await screen.findByRole('textbox', { name: /Robot name/ })).toBeInTheDocument();

        await user.click(getStep(/Choose robot/));
        expect(await screen.findByRole('button', { name: /SO101/, pressed: true })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Solo Arm/, pressed: false })).toBeInTheDocument();

        await user.click(getStep(/Choose variant/));
        expect(await screen.findByRole('button', { name: /Leader/, pressed: true })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Follower/, pressed: false })).toBeInTheDocument();

        await user.click(getStep(/Connect & configure/));
        expect(await screen.findByRole('textbox', { name: /Robot name/ })).toHaveValue('SO101 Leader');
    });
});
