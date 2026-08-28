import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { describe, expect, it } from 'vitest';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { render } from '../../../../test-utils/render';
import { RobotFormProvider } from '../provider';
import { SO101FormFields } from './so101';

const IDENTIFY_PATH = '/api/robots/catalog/{robot_type}/identify';
const DISCOVER_PATH = '/api/robots/catalog/{robot_type}/discover';

const IDENTIFY_OVERLOAD_MESSAGE =
    'Robot identify failed: gripper stopped responding during motion. The servo may have tripped overload protection. Power-cycle the robot and try again.';

const renderSo101Form = () =>
    render(
        <RobotFormProvider>
            <SO101FormFields />
        </RobotFormProvider>
    );

const clickIdentify = async (user: ReturnType<typeof userEvent.setup>) => {
    await user.click(await screen.findByRole('button', { name: 'Identify' }));
};

describe('SO101FormFields identify errors', () => {
    it('renders the actual identify error instead of Permission Denied', async () => {
        server.use(
            http.get(DISCOVER_PATH, () => HttpResponse.json([])),
            http.post(IDENTIFY_PATH, () =>
                HttpResponse.json(
                    {
                        error_code: 'robot_identify_error',
                        message: IDENTIFY_OVERLOAD_MESSAGE,
                        http_status: 400,
                    },
                    { status: 400 }
                )
            )
        );

        const user = userEvent.setup();
        renderSo101Form();

        await clickIdentify(user);

        expect(await screen.findByText('Identify Failed')).toBeInTheDocument();
        expect(screen.getByText(/Power-cycle the robot and try again/)).toBeInTheDocument();
        expect(screen.queryByText('Permission Denied')).not.toBeInTheDocument();
    });

    it('renders Permission Denied guidance for serial permission failures', async () => {
        server.use(
            http.get(DISCOVER_PATH, () => HttpResponse.json([])),
            http.post(IDENTIFY_PATH, () =>
                HttpResponse.json(
                    {
                        error_code: 'serial_permission_denied',
                        message: 'Permission denied while opening the serial device.',
                        http_status: 403,
                    },
                    { status: 403 }
                )
            )
        );

        const user = userEvent.setup();
        renderSo101Form();

        await clickIdentify(user);

        expect(await screen.findByText('Permission Denied')).toBeInTheDocument();
        expect(screen.queryByText('Identify Failed')).not.toBeInTheDocument();
    });
});
