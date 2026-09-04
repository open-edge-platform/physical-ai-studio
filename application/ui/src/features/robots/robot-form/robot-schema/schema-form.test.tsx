import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { describe, expect, it } from 'vitest';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { render } from '../../../../test-utils/render';
import { RobotFormProvider, useRobotForm } from '../provider';
import { SchemaForm } from './schema-form';

const bimanualRebotSchema: Parameters<typeof SchemaForm>[0]['schema'] = {
    $defs: {
        RebotB601FollowerConfigPayload: {
            type: 'object',
            properties: {
                port: {
                    type: 'string',
                    title: 'Port',
                },
                can_adapter: {
                    type: 'string',
                    title: 'Can Adapter',
                    default: 'damiao',
                    'x-physicalai-ui': { advanced_configuration: true },
                },
            },
            required: ['port'],
            'x-physicalai-ui': [
                {
                    kind: 'section',
                    id: 'connection',
                    title: 'Select robot',
                    items: [
                        {
                            kind: 'connection',
                            label: 'Select robot',
                            bind: { connection: 'port' },
                            device_discovery: true,
                        },
                    ],
                },
            ],
        },
    },
    type: 'object',
    properties: {
        left_arm_config: { $ref: '#/$defs/RebotB601FollowerConfigPayload', description: 'left_arm_config' },
        right_arm_config: { $ref: '#/$defs/RebotB601FollowerConfigPayload', description: 'right_arm_config' },
    },
    required: ['left_arm_config', 'right_arm_config'],
};

const lerobotSO101Schema: Parameters<typeof SchemaForm>[0]['schema'] = {
    $defs: {
        CameraConfigPayload: {
            type: 'object',
            properties: {
                fps: { type: 'integer', title: 'Fps' },
            },
        },
    },
    type: 'object',
    properties: {
        port: {
            type: 'string',
            title: 'Port',
        },
        cameras: {
            type: 'object',
            title: 'Cameras',
            default: {},
            additionalProperties: { $ref: '#/$defs/CameraConfigPayload' },
        },
    },
    required: ['port'],
    'x-physicalai-ui': [
        {
            kind: 'section',
            id: 'connection',
            items: [
                { kind: 'connection', label: 'Select robot', device_discovery: true, bind: { connection: 'port' } },
            ],
        },
    ],
};

const rebotSchema: Parameters<typeof SchemaForm>[0]['schema'] = {
    type: 'object',
    properties: {
        connection_string: { type: 'string' },
        serial_number: { type: 'string' },
    },
    'x-physicalai-ui': [
        {
            kind: 'section',
            id: 'connection',
            items: [
                {
                    kind: 'connection',
                    label: 'Select robot',
                    device_discovery: true,
                    bind: { connection: 'connection_string', serial_number: 'serial_number' },
                },
            ],
        },
    ],
};

const Payload = () => {
    const { payload } = useRobotForm();
    return <output>{JSON.stringify(payload)}</output>;
};

describe('SchemaForm', () => {
    it('renders top-level and group informational text blocks', async () => {
        const schema: Parameters<typeof SchemaForm>[0]['schema'] = {
            type: 'object',
            properties: {
                connection_string: {
                    type: 'string',
                },
            },
            'x-physicalai-ui': [
                {
                    kind: 'section',
                    id: 'connection',
                    items: [
                        {
                            kind: 'info',
                            title: 'Before connecting',
                            text: 'Power on the robot and unlock the arm.',
                        },
                        {
                            kind: 'connection',
                            label: 'Connection',
                            device_discovery: true,
                            bind: { connection: 'connection_string' },
                        },
                        {
                            kind: 'info',
                            text: 'If no ports are listed, click refresh.',
                        },
                    ],
                },
            ],
        };

        render(
            <RobotFormProvider>
                <SchemaForm schema={schema} />
            </RobotFormProvider>
        );

        expect(screen.getByText('Before connecting')).toBeVisible();
        expect(screen.getByText('Power on the robot and unlock the arm.')).toBeVisible();
        expect(screen.getByText('If no ports are listed, click refresh.')).toBeVisible();
        expect(await screen.findByRole('button', { name: 'Connection' })).toBeVisible();
        expect(screen.queryByRole('textbox', { name: 'Connection String' })).not.toBeInTheDocument();
    });

    it('shows description help text for boolean fields', () => {
        const schema: Parameters<typeof SchemaForm>[0]['schema'] = {
            type: 'object',
            properties: {
                torque_enabled: {
                    type: 'boolean',
                    title: 'Torque Enabled',
                    description: 'Toggle motor torque on startup.',
                },
            },
        };

        render(
            <RobotFormProvider>
                <SchemaForm schema={schema} />
            </RobotFormProvider>
        );

        expect(screen.getByText('Toggle motor torque on startup.')).toBeVisible();
    });

    it('renders recursively nested sections in item order', () => {
        const schema: Parameters<typeof SchemaForm>[0]['schema'] = {
            type: 'object',
            properties: {
                port: { type: 'string', title: 'Port' },
            },
            'x-physicalai-ui': [
                {
                    kind: 'section',
                    id: 'setup',
                    title: 'Setup',
                    items: [
                        { kind: 'info', text: 'Connect the robot before configuring it.' },
                        {
                            kind: 'section',
                            id: 'manual-connection',
                            title: 'Manual connection',
                            items: [{ kind: 'field', name: 'port' }],
                        },
                    ],
                },
            ],
        };

        render(
            <RobotFormProvider>
                <SchemaForm schema={schema} />
            </RobotFormProvider>
        );

        expect(screen.getByRole('heading', { name: 'Setup' })).toBeVisible();
        expect(screen.getByText('Connect the robot before configuring it.')).toBeVisible();
        expect(screen.getByRole('heading', { name: 'Manual connection' })).toBeVisible();
        expect(screen.getByRole('textbox', { name: 'Port' })).toBeVisible();
    });

    it('renders unowned fields after explicit items', () => {
        const schema: Parameters<typeof SchemaForm>[0]['schema'] = {
            type: 'object',
            properties: {
                port: { type: 'string', title: 'Port' },
                baud_rate: { type: 'integer', title: 'Baud Rate' },
            },
            'x-physicalai-ui': [{ kind: 'field', name: 'port' }],
        };

        render(
            <RobotFormProvider>
                <SchemaForm schema={schema} />
            </RobotFormProvider>
        );

        expect(screen.getByRole('textbox', { name: 'Port' })).toBeVisible();
        expect(screen.getByRole('spinbutton', { name: 'Baud Rate' })).toBeVisible();
    });

    it('falls back to JSON Schema fields when plugin UI metadata uses an unsupported shape', () => {
        const schema = {
            type: 'object',
            properties: {
                connection_string: { type: 'string', title: 'Connection String' },
            },
            'x-physicalai-ui': { groups: { connection: {} } },
        } as unknown as Parameters<typeof SchemaForm>[0]['schema'];

        render(
            <RobotFormProvider>
                <SchemaForm schema={schema} />
            </RobotFormProvider>
        );

        expect(screen.getByRole('textbox', { name: 'Connection String' })).toBeVisible();
    });

    it('renders defaulted fields that are marked required', () => {
        const schema: Parameters<typeof SchemaForm>[0]['schema'] = {
            type: 'object',
            properties: {
                id: {
                    type: 'string',
                    title: 'Robot ID',
                    default: '',
                    'x-physicalai-ui': { required: true },
                },
            },
        };

        render(
            <RobotFormProvider>
                <SchemaForm schema={schema} />
            </RobotFormProvider>
        );

        expect(screen.getByRole('textbox', { name: /Robot ID/ })).toBeRequired();
    });

    it('renders a required connection field with required styling', async () => {
        const requiredConnectionSchema: Parameters<typeof SchemaForm>[0]['schema'] = {
            type: 'object',
            properties: {
                port: { type: 'string', title: 'Required connection' },
            },
            required: ['port'],
            'x-physicalai-ui': [
                {
                    kind: 'connection',
                    label: 'Required connection',
                    bind: { connection: 'port' },
                },
            ],
        };
        render(
            <RobotFormProvider>
                <SchemaForm schema={requiredConnectionSchema} />
            </RobotFormProvider>
        );

        await screen.findByRole('button', { name: /Required connection/ });
        expect(screen.getByRole('img', { name: '(required)' })).toBeVisible();
    });

    it('renders an optional connection field without required styling', async () => {
        const optionalConnectionSchema: Parameters<typeof SchemaForm>[0]['schema'] = {
            type: 'object',
            properties: {
                port: { type: 'string', title: 'Optional connection' },
            },
            'x-physicalai-ui': [
                {
                    kind: 'connection',
                    label: 'Optional connection',
                    bind: { connection: 'port' },
                },
            ],
        };

        render(
            <RobotFormProvider>
                <SchemaForm schema={optionalConnectionSchema} />
            </RobotFormProvider>
        );

        await screen.findByRole('button', { name: /Optional connection/ });
        expect(screen.queryByRole('img', { name: '(required)' })).not.toBeInTheDocument();
    });

    it('renders an error when identify fails', async () => {
        const identifySchema: Parameters<typeof SchemaForm>[0]['schema'] = {
            type: 'object',
            properties: {
                port: { type: 'string', title: 'Port' },
            },
            'x-physicalai-ui': [
                {
                    kind: 'connection',
                    label: 'Select robot',
                    device_discovery: true,
                    identify: true,
                    bind: { connection: 'port' },
                },
            ],
        };
        server.use(
            http.get('/api/robots/catalog/{robot_type}/discover', () => HttpResponse.json([])),
            http.post('/api/robots/catalog/{robot_type}/identify', () =>
                HttpResponse.json({ message: 'Could not reach the robot.' }, { status: 400 })
            )
        );
        const user = userEvent.setup();

        render(
            <RobotFormProvider>
                <SchemaForm schema={identifySchema} />
            </RobotFormProvider>
        );

        await user.click(await screen.findByRole('button', { name: 'Identify' }));

        expect(await screen.findByText(/Identify Failed/)).toBeVisible();
        expect(screen.getByText(/Could not reach the robot\./)).toBeVisible();
    });

    it('shows a permission denied message when identify fails with a serial permission error', async () => {
        const identifySchema: Parameters<typeof SchemaForm>[0]['schema'] = {
            type: 'object',
            properties: {
                port: { type: 'string', title: 'Port' },
            },
            'x-physicalai-ui': [
                {
                    kind: 'connection',
                    label: 'Select robot',
                    device_discovery: true,
                    identify: true,
                    bind: { connection: 'port' },
                },
            ],
        };
        server.use(
            http.get('/api/robots/catalog/{robot_type}/discover', () => HttpResponse.json([])),
            http.post('/api/robots/catalog/{robot_type}/identify', () =>
                HttpResponse.json({ error_code: 'serial_permission_denied', message: 'No access' }, { status: 403 })
            )
        );
        const user = userEvent.setup();

        render(
            <RobotFormProvider>
                <SchemaForm schema={identifySchema} />
            </RobotFormProvider>
        );

        await user.click(await screen.findByRole('button', { name: 'Identify' }));

        expect(await screen.findByText(/Permission Denied/)).toBeVisible();
    });

    it('identifies the IP address configured in an IP address field', async () => {
        const requests: { robotType: string; payload: unknown }[] = [];
        const ipAddressSchema: Parameters<typeof SchemaForm>[0]['schema'] = {
            type: 'object',
            properties: {
                connection_string: { type: 'string', title: 'Robot IP address' },
            },
            required: ['connection_string'],
            'x-physicalai-ui': [
                {
                    kind: 'ip_address',
                    name: 'connection_string',
                    label: 'Robot IP address',
                    identify: true,
                },
            ],
        };
        server.use(
            http.post('/api/robots/catalog/{robot_type}/identify', async ({ params, request }) => {
                requests.push({ robotType: params.robot_type, payload: await request.json() });
                return new HttpResponse(null, { status: 200 });
            })
        );
        const user = userEvent.setup();

        render(
            <RobotFormProvider robot={{ type: 'Trossen_WidowXAI_Follower', name: '', payload: {} }}>
                <SchemaForm schema={ipAddressSchema} />
            </RobotFormProvider>
        );

        const identify = screen.getByRole('button', { name: 'Identify' });
        expect(identify).toBeDisabled();

        await user.type(screen.getByRole('textbox', { name: /Robot IP address/ }), ' 192.168.1.100 ');
        await user.click(identify);

        await waitFor(() =>
            expect(requests).toEqual([
                {
                    robotType: 'Trossen_WidowXAI_Follower',
                    payload: { connection_string: '192.168.1.100' },
                },
            ])
        );
    });

    it('renders an optional IP address field without required styling', () => {
        const ipAddressSchema: Parameters<typeof SchemaForm>[0]['schema'] = {
            type: 'object',
            properties: {
                connection_string: { type: 'string', title: 'Robot IP address' },
            },
            'x-physicalai-ui': [
                {
                    kind: 'ip_address',
                    name: 'connection_string',
                },
            ],
        };

        render(
            <RobotFormProvider robot={{ type: 'Trossen_WidowXAI_Follower', name: '', payload: {} }}>
                <SchemaForm schema={ipAddressSchema} />
            </RobotFormProvider>
        );

        expect(screen.getByRole('textbox', { name: 'IP address' })).not.toBeRequired();
    });

    it('identifies each bimanual IP address through the configured single-arm robot type', async () => {
        const requests: { robotType: string; payload: unknown }[] = [];
        const bimanualIpAddressSchema: Parameters<typeof SchemaForm>[0]['schema'] = {
            type: 'object',
            properties: {
                connection_string_left: { type: 'string', title: 'Left arm IP address' },
                connection_string_right: { type: 'string', title: 'Right arm IP address' },
            },
            required: ['connection_string_left', 'connection_string_right'],
            'x-physicalai-ui': [
                {
                    kind: 'ip_address',
                    name: 'connection_string_left',
                    label: 'Left arm IP address',
                    identify: true,
                    identify_robot_type: 'Trossen_WidowXAI_Follower',
                },
                {
                    kind: 'ip_address',
                    name: 'connection_string_right',
                    label: 'Right arm IP address',
                    identify: true,
                    identify_robot_type: 'Trossen_WidowXAI_Follower',
                },
            ],
        };
        server.use(
            http.post('/api/robots/catalog/{robot_type}/identify', async ({ params, request }) => {
                requests.push({ robotType: params.robot_type, payload: await request.json() });
                return new HttpResponse(null, { status: 200 });
            })
        );
        const user = userEvent.setup();

        render(
            <RobotFormProvider robot={{ type: 'Trossen_Bimanual_WidowXAI_Follower', name: '', payload: {} }}>
                <SchemaForm schema={bimanualIpAddressSchema} />
            </RobotFormProvider>
        );

        await user.type(screen.getByRole('textbox', { name: /Left arm IP address/ }), '192.168.1.100');
        await user.type(screen.getByRole('textbox', { name: /Right arm IP address/ }), '192.168.1.101');
        await user.click(screen.getAllByRole('button', { name: 'Identify' })[0]);
        await user.click(screen.getAllByRole('button', { name: 'Identify' })[1]);

        await waitFor(() =>
            expect(requests).toEqual([
                {
                    robotType: 'Trossen_WidowXAI_Follower',
                    payload: { connection_string: '192.168.1.100' },
                },
                {
                    robotType: 'Trossen_WidowXAI_Follower',
                    payload: { connection_string: '192.168.1.101' },
                },
            ])
        );
    });

    it('stores serial-capable device values in their respective payload fields', async () => {
        server.use(
            http.get('/api/robots/catalog/{robot_type}/discover', () =>
                HttpResponse.json([{ serial_number: '00000000050C', connection_string: '/dev/ttyACM0' }])
            )
        );
        const user = userEvent.setup();

        render(
            <RobotFormProvider>
                <SchemaForm schema={rebotSchema} />
                <Payload />
            </RobotFormProvider>
        );

        await user.click(await screen.findByRole('button', { name: /Select robot/ }));
        await user.click(screen.getByRole('option', { name: '00000000050C' }));
        await user.keyboard('{Escape}');

        expect(await screen.findByRole('status')).toHaveTextContent(
            JSON.stringify({ connection_string: '/dev/ttyACM0', serial_number: '00000000050C' })
        );
    });

    it('stores the raw connection string when selecting a device without a serial number', async () => {
        server.use(
            http.get('/api/robots/catalog/{robot_type}/discover', () =>
                HttpResponse.json([{ serial_number: null, connection_string: '/dev/ttyUSB0' }])
            )
        );
        const user = userEvent.setup();

        render(
            <RobotFormProvider>
                <SchemaForm schema={lerobotSO101Schema} />
                <Payload />
            </RobotFormProvider>
        );

        await user.click(await screen.findByRole('button', { name: /Select robot/ }));
        await user.click(screen.getByRole('option', { name: 'No serial number' }));

        expect(await screen.findByRole('status')).toHaveTextContent(
            JSON.stringify({ cameras: {}, port: '/dev/ttyUSB0' })
        );
    });

    it('stores a manually entered value that matches a discovered connection', async () => {
        server.use(
            http.get('/api/robots/catalog/{robot_type}/discover', () =>
                HttpResponse.json([{ serial_number: null, connection_string: '/dev/ttyUSB0' }])
            )
        );
        const user = userEvent.setup();

        render(
            <RobotFormProvider>
                <SchemaForm schema={lerobotSO101Schema} />
                <Payload />
            </RobotFormProvider>
        );

        await user.click(await screen.findByRole('button', { name: /Select robot/ }));
        await user.type(screen.getByRole('searchbox', { name: /Select robot/ }), '/dev/ttyUSB0');
        await user.keyboard('{Escape}');

        expect(screen.getByRole('status')).toHaveTextContent(JSON.stringify({ cameras: {}, port: '/dev/ttyUSB0' }));
    });

    it('does not store NaN when a numeric field is cleared', async () => {
        const schema: Parameters<typeof SchemaForm>[0]['schema'] = {
            type: 'object',
            properties: {
                baud_rate: { type: 'integer', title: 'Baud Rate', default: 115200 },
            },
        };
        const user = userEvent.setup();

        render(
            <RobotFormProvider>
                <SchemaForm schema={schema} />
                <Payload />
            </RobotFormProvider>
        );

        await user.clear(screen.getByRole('spinbutton', { name: 'Baud Rate' }));

        expect(screen.getByRole('status')).toHaveTextContent('{}');
    });

    it('skips unsupported dictionary fields', async () => {
        render(
            <RobotFormProvider>
                <SchemaForm schema={lerobotSO101Schema} />
            </RobotFormProvider>
        );

        expect(screen.getByRole('button', { name: /Select robot/ })).toBeVisible();
        expect(screen.queryByRole('heading', { name: 'Cameras' })).not.toBeInTheDocument();
        expect(screen.queryByRole('textbox', { name: 'Cameras' })).not.toBeInTheDocument();
    });

    it('renders connection pickers from referenced nested object schemas', () => {
        render(
            <RobotFormProvider>
                <SchemaForm schema={bimanualRebotSchema} />
                <Payload />
            </RobotFormProvider>
        );

        expect(screen.getByRole('heading', { name: 'Left Arm Config' })).toBeVisible();
        expect(screen.getByRole('heading', { name: 'Right Arm Config' })).toBeVisible();
        expect(screen.queryByRole('switch', { name: 'Show advanced options' })).not.toBeInTheDocument();
        expect(screen.queryAllByRole('textbox', { name: 'Can Adapter' })).toHaveLength(0);
        expect(screen.getAllByRole('button', { name: /Select robot/ })).toHaveLength(2);
    });

    it('hides a section when all of its fields are advanced configuration fields', () => {
        const schema: Parameters<typeof SchemaForm>[0]['schema'] = {
            type: 'object',
            properties: {
                calibration: {
                    type: 'object',
                    title: 'Calibration',
                    default: {},
                    'x-physicalai-ui': { advanced_configuration: true },
                    properties: {
                        offset: {
                            type: 'integer',
                            title: 'Offset',
                            default: 0,
                        },
                    },
                },
            },
            'x-physicalai-ui': [
                {
                    kind: 'section',
                    id: 'calibration',
                    title: 'Calibration',
                    items: [{ kind: 'field', name: 'calibration' }],
                },
            ],
        };

        render(
            <RobotFormProvider>
                <SchemaForm schema={schema} />
            </RobotFormProvider>
        );

        expect(screen.queryByRole('heading', { name: 'Calibration' })).not.toBeInTheDocument();
    });

    it('renders defaulted fields unless they are advanced configuration fields', () => {
        const schema: Parameters<typeof SchemaForm>[0]['schema'] = {
            type: 'object',
            properties: {
                baud_rate: { type: 'integer', title: 'Baud Rate', default: 115200 },
                torque_enabled: { type: 'boolean', title: 'Torque Enabled', default: true },
            },
        };

        render(
            <RobotFormProvider>
                <SchemaForm schema={schema} />
            </RobotFormProvider>
        );

        expect(screen.getByRole('spinbutton', { name: 'Baud Rate' })).toBeVisible();
        expect(screen.getByRole('switch', { name: 'Torque Enabled' })).toBeChecked();
    });

    it('keeps advanced configuration fields hidden when the toggle is hidden', () => {
        const schema: Parameters<typeof SchemaForm>[0]['schema'] = {
            type: 'object',
            properties: {
                connection_string: { type: 'string', title: 'Connection String' },
                use_ros: {
                    type: 'boolean',
                    title: 'Use ROS',
                    default: false,
                    'x-physicalai-ui': { advanced_configuration: true },
                },
            },
        };

        render(
            <RobotFormProvider>
                <SchemaForm schema={schema} />
            </RobotFormProvider>
        );

        expect(screen.getByRole('textbox', { name: 'Connection String' })).toBeVisible();
        expect(screen.queryByRole('switch', { name: 'Show advanced options' })).not.toBeInTheDocument();
        expect(screen.queryByRole('switch', { name: 'Use ROS' })).not.toBeInTheDocument();
    });

    it('keeps advanced configuration field defaults in the payload', () => {
        const schema: Parameters<typeof SchemaForm>[0]['schema'] = {
            type: 'object',
            properties: {
                connection_string: { type: 'string', title: 'Connection String' },
                control_mode: {
                    type: 'string',
                    title: 'Control Mode',
                    default: 'position',
                    'x-physicalai-ui': { advanced_configuration: true },
                },
            },
        };

        render(
            <RobotFormProvider>
                <SchemaForm schema={schema} />
                <Payload />
            </RobotFormProvider>
        );

        expect(screen.getByRole('status')).toHaveTextContent(JSON.stringify({ control_mode: 'position' }));
    });
});
