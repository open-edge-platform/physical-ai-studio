import { screen } from '@testing-library/react';
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
                can_adapter: { type: 'string', title: 'Can Adapter', default: 'damiao' },
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

    it('renders defaulted UI-required fields without showing default fields', () => {
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

        await user.click(await screen.findByRole('button', { name: 'Select robot' }));
        await user.click(screen.getByRole('option', { name: '00000000050C' }));
        await user.keyboard('{Escape}');

        expect(screen.getByRole('status')).toHaveTextContent(
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

        await user.click(await screen.findByRole('button', { name: 'Select robot' }));
        await user.click(screen.getByRole('option', { name: 'No serial number' }));

        expect(screen.getByRole('status')).toHaveTextContent(JSON.stringify({ cameras: {}, port: '/dev/ttyUSB0' }));
    });

    it('skips unsupported dictionary fields', async () => {
        const user = userEvent.setup();

        render(
            <RobotFormProvider>
                <SchemaForm schema={lerobotSO101Schema} />
            </RobotFormProvider>
        );

        expect(screen.getByRole('button', { name: 'Select robot' })).toBeVisible();
        expect(screen.queryByRole('heading', { name: 'Cameras' })).not.toBeInTheDocument();
        expect(screen.queryByRole('textbox', { name: 'Cameras' })).not.toBeInTheDocument();

        await user.click(screen.getByRole('switch', { name: 'Show default fields' }));

        expect(screen.queryByRole('heading', { name: 'Cameras' })).not.toBeInTheDocument();
        expect(screen.queryByRole('textbox', { name: 'Cameras' })).not.toBeInTheDocument();
    });

    it('renders connection pickers from referenced nested object schemas', async () => {
        const user = userEvent.setup();

        render(
            <RobotFormProvider>
                <SchemaForm schema={bimanualRebotSchema} />
                <Payload />
            </RobotFormProvider>
        );

        expect(screen.getByRole('heading', { name: 'Left Arm Config' })).toBeVisible();
        expect(screen.getByRole('heading', { name: 'Right Arm Config' })).toBeVisible();
        expect(screen.queryAllByRole('textbox', { name: 'Can Adapter' })).toHaveLength(0);

        await user.click(screen.getByRole('switch', { name: 'Show default fields' }));
        const canAdapters = screen.getAllByRole('textbox', { name: 'Can Adapter' });
        expect(canAdapters).toHaveLength(2);
        expect(canAdapters[0]).toHaveValue('damiao');
        expect(canAdapters[1]).toHaveValue('damiao');

        expect(screen.getAllByRole('button', { name: 'Select robot' })).toHaveLength(2);
    });

    it('hides a section when all of its fields are hidden default fields', () => {
        const schema: Parameters<typeof SchemaForm>[0]['schema'] = {
            type: 'object',
            properties: {
                calibration: {
                    type: 'object',
                    title: 'Calibration',
                    default: {},
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
});
