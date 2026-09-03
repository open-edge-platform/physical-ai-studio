import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { describe, expect, it } from 'vitest';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { FormFields, RobotType } from './form';
import { RobotFormProvider } from './provider';

const so101FollowerDefinition = {
    type: 'SO101_Follower',
    display_name: 'SO101 Follower',
    role: 'follower',
    urdf_path: '/api/robots/catalog/SO101_Follower/urdf',
    package_map: {},
    joint_map: {},
} as const;

const so101LeaderDefinition = {
    type: 'SO101_Leader',
    display_name: 'SO101 Leader',
    role: 'leader',
    urdf_path: '/api/robots/catalog/SO101_Leader/urdf',
    package_map: {},
    joint_map: {},
} as const;

const useCatalogMock = () => {
    server.use(
        http.get('/api/robots/catalog', () => HttpResponse.json([so101FollowerDefinition, so101LeaderDefinition])),
        http.get('/api/robots/catalog/{robot_type}/schema', () =>
            HttpResponse.json({ type: 'object', properties: {}, required: [] })
        )
    );
};

const renderRobotTypeAndFields = () =>
    render(
        <RobotFormProvider>
            <RobotType />
            <FormFields />
        </RobotFormProvider>
    );

const selectType = async (user: ReturnType<typeof userEvent.setup>, name: string) => {
    await user.click(await screen.findByRole('button', { name: /Robot type/ }));
    await user.click(await screen.findByRole('option', { name }));
};

describe('RobotType and FormFields', () => {
    it('auto-focuses the name field on mount', async () => {
        useCatalogMock();

        renderRobotTypeAndFields();

        expect(await screen.findByRole('textbox', { name: /Robot name/ })).toHaveFocus();
    });

    it('prefills the robot name with the selected type display name when the name is empty', async () => {
        useCatalogMock();
        const user = userEvent.setup();

        renderRobotTypeAndFields();
        await selectType(user, 'SO101 Follower');

        expect(await screen.findByRole('textbox', { name: /Robot name/ })).toHaveValue('SO101 Follower');
    });

    it('overwrites the name when it still matches the previously suggested type name', async () => {
        useCatalogMock();
        const user = userEvent.setup();

        renderRobotTypeAndFields();
        await selectType(user, 'SO101 Follower');
        await selectType(user, 'SO101 Leader');

        expect(await screen.findByRole('textbox', { name: /Robot name/ })).toHaveValue('SO101 Leader');
    });

    it('focuses the name field after a type change that prefills the name', async () => {
        useCatalogMock();
        const user = userEvent.setup();

        renderRobotTypeAndFields();
        await selectType(user, 'SO101 Follower');
        await selectType(user, 'SO101 Leader');

        expect(await screen.findByRole('textbox', { name: /Robot name/ })).toHaveFocus();
    });

    it('does not overwrite a custom name when changing the type', async () => {
        useCatalogMock();
        const user = userEvent.setup();

        renderRobotTypeAndFields();
        await selectType(user, 'SO101 Follower');

        const nameField = await screen.findByRole('textbox', { name: /Robot name/ });
        await user.clear(nameField);
        await user.type(nameField, 'My robot');

        await selectType(user, 'SO101 Leader');

        expect(await screen.findByRole('textbox', { name: /Robot name/ })).toHaveValue('My robot');
    });

    it('does not overwrite an already-set robot name when changing the type', async () => {
        useCatalogMock();
        const user = userEvent.setup();

        render(
            <RobotFormProvider robot={{ type: 'SO101_Follower', name: 'My arm', payload: {} }}>
                <RobotType />
                <FormFields />
            </RobotFormProvider>
        );

        const nameField = await screen.findByRole('textbox', { name: /Robot name/ });
        expect(nameField).toHaveValue('My arm');

        await selectType(user, 'SO101 Follower');

        expect(nameField).toHaveValue('My arm');
    });
});
