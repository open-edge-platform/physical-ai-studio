import { describe, expect, it } from '@rstest/core';
import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { getMockedRobotCatalogDefinition } from '../../../../test-utils/mocks/mock-robot-catalog-definition';
import { render } from '../../../../test-utils/render';
import { RobotModelsProvider } from '../../robot-models-context';
import { RobotThumbnail } from './robot-thumbnail';

const renderThumbnail = (urdf_path: string | null) => {
    server.use(
        http.get('/api/robots/catalog', () =>
            HttpResponse.json([
                // The spec types urdf_path as a string, but the backend can omit it for a robot without a model.
                getMockedRobotCatalogDefinition({ type: 'SO101_Follower', urdf_path: urdf_path as string }),
            ])
        )
    );

    return render(
        <RobotModelsProvider>
            <RobotThumbnail type='SO101_Follower' />
        </RobotModelsProvider>
    );
};

describe('RobotThumbnail', () => {
    it.each([null, ''])('shows "No preview" instead of loading forever when the URDF path is %j', async (path) => {
        renderThumbnail(path);

        expect(await screen.findByText('No preview')).toBeInTheDocument();
        expect(screen.queryByText('Loading…')).not.toBeInTheDocument();
    });

    it('shows "No preview" when the URDF cannot be loaded', async () => {
        server.use(http.get('/api/robots/catalog/{robot_type}/urdf', () => new HttpResponse(null, { status: 404 })));

        renderThumbnail('http://localhost:7860/api/robots/catalog/SO101_Follower/urdf');

        expect(await screen.findByText('No preview')).toBeInTheDocument();
        expect(screen.queryByText('Loading…')).not.toBeInTheDocument();
    });
});
