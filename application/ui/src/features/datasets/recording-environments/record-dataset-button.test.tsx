import { rs } from '@rstest/core';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { useLocation } from 'react-router';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { getMockedEnvironmentOutput } from '../../../test-utils/mocks/mock-environment';
import { getMockedRecordingEnvironment } from '../../../test-utils/mocks/mock-recording-environment';
import { render } from '../../../test-utils/render';
import { RecordDatasetButton } from './record-dataset-button';

const TestPage = ({ label }: { label: string }) => {
    const location = useLocation();
    return (
        <>
            <div>Path: {location.pathname}</div>
            <RecordDatasetButton project_id='project-1' dataset_id='dataset-1'>
                {label}
            </RecordDatasetButton>
        </>
    );
};

const renderButton = (label: string) =>
    render(<TestPage label={label} />, {
        route: '/projects/project-1/datasets/dataset-1',
        path: '/projects/:project_id/datasets/:dataset_id/*',
    });

describe('RecordDatasetButton', () => {
    it.each(['Start recording', 'Add Episode'])(
        'prompts before navigating from %s and stays on cancel',
        async (label) => {
            const user = userEvent.setup();
            server.use(
                http.get('/api/projects/{project_id}/environments', () =>
                    HttpResponse.json([
                        getMockedEnvironmentOutput(),
                        getMockedEnvironmentOutput({ id: 'environment-2', name: 'Second environment' }),
                    ])
                ),
                http.get('/api/dataset/{dataset_id}/recording-environments', () => HttpResponse.json([]))
            );
            renderButton(label);
            expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
            await user.click(screen.getByRole('button', { name: label }));
            expect(await screen.findByRole('dialog')).toBeInTheDocument();
            expect(screen.getByText('Path: /projects/project-1/datasets/dataset-1')).toBeInTheDocument();
            await user.click(screen.getByRole('button', { name: 'Cancel' }));
            expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
            expect(screen.getByRole('button', { name: label })).toBeEnabled();
            expect(screen.getByText('Path: /projects/project-1/datasets/dataset-1')).toBeInTheDocument();
        }
    );

    it('associates the only environment before navigating to the recording page', async () => {
        const user = userEvent.setup();
        const create = rs.fn();
        server.use(
            http.get('/api/projects/{project_id}/environments', () =>
                HttpResponse.json([getMockedEnvironmentOutput()])
            ),
            http.get('/api/dataset/{dataset_id}/recording-environments', () => HttpResponse.json([])),
            http.post('/api/dataset/{dataset_id}/recording-environments', () => {
                create();
                return HttpResponse.json(getMockedRecordingEnvironment(), { status: 201 });
            })
        );
        renderButton('Start recording');
        await user.click(screen.getByRole('button', { name: 'Start recording' }));

        expect(await screen.findByText('Path: /projects/project-1/datasets/dataset-1/record')).toBeInTheDocument();
        expect(create).toHaveBeenCalledOnce();
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });
});
