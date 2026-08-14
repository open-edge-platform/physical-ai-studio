import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Tabs } from 'react-aria-components';
import { describe, expect, it, vi } from 'vitest';

import { SchemaDatasetOutput } from '../../api/openapi-spec';
import { render } from '../../test-utils/render';
import { DatasetTabs } from './dataset-tabs';

const PROJECT_ID = 'project-id';
const FIRST_DATASET_ID = 'first-dataset-id';
const SECOND_DATASET_ID = 'second-dataset-id';

const datasets: SchemaDatasetOutput[] = [
    {
        id: FIRST_DATASET_ID,
        name: 'First dataset',
        default_task: 'first task',
        project_id: PROJECT_ID,
        environment_id: 'first-environment-id',
    },
    {
        id: SECOND_DATASET_ID,
        name: 'Second dataset',
        default_task: 'second task',
        project_id: PROJECT_ID,
        environment_id: 'second-environment-id',
    },
];

const renderDatasetTabs = (selectedDatasetId = FIRST_DATASET_ID) => {
    return render(
        <Tabs selectedKey={selectedDatasetId}>
            <DatasetTabs datasets={datasets} selectedDatasetId={selectedDatasetId} />
        </Tabs>,
        {
            route: `/projects/${PROJECT_ID}/datasets/${selectedDatasetId}`,
            path: '/projects/:project_id/datasets/:dataset_id',
        }
    );
};

describe('DatasetTabs', () => {
    it('renders the action menu only for the selected dataset tab', async () => {
        renderDatasetTabs();

        const firstTab = await screen.findByRole('tab', { name: /first dataset/i });
        const secondTab = screen.getByRole('tab', { name: /second dataset/i });

        expect(firstTab).toHaveAttribute('data-selected');
        expect(firstTab).toHaveTextContent('First dataset');
        expect(secondTab).not.toHaveAttribute('data-selected');
        expect(secondTab).toHaveTextContent('Second dataset');
        expect(firstTab.querySelector('svg')).not.toBeNull();
        expect(secondTab.querySelector('svg')).toBeNull();
    });

    it('exports the selected dataset from its contextual menu', async () => {
        const user = userEvent.setup();
        const open = vi.spyOn(window, 'open').mockImplementation(() => null);

        renderDatasetTabs();

        const selectedTab = await screen.findByRole('tab', { name: /first dataset/i });
        const menuTrigger = selectedTab.querySelector('[aria-haspopup="true"]');

        expect(menuTrigger).not.toBeNull();
        if (menuTrigger === null) {
            return;
        }

        await user.click(menuTrigger);
        await user.click(await screen.findByRole('menuitem', { name: 'Export dataset' }));

        expect(open).toHaveBeenCalledWith(
            `http://localhost:7860/api/dataset/${FIRST_DATASET_ID}/download`,
            '_blank',
            'noopener,noreferrer'
        );
    });

    it('opens the add and import menu', async () => {
        const user = userEvent.setup();
        renderDatasetTabs();

        const addDatasetButton = await screen.findByRole('button', { name: 'Add dataset' });
        await user.click(addDatasetButton);

        expect(addDatasetButton).toHaveAttribute('aria-expanded', 'true');
    });
});
