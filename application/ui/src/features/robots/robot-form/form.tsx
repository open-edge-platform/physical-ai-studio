import { useEffect } from 'react';

import { TextField } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { FormHeading } from '../../../components/form-heading/form-heading';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../paths';
import { useRobotCatalogQuery } from '../robot-catalog.hooks';
import { SchemaRobotType } from '../robot-types';
import { BimanualSO101FormFields } from './catalog/bimanual-so101';
import { useRobotForm } from './provider';
import { SchemaForm } from './robot-schema/schema-form';

export const FormFields = () => {
    const { activeType, name, setName, nameFieldRef } = useRobotForm();
    const catalogQuery = useRobotCatalogQuery();
    const suggestedName = catalogQuery.data.find(({ type }) => type === activeType)?.display_name;

    useEffect(() => {
        if (name !== '' && name === suggestedName) {
            nameFieldRef.current?.focus();
        }
    }, [name, nameFieldRef, suggestedName]);

    return (
        <>
            <TextField
                isRequired
                label='Robot name'
                width='100%'
                value={name}
                onChange={setName}
                ref={nameFieldRef}
                // eslint-disable-next-line jsx-a11y/no-autofocus
                autoFocus
            />
            {activeType !== undefined && <SelectedRobotFields activeType={activeType} />}
        </>
    );
};

const SelectedRobotFields = ({ activeType }: { activeType: string }) => {
    const schema = useRobotCatalogSchema(activeType);
    const isBimanualSO101 = activeType === 'BimanualSO101_Follower' || activeType === 'BimanualSO101_Leader';
    return isBimanualSO101 ? (
        <BimanualSO101FormFields />
    ) : (
        <SchemaForm schema={schema.data as Parameters<typeof SchemaForm>[0]['schema']} />
    );
};

const useRobotCatalogSchema = (robotType: SchemaRobotType) => {
    return $api.useSuspenseQuery('get', '/api/robots/catalog/{robot_type}/schema', {
        params: { path: { robot_type: robotType } },
    });
};

export const RobotFormHeading = ({ heading }: { heading: string }) => {
    const { project_id } = useProjectId();
    return (
        <FormHeading heading={heading} backTo={paths.project.robots.index({ project_id })} backLabel='Back to robots' />
    );
};
