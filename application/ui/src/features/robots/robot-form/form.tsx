import { useEffect } from 'react';

import { Button, Flex, Heading, Icon, Item, Picker, TextField } from '@geti-ui/ui';
import { ChevronLeft } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { useRobotCatalogQuery } from '../robot-catalog.hooks';
import { SchemaRobotType } from '../robot-types';
import { BimanualSO101FormFields } from './catalog/bimanual-so101';
import { useRobotForm } from './provider';
import { SchemaForm } from './robot-schema/schema-form';

export const RobotType = () => {
    const { activeType, name, setName } = useRobotForm();
    const { setActiveType } = useRobotForm();
    const catalogQuery = useRobotCatalogQuery();

    return (
        <Picker
            isRequired
            label='Robot type'
            width='100%'
            selectedKey={activeType}
            onSelectionChange={(selected) => {
                if (selected === null) {
                    return;
                }
                const entry = catalogQuery.data.find(({ type }) => type === selected);
                setActiveType(selected.toString());
                if (entry === undefined) {
                    return;
                }
                const previousSuggestedName = catalogQuery.data.find(({ type }) => type === activeType)?.display_name;
                if (name === '' || name === previousSuggestedName) {
                    setName(entry.display_name);
                }
            }}
        >
            {catalogQuery.data.map((entry) => (
                <Item key={entry.type}>{entry.display_name}</Item>
            ))}
        </Picker>
    );
};

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
        <Flex alignItems='center' gap='size-200'>
            <Button
                href={paths.project.robots.index({ project_id })}
                variant='secondary'
                UNSAFE_style={{ border: 'none' }}
            >
                <Icon>
                    <ChevronLeft color='white' fill='white' />
                </Icon>
            </Button>
            <Heading>{heading}</Heading>
        </Flex>
    );
};
