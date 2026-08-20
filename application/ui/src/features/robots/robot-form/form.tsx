import { useState } from 'react';

import { Button, DialogContainer, Flex, Heading, Icon, Text, TextField } from '@geti-ui/ui';
import { ChevronLeft } from '@geti-ui/ui/icons';

import { $api } from '../../../api/client';
import { useProjectId } from '../../../features/projects/use-project';
import { paths } from '../../../router';
import { useRobotCatalogQuery } from '../robot-catalog.hooks';
import { SchemaForm } from './robot-schema/schema-form';
import { BimanualSO101FormFields } from './catalog/bimanual-so101';
import { useRobotForm } from './provider';
import { RobotCatalogDialog } from './robot-catalog-dialog';

export const RobotType = () => {
    const { activeType } = useRobotForm();
    const catalog = useRobotCatalogQuery();
    const [isCatalogOpen, setCatalogOpen] = useState(activeType === undefined);
    const activeRobot = catalog.data.find((entry) => entry.type === activeType);
    return (
        <Flex direction='column' gap='size-100'>
            <Text>Robot type</Text>
            <Button variant='secondary' width='100%' onPress={() => setCatalogOpen(true)}>
                {activeRobot?.display_name ?? 'Select robot type'}
            </Button>
            <DialogContainer type='fullscreen' onDismiss={() => setCatalogOpen(false)}>
                {isCatalogOpen && <RobotCatalogDialog close={() => setCatalogOpen(false)} />}
            </DialogContainer>
        </Flex>
    );
};

export const FormFields = () => {
    const { activeType, name, setName } = useRobotForm();
    return (
        <>
            <TextField isRequired label='Robot name' width='100%' value={name} onChange={setName} />
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
        schema.data && <SchemaForm schema={schema.data as Parameters<typeof SchemaForm>[0]['schema']} />
    );
};

const useRobotCatalogSchema = (robotType: string) => {
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
