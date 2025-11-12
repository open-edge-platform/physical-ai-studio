import { Flex, Heading, Key, Tag, Text, View } from '@geti/ui';
import { clsx } from 'clsx';
import { NavLink } from 'react-router-dom';

import { $api } from '../../../api/client';
import { SchemaProjectInput } from '../../../api/openapi-spec';
import thumbnailUrl from '../../../assets/mocked-project-thumbnail.png';
import { paths } from '../../../router';
import { MenuActions } from './menu-actions.component';

import classes from './project-list.module.scss';

type ProjectCardProps = {
    item: SchemaProjectInput;
    isActive: boolean;
};

const robotNameFromTypeMap: { [key: string]: string } = {
    so100_follower: 'SO-100',
    so101_follower: 'SO-101',
    koch_follower: 'Koch',
    stretch3: 'Stretch 3',
    lekiwi: 'LeKiwi',
    viperx: 'ViperX',
    hope_jr_arm: 'HopeJrArm',
    bi_so100_follower: 'Bi SO-100',
    reachy2: 'Reachy2 Robot',
};

export const ProjectCard = ({ item, isActive }: ProjectCardProps) => {
    const deleteMutation = $api.useMutation('delete', '/api/projects/{project_id}');

    const onAction = (key: Key) => {
        switch (key.toString()) {
            case 'delete':
                if (item.id !== undefined) {
                    deleteMutation.mutate({
                        params: { path: { project_id: item.id } },
                    });
                }
                return;
        }
    };

    return (
        <NavLink to={paths.project.robots.index({ project_id: item.id! })}>
            <Flex UNSAFE_className={clsx({ [classes.card]: true, [classes.activeCard]: isActive })}>
                <View aria-label={'project thumbnail'}>
                    <img src={thumbnailUrl} alt={item.name} />
                </View>

                <View width={'100%'} padding={'size-200'}>
                    <Flex alignItems={'center'} justifyContent={'space-between'}>
                        <Heading level={3}>{item.name}</Heading>
                        <MenuActions onAction={onAction} />
                    </Flex>

                    <Flex marginBottom={'size-200'} gap={'size-50'}>
                        {isActive && item.config && (
                            <Tag
                                withDot={false}
                                text={robotNameFromTypeMap[item.config.robot_type] ?? item.config.robot_type}
                                className={clsx(classes.tag, classes.activeTag)}
                            />
                        )}
                    </Flex>

                    <Flex alignItems={'center'} gap={'size-100'} direction={'row'} wrap='wrap'>
                        {item.updated_at !== undefined && (
                            <Text>• Edited: {new Date(item.updated_at!).toLocaleString()}</Text>
                        )}
                        {item.config && <Text>• Cameras: {item.config.cameras.map((c) => c.name).join(', ')}</Text>}
                        {item.datasets.length > 0 && (
                            <Text>• Datasets: {item.datasets.map((d) => d.name).join(', ')}</Text>
                        )}
                    </Flex>
                </View>
            </Flex>
        </NavLink>
    );
};
