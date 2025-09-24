import { Flex, Heading, Tag, Text, View, Key } from '@geti/ui';
import { clsx } from 'clsx';
import { NavLink } from 'react-router-dom';

import { SchemaProject } from '../../../api/openapi-spec';
import thumbnailUrl from '../../../assets/mocked-project-thumbnail.png';
import { paths } from '../../../router';
import { MenuActions } from './menu-actions.component';

import classes from './project-list.module.scss';
import { $api } from '../../../api/client';

type ProjectCardProps = {
    item: SchemaProject;
    isActive: boolean;
};

export const ProjectCard = ({ item, isActive }: ProjectCardProps) => {
    const deleteMutation = $api.useMutation("delete", "/api/projects/{project_id}")

    const onAction = (key: Key) => {
        switch(key.toString()) {
            case "delete":
                if (item.id !== undefined){
                    deleteMutation.mutate({
                        params: { path: { project_id: item.id } }
                    })
                }
                return
        }
    }

    return (
        <NavLink to={paths.project.robotConfiguration({ project_id: item.id! })}>
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
                        {isActive && (
                            <Tag withDot={false} text='Active' className={clsx(classes.tag, classes.activeTag)} />
                        )}
                        <Tag withDot={false} text={item.name} className={classes.tag} />
                    </Flex>

                    <Flex alignItems={'center'} gap={'size-100'} direction={'row'} wrap='wrap'>
                        {item.updated_at !== undefined && <Text>• Edited: {new Date(item.updated_at!).toLocaleString()}</Text>}
                        <Text>• Datasets: </Text>
                        <Text>• Cameras: </Text>
                        <Text>
                            • Robots: 
                        </Text>
                    </Flex>
                </View>
            </Flex>
        </NavLink>
    );
};
