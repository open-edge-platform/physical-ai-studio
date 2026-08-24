import { ActionButton, Content, Dialog, DialogTrigger, Divider, Flex, Header, Heading, View } from '@geti-ui/ui';

import { paths } from '../../../router';
import { ProjectThumbnail } from '../project-thumbnail/project-thumbnail';
import { useProject } from './../use-project';
import { PanelLink } from './page-link/panel-link.component';

import classes from './project-menu.module.css';

interface SelectedProjectProps {
    projectId: string;
    name: string;
}

const SelectedProjectButton = ({ projectId, name }: SelectedProjectProps) => {
    return (
        <ActionButton
            aria-label={`Selected project ${name}`}
            isQuiet
            staticColor='white'
            UNSAFE_className={classes.selectedProjectButton}
        >
            <View marginEnd='size-100' UNSAFE_className={classes.thumbnailWrapper}>
                <ProjectThumbnail projectId={projectId} name={name} size={40} />
            </View>
            <View margin={'size-50'}>{name}</View>
        </ActionButton>
    );
};

export const ProjectMenu = () => {
    const project = useProject();

    const selectedProjectName = project.name;

    return (
        <DialogTrigger type='popover' hideArrow>
            <SelectedProjectButton projectId={project.id} name={selectedProjectName} />

            <Dialog width={'size-4600'} UNSAFE_className={classes.dialog}>
                <Header marginTop={'size-250'}>
                    <Flex
                        direction={'column'}
                        justifyContent={'center'}
                        width={'100%'}
                        alignItems={'center'}
                        gap={'size-100'}
                    >
                        <View UNSAFE_className={classes.thumbnailWrapper}>
                            <ProjectThumbnail projectId={project.id} name={selectedProjectName} size={70} />
                        </View>

                        <Heading level={2} marginBottom={0}>
                            {selectedProjectName}
                        </Heading>
                    </Flex>
                </Header>
                <Content UNSAFE_className={classes.panelContent}>
                    <Divider size={'S'} marginTop={'size-200'} />

                    <PanelLink href={paths.projects.index({})}>Projects</PanelLink>
                    <PanelLink href={paths.settings.index({})}>Settings</PanelLink>
                    <PanelLink href={paths.openapi({})}>OpenAPI Spec</PanelLink>
                </Content>
            </Dialog>
        </DialogTrigger>
    );
};
