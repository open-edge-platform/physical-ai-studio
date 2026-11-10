import { Outlet } from 'react-router';

import { AddResourceButton } from '../../components/add-resource-button/add-resource-button';
import { CollapsibleSidebarLayout } from '../../components/collapsible-sidebar-layout/collapsible-sidebar-layout';
import { useProjectId } from '../../features/projects/use-project';
import { RobotsList } from '../../features/robots/robots-list';
import { paths } from '../../paths';

export const Layout = () => {
    const { project_id } = useProjectId();

    return (
        <CollapsibleSidebarLayout
            paddingTop={'size-200'}
            paddingBottom={'size-200'}
            paddingStart={'size-200'}
            paddingEnd={'size-200'}
            toolbar={
                <AddResourceButton to={paths.project.robots.new({ project_id })}>Configure new robot</AddResourceButton>
            }
            sidebar={<RobotsList />}
        >
            <Outlet />
        </CollapsibleSidebarLayout>
    );
};
