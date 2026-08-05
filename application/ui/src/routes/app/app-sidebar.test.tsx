import { Tabs } from '@geti-ui/ui';
import { screen } from '@testing-library/react';

import { render } from '../../test-utils/render';
import { AppSidebar } from './app-sidebar';
import { disabledNavItemKeys, navItems } from './nav-items';

const renderSidebar = (selectedKey = 'projects') =>
    render(
        <Tabs
            orientation='vertical'
            aria-label='Main navigation'
            selectedKey={selectedKey}
            disabledKeys={disabledNavItemKeys}
        >
            <AppSidebar />
        </Tabs>,
        { route: '/projects', path: '/projects' }
    );

describe('AppSidebar', () => {
    it('renders all nav item labels', () => {
        renderSidebar();

        navItems.forEach((item) => {
            expect(screen.getByText(item.label)).toBeInTheDocument();
        });
    });

    it('selects Projects when the route is /projects', () => {
        renderSidebar('projects');

        expect(screen.getByRole('tab', { name: 'Projects' })).toHaveAttribute('aria-selected', 'true');
    });

    it('marks the enabled items as links, disabled are not visible', () => {
        renderSidebar();

        const projectsTab = screen.getByRole('tab', { name: 'Projects' });
        expect(projectsTab).toHaveAttribute('href', '/projects');

        navItems
            .filter((item) => !item.enabled)
            .forEach((item) => {
                expect(screen.queryByRole('tab', { name: item.label })).not.toBeInTheDocument();
            });
    });
});
