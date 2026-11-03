import { Activity, ReactNode, Suspense, TransitionEvent, useId, useState } from 'react';

import { ActionButton, Flex, Grid, Loading, minmax, Tooltip, TooltipTrigger, View } from '@geti-ui/ui';
import { ChevronDoubleLeft, ChevronDoubleRight } from '@geti-ui/ui/icons';

import classes from './collapsible-sidebar-layout.module.css';

type SidebarState = 'expanded' | 'collapsing' | 'collapsed';

const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const CenteredLoading = () => {
    return (
        <Flex width='100%' height='100%' alignItems='center' justifyContent='center'>
            <Loading mode='inline' />
        </Flex>
    );
};

interface CollapsibleSidebarLayoutProps {
    toolbar: ReactNode;
    sidebar: ReactNode;
    children: ReactNode;
}

export const CollapsibleSidebarLayout = ({ toolbar, sidebar, children }: CollapsibleSidebarLayoutProps) => {
    const contentId = useId();
    const [state, setState] = useState<SidebarState>('expanded');

    const isCollapsed = state !== 'expanded';
    const label = isCollapsed ? 'Expand sidebar' : 'Collapse sidebar';
    const activityMode = state === 'collapsed' ? 'hidden' : 'visible';

    const toggle = () => {
        if (isCollapsed) {
            setState('expanded');
        } else {
            setState(prefersReducedMotion() ? 'collapsed' : 'collapsing');
        }
    };

    const handleTransitionEnd = (event: TransitionEvent<HTMLElement>) => {
        if (event.target === event.currentTarget && event.propertyName === 'width') {
            setState((current) => (current === 'collapsing' ? 'collapsed' : current));
        }
    };

    return (
        <Grid columns={['auto', minmax(0, '1fr')]} rows={[minmax(0, '1fr')]} height='100%' minHeight={0}>
            <aside className={classes.sidebar} data-collapsed={isCollapsed} onTransitionEnd={handleTransitionEnd}>
                <TooltipTrigger>
                    <ActionButton
                        isQuiet
                        aria-label={label}
                        aria-expanded={!isCollapsed}
                        aria-controls={contentId}
                        onPress={toggle}
                    >
                        {isCollapsed ? <ChevronDoubleRight /> : <ChevronDoubleLeft />}
                    </ActionButton>
                    <Tooltip>{label}</Tooltip>
                </TooltipTrigger>
                <View UNSAFE_className={classes.toolbar}>
                    <Activity mode={activityMode}>{toolbar}</Activity>
                </View>
                <View id={contentId} UNSAFE_className={classes.content}>
                    <Activity mode={activityMode}>
                        <Suspense fallback={<CenteredLoading />}>{sidebar}</Suspense>
                    </Activity>
                </View>
            </aside>
            <View backgroundColor='gray-50' height='100%' minHeight={0} minWidth={0} overflow='auto'>
                <Suspense fallback={<CenteredLoading />}>{children}</Suspense>
            </View>
        </Grid>
    );
};
