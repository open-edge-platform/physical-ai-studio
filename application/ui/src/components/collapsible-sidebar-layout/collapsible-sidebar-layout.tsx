import { Activity, CSSProperties, ReactNode, Suspense, TransitionEvent, useId, useState } from 'react';

import {
    ActionButton,
    dimensionalValue,
    Flex,
    Grid,
    Loading,
    minmax,
    Tooltip,
    TooltipTrigger,
    View,
    type DimensionValue,
} from '@geti-ui/ui';
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
    toolbar?: ReactNode;
    sidebar: ReactNode;
    children: ReactNode;
    side?: 'left' | 'right';
    width?: DimensionValue;
}

export const CollapsibleSidebarLayout = ({
    toolbar,
    sidebar,
    children,
    side = 'left',
    width = 'size-6000',
}: CollapsibleSidebarLayoutProps) => {
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

    const isRight = side === 'right';

    const CollapseIcon = isRight ? ChevronDoubleRight : ChevronDoubleLeft;
    const ExpandIcon = isRight ? ChevronDoubleLeft : ChevronDoubleRight;

    const sidebarStyle = {
        '--sidebar-expanded-width': dimensionalValue(width),
    } as CSSProperties;

    const aside = (
        <aside
            className={classes.sidebar}
            data-collapsed={isCollapsed}
            style={sidebarStyle}
            onTransitionEnd={handleTransitionEnd}
        >
            <Flex direction='column' height='100%' minHeight={0}>
                <Flex flexShrink={0} justifyContent={isCollapsed ? 'center' : isRight ? 'end' : 'start'}>
                    <TooltipTrigger>
                        <ActionButton
                            isQuiet
                            aria-label={label}
                            aria-expanded={!isCollapsed}
                            aria-controls={contentId}
                            onPress={toggle}
                        >
                            {isCollapsed ? <ExpandIcon /> : <CollapseIcon />}
                        </ActionButton>
                        <Tooltip>{label}</Tooltip>
                    </TooltipTrigger>
                </Flex>
                {toolbar !== undefined && (
                    <View marginTop='size-100' flexShrink={0} UNSAFE_className={classes.toolbar}>
                        <Activity mode={activityMode}>{toolbar}</Activity>
                    </View>
                )}
                <View id={contentId} marginTop='size-100' flexGrow={1} minHeight={0} UNSAFE_className={classes.content}>
                    <Activity mode={activityMode}>
                        <Suspense fallback={<CenteredLoading />}>{sidebar}</Suspense>
                    </Activity>
                </View>
            </Flex>
        </aside>
    );

    return (
        <Grid
            columns={isRight ? [minmax(0, '1fr'), 'auto'] : ['auto', minmax(0, '1fr')]}
            rows={[minmax(0, '1fr')]}
            height='100%'
            minHeight={0}
        >
            {!isRight && aside}
            <View backgroundColor='gray-50' height='100%' minHeight={0} minWidth={0} overflow='auto'>
                <Suspense fallback={<CenteredLoading />}>{children}</Suspense>
            </View>
            {isRight && aside}
        </Grid>
    );
};
