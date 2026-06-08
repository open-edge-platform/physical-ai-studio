import type { DockviewTheme } from 'dockview-react';

import styles from './theme.module.scss';

/**
 * Physical AI Studio custom DockView theme
 *
 * This theme integrates DockView with our Spectrum-based design system,
 * providing a consistent dark theme with compact layout and smooth interactions.
 *
 * Key features:
 * - Grayscale color palette using Spectrum CSS variables
 * - Rounded corners matching app-wide border radius
 * - Smooth tab animations for better UX
 * - Compact layout with no gaps between panels
 * - Hover-visible sash/splitters for resizing
 *
 * To customize colors, edit theme.module.scss
 */
export const physicalAiTheme: DockviewTheme = {
    name: 'physical-ai',
    className: styles.dockviewThemePhysicalAi,
    colorScheme: 'dark',

    // Smooth tab animations when reordering
    tabAnimation: 'smooth',

    // Line-style insertion indicator (clean, modern)
    dndTabIndicator: 'line',

    // No gap between panels (compact layout)
    gap: 0,

    // Match tab container height
    edgeGroupCollapsedSize: 40,

    // Accent color border on drop zones
    dndOverlayBorder: '2px solid var(--energy-blue)',

    // Simple tab group indicator (no wrap effect)
    tabGroupIndicator: 'none',

    // Overlay mounts to root for better z-index control
    dndOverlayMounting: 'absolute',

    // Drop overlay covers entire group (simpler UX)
    dndPanelOverlay: 'group',
};
