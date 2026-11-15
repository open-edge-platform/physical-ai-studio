import { useEffect } from 'react';

import { View } from '@geti-ui/ui';

/**
 * Native React route for `/projects/:project_id/so101-architecture`.
 * Mounts and activates the SO-101 Model-Based Pick-and-Place Architecture Workbench.
 */
export const SO101Architecture = () => {
    useEffect(() => {
        const toggleBtn = document.getElementById('so101-arch-toggle-btn');
        if (toggleBtn && !toggleBtn.classList.contains('active')) {
            toggleBtn.click();
        }
        return () => {
            const overlay = document.getElementById('so101-workbench-overlay');
            if (overlay) {
                overlay.style.display = 'none';
            }
        };
    }, []);

    return <View height='100%' width='100%' id='so101-architecture-route-host' />;
};
