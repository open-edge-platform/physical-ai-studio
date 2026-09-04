import { useState } from 'react';

import { toast } from '@geti-ui/ui';

import { $api } from '../../api/client';
import { getApiErrorMessage } from '../../api/errors';
import { useRestartState } from '../system/restart-state';

export const usePluginsQuery = () => {
    return $api.useSuspenseQuery('get', '/api/plugins', {
        meta: { skipInvalidation: true },
    });
};

export const useInstallPluginMutation = () => {
    return $api.useMutation('post', '/api/plugins', {
        meta: { invalidates: [['get', '/api/plugins']] },
    });
};

export const useUninstallPluginMutation = () => {
    return $api.useMutation('delete', '/api/plugins/{plugin_id}', {
        meta: { invalidates: [['get', '/api/plugins']] },
    });
};

export const usePluginActions = () => {
    const installMutation = useInstallPluginMutation();
    const uninstallMutation = useUninstallPluginMutation();
    const { restartRequired, triggerRestartRequired, openRestartPrompt } = useRestartState();
    const [busyId, setBusyId] = useState<string | undefined>(undefined);

    const isBusy = busyId !== undefined;

    const install = async (pluginId: string) => {
        setBusyId(pluginId);
        try {
            await installMutation.mutateAsync({ body: { plugin_id: pluginId } });
            triggerRestartRequired();
            openRestartPrompt();
        } catch (error) {
            toast.negative(getApiErrorMessage(error) ?? 'Failed to install the plugin.');
        } finally {
            setBusyId(undefined);
        }
    };

    const uninstall = async (pluginId: string) => {
        setBusyId(pluginId);
        try {
            await uninstallMutation.mutateAsync({ params: { path: { plugin_id: pluginId } } });
            triggerRestartRequired();
            openRestartPrompt();
        } catch (error) {
            toast.negative(getApiErrorMessage(error) ?? 'Failed to uninstall the plugin.');
        } finally {
            setBusyId(undefined);
        }
    };

    return { isBusy, busyId, restartRequired, install, uninstall };
};
