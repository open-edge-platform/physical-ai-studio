import { useState } from 'react';

import { toast } from '@geti-ui/ui';

import { $api } from '../../api/client';
import { getApiErrorMessage, isResourceInUseError } from '../../api/errors';
import { useRestartState } from './restart-state';

export const usePluginsQuery = () => {
    return $api.useSuspenseQuery('get', '/api/plugins', {
        meta: { skipInvalidation: true },
    });
};

export const useInstallPluginMutation = () => {
    return $api.useMutation('post', '/api/plugins/{plugin_id}/install', {
        meta: { invalidates: [['get', '/api/plugins']] },
    });
};

export const useUninstallPluginMutation = () => {
    return $api.useMutation('post', '/api/plugins/{plugin_id}/uninstall', {
        meta: { invalidates: [['get', '/api/plugins']] },
    });
};

export const useRestartServerMutation = () => {
    const { restartServer, restartStatus } = useRestartState();
    const isPending = restartStatus !== 'idle' && restartStatus !== 'failed';
    return { restartServer, restartStatus, isPending };
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
            await installMutation.mutateAsync({ params: { path: { plugin_id: pluginId } } });
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
            if (isResourceInUseError(error)) {
                toast.info(getApiErrorMessage(error) ?? 'This plugin is in use and cannot be uninstalled.');
                return;
            }
            toast.negative(getApiErrorMessage(error) ?? 'Failed to uninstall the plugin.');
        } finally {
            setBusyId(undefined);
        }
    };

    return { isBusy, busyId, restartRequired, install, uninstall };
};
