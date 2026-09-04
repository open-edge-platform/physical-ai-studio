import { featureFlags } from '../../config/feature-flags';
import { PluginsView } from '../../features/plugins/plugins';
import { PluginsDisabledView } from '../../features/plugins/plugins-disabled';

export const Plugins = () => {
    return featureFlags.plugins ? <PluginsView /> : <PluginsDisabledView />;
};
