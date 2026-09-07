import { $api } from '../../../api/client';

export const useSettingsPatch = () =>
    $api.useMutation('patch', '/api/settings', {
        meta: { invalidates: [['get', '/api/settings']] },
    });
