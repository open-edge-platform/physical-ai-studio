import { Text } from '@geti-ui/ui';

import { $api } from '../../../api/client';
import { SettingsSection } from './settings-section';

import classes from './general-settings.module.css';

const Entry = ({ name, value }: { name: string; value: string }) => {
    return (
        <li>
            <Text>
                {name}: <span className={classes.aboutValue}>{value}</span>
            </Text>
        </li>
    );
};

export const AboutSettings = () => {
    const { data: systemInfo, isError, isPending } = $api.useQuery('get', '/api/system/info');

    return (
        <SettingsSection
            title='Physical AI Studio'
            description='Version and source provenance of this Studio installation.'
            isDirty={false}
            isPending={false}
            saved={false}
            onSave={() => {}}
            showSave={false}
        >
            {isPending ? <Text>Loading Studio information…</Text> : null}
            {isError || !systemInfo ? <Text>Studio version information is unavailable.</Text> : null}
            {systemInfo ? (
                <ul>
                    <Entry name='Version' value={systemInfo.studio_version} />
                    {systemInfo.git_tag ? <Entry name='Source tag' value={systemInfo.git_tag} /> : null}
                    {systemInfo.git_sha ? <Entry name='Commit' value={systemInfo.git_sha} /> : null}
                    {systemInfo.git_commit_date ? <Entry name='Committed' value={systemInfo.git_commit_date} /> : null}
                    {systemInfo.git_dirty ? <Entry name='Local changes' value='yes' /> : null}
                </ul>
            ) : null}
        </SettingsSection>
    );
};
