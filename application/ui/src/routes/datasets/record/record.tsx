import { useState } from 'react';

import { SchemaTeleoperationConfig } from '../../../api/openapi-spec';
import { useProject } from '../../../features/projects/use-project';
import { HardwareSetup } from './hardware-setup';
import { useParams } from 'react-router';
import { $api } from '../../../api/client';

export const Record = () => {
    const { dataset_id } = useParams<{ dataset_id: string }>();
    const project = useProject();


    const [teleoperationConfig, setTeleoperationConfig] = useState<SchemaTeleoperationConfig>({
        dataset_id: dataset_id!,
        cameras: project.config?.cameras ?? [],
        robots: [],
        task: ""
    });

    return <HardwareSetup config={teleoperationConfig} setConfig={setTeleoperationConfig} />;
};
