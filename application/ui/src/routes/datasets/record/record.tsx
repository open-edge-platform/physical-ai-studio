import { useState } from 'react';

import { SchemaTeleoperationConfig } from '../../../api/openapi-spec';
import { HardwareSetup } from './hardware-setup';
import { Recording } from './recording';
import { useProject } from '../../../features/projects/use-project';
import { ProjectSetup } from './project-setup';

export const Record = () => {
    const [config, setConfig] = useState<SchemaTeleoperationConfig>();
    const project = useProject();
    if (!project.config) {
        return <ProjectSetup />
    }

    if (config) {
        return <Recording setup={config} />;
    } else {
        return <HardwareSetup onDone={setConfig} />;
    }
};
