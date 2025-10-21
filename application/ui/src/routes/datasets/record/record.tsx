import { useState } from 'react';

import { SchemaTeleoperationConfig } from '../../../api/openapi-spec';
import { useProject } from '../../../features/projects/use-project';
import { HardwareSetup } from './hardware-setup';
import { ProjectSetup } from './project-setup';
import { Recording } from './recording';

export const Record = () => {
    const [config, setConfig] = useState<SchemaTeleoperationConfig>();
    const project = useProject();
    if (!project.config) {
        return <ProjectSetup />;
    }

    if (config) {
        return <Recording setup={config} />;
    } else {
        return <HardwareSetup onDone={setConfig} />;
    }
};
