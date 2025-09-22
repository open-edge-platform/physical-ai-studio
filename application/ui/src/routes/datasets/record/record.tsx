import { useState } from 'react';

import { SchemaProjectConfig } from '../../../api/openapi-spec';
import { useProject } from '../../projects/project.provider';
import { HardwareSetup } from './hardware-setup';

export const Record = () => {
    const { project: projectConfig } = useProject();
    const [project, setProject] = useState<SchemaProjectConfig>(projectConfig);

    return <HardwareSetup project={project} setProject={setProject} />;
};
