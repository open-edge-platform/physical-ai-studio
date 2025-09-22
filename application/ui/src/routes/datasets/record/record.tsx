import { useState } from 'react';

import { SchemaProjectConfig } from '../../../api/openapi-spec';
import { useProject } from '../../../features/projects/use-project';
import { HardwareSetup } from './hardware-setup';

export const Record = () => {
    const projectConfig = useProject();
    const [project, setProject] = useState<SchemaProjectConfig>(projectConfig);

    return <HardwareSetup project={project} setProject={setProject} />;
};
