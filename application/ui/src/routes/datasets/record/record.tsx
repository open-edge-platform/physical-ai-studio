import { useState } from 'react';

import { SchemaTeleoperationConfig } from '../../../api/openapi-spec';
import { HardwareSetup } from './hardware-setup';
import { Recording } from './recording';

export const Record = () => {
    const [config, setConfig] = useState<SchemaTeleoperationConfig>();

    if (config) {
        return <Recording setup={config} />;
    } else {
        return <HardwareSetup onDone={setConfig} />;
    }
};
