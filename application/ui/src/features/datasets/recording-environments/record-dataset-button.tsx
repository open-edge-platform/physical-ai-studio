import { ReactNode, Suspense, useState } from 'react';

import { Button, Loading } from '@geti-ui/ui';
import { Navigate } from 'react-router';

import { paths } from '../../../paths';
import { RecordingEnvironmentGate } from './recording-environment-gate';

interface RecordDatasetButtonProps {
    project_id: string;
    dataset_id: string;
    children: ReactNode;
}

export const RecordDatasetButton = ({ project_id, dataset_id, children }: RecordDatasetButtonProps) => {
    const [isStarting, setIsStarting] = useState(false);

    return (
        <>
            <Button variant='secondary' onPress={() => setIsStarting(true)} isDisabled={isStarting}>
                {children}
            </Button>
            {isStarting && (
                <Suspense fallback={<Loading mode='overlay' />}>
                    <RecordingEnvironmentGate
                        project_id={project_id}
                        dataset_id={dataset_id}
                        onCancel={() => setIsStarting(false)}
                    >
                        {() => <Navigate to={paths.project.datasets.record({ project_id, dataset_id })} />}
                    </RecordingEnvironmentGate>
                </Suspense>
            )}
        </>
    );
};
