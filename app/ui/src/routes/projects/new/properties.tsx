import { Form, NumberField, TextField } from '@geti/ui';

import { useNewProject } from './new-project.provider';

export const PropertiesView = () => {
    const { project, setProject } = useNewProject();

    const setProjectName = (name: string) => setProject({ ...project, name });
    const setProjectFPS = (fps: number) => setProject({ ...project, fps });

    return (
        <Form maxWidth='size-3600'>
            <TextField label='name' value={project.name} onChange={setProjectName} />
            <NumberField label='fps' value={project.fps} minValue={1} onChange={setProjectFPS} />
        </Form>
    );
};
