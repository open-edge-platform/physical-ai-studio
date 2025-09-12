import { Form, TextField, NumberField } from '@geti/ui';
import { useProjectDataContext } from "./project-config.provider";

export const PropertiesView = () => {
    const {project, setProject} = useProjectDataContext();

    const setProjectName = (name: string) => setProject({...project, name});
    const setProjectFPS = (fps: number) => setProject({...project, fps});

    return (
        <Form maxWidth="size-3600">
            <TextField label="name" value={project.name} onChange={setProjectName}/>
            <NumberField label="fps" value={project.fps} minValue={1} onChange={setProjectFPS}/>
        </Form>
    )
}
