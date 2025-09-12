import { createContext, ReactNode } from "react";
import { SchemaProjectConfig } from "../../api/openapi-spec";
import { $api } from "../../api/client";


interface ProjectContext {
    project: SchemaProjectConfig
}
export const ProjectContext = createContext<ProjectContext | undefined>(undefined);


export const ProjectProvider = ({ children, project_id }: { children: ReactNode, project_id: string }) => {
    const {data: project} = $api.useSuspenseQuery('get','/api/projects/{id}', {
        params: { path: { id: project_id } },
    });

    return (
        <ProjectContext.Provider value={{
            project,
        }}>
            {children}
        </ProjectContext.Provider>
    )
}