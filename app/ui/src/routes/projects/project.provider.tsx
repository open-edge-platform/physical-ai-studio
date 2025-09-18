import { createContext, ReactNode, useContext } from 'react';

import { $api } from '../../api/client';
import { SchemaProjectConfig } from '../../api/openapi-spec';

interface ProjectContext {
    project: SchemaProjectConfig;
}
export const ProjectContext = createContext<ProjectContext | undefined>(undefined);

export const ProjectProvider = ({ children, project_id }: { children: ReactNode; project_id: string }) => {
    const { data: project } = $api.useSuspenseQuery('get', '/api/projects/{id}', {
        params: { path: { id: project_id } },
    });

    return (
        <ProjectContext.Provider
            value={{
                project,
            }}
        >
            {children}
        </ProjectContext.Provider>
    );
};

export function useProject(): ProjectContext {
    const context = useContext(ProjectContext);

    if (context === undefined) {
        throw new Error('No project context');
    }

    return context;
}
