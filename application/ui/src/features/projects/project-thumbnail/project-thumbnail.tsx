import { useState } from 'react';

import { Flex } from '@geti-ui/ui';

import { fetchClient } from '../../../api/client';
import { ReactComponent as ProjectThumbnailPlaceholder } from '../../../assets/icons/physicalai-studio-logo.svg';

import classes from './project-thumbnail.module.css';

type ProjectThumbnailProps = {
    projectId: string;
    name: string;
    size: number;
};

export const ProjectThumbnail = ({ projectId, name, size }: ProjectThumbnailProps) => {
    const [hasThumbnailError, setHasThumbnailError] = useState(false);

    const thumbnailUrl = fetchClient.PATH('/api/projects/{project_id}/thumbnail', {
        params: { path: { project_id: projectId }, query: { width: size, height: size } },
    });

    if (hasThumbnailError) {
        const placeholderSize = Math.round(size * 0.6);

        return (
            <Flex width={size} height={size} justifyContent={'center'} alignItems={'center'}>
                <ProjectThumbnailPlaceholder
                    width={placeholderSize}
                    height={placeholderSize}
                    className={classes.placeholder}
                    aria-hidden
                />
            </Flex>
        );
    }

    return (
        <img
            style={{ width: size, height: size }}
            className={classes.thumbnail}
            src={thumbnailUrl}
            alt={name}
            onError={() => setHasThumbnailError(true)}
        />
    );
};
