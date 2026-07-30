import { getMainPageInProjectUrl } from './project-navigation';

describe('getMainPageInProjectUrl', () => {
    it('selects the Remote Servers tab for the project-scoped route', () => {
        expect(getMainPageInProjectUrl('/projects/project-1/remote-servers')).toBe('remote-servers');
    });

    it('keeps robot sub-routes grouped under the Robots tab', () => {
        expect(getMainPageInProjectUrl('/projects/project-1/cameras/camera-1')).toBe('robots');
        expect(getMainPageInProjectUrl('/projects/project-1/environments/environment-1')).toBe('robots');
    });
});
