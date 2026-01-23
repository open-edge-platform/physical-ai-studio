import { $api } from '../../../../api/client';
import { CameraDriver } from '../provider';

export const useAvailableCameras = (driver: CameraDriver) => {
    const query = $api.useSuspenseQuery('get', '/api/hardware/cameras');
    const availableCameras = query.data.filter(
        (camera) => camera.driver === driver || (driver === 'usb_camera' && camera.driver === 'webcam')
    );

    return availableCameras;
};

export const useAllAvailableCameras = () => {
    return $api.useSuspenseQuery('get', '/api/hardware/cameras');
};

export const useSupportedFormats = (driver: CameraDriver, fingerprint: string | undefined) => {
    const query = $api.useQuery(
        'get',
        '/api/cameras/supported_formats/{driver}',
        {
            params: {
                path: { driver },
                query: { fingerprint: fingerprint ?? '' },
            },
        },
        { enabled: !!fingerprint }
    );

    return query.data ?? [];
};
