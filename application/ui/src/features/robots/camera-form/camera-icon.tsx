import { ReactComponent as GenICamIcon } from '../../../assets/icons/genicam.svg';
import { ReactComponent as IpCameraIcon } from '../../../assets/icons/ip-camera.svg';
import { ReactComponent as WebcamIcon } from '../../../assets/icons/webcam.svg';
import { CameraDriver } from './provider';

interface SourceIconProps {
    type: CameraDriver | string;
    width?: string;
}

export const CameraIcon = ({ type, width }: SourceIconProps) => {
    if (type === 'usb_camera') {
        return <WebcamIcon width={width} />;
    }
    if (type === 'realsense') {
        return <WebcamIcon width={width} />;
    }
    if (type === 'ipcam') {
        return <IpCameraIcon width={width} />;
    }
    if (type === 'basler') {
        return <WebcamIcon width={width} />;
    }
    if (type === 'genicam') {
        return <GenICamIcon width={width} />;
    }

    return <WebcamIcon width={width} />;
};
