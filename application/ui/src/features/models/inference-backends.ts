import { type FunctionComponent, type SVGProps } from 'react';

import { ReactComponent as ExecuTorchLogo } from './../../assets/logos/executorch-logo-small.svg';
import { ReactComponent as ONNXLogo } from './../../assets/logos/onnx-logo-small.svg';
import { ReactComponent as OpenVINOLogo } from './../../assets/logos/OpenVINO-small.svg';
import { ReactComponent as TorchLogo } from './../../assets/logos/pytorch-logo-small.svg';

export interface InferenceBackendConfig {
    type: string;
    label: string;
    description: string;
    logo: FunctionComponent<SVGProps<SVGSVGElement>>;
}

export const INFERENCE_BACKENDS: Record<string, InferenceBackendConfig> = {
    torch: {
        type: 'torch',
        label: 'PyTorch',
        description: 'Run inference using PyTorch framework.',
        logo: TorchLogo,
    },
    openvino: {
        type: 'openvino',
        label: 'OpenVINO',
        description: 'Run inference using OpenVINO.',
        logo: OpenVINOLogo,
    },
    onnx: {
        type: 'onnx',
        label: 'ONNX',
        description: 'Run inference using the ONNX Runtime.',
        logo: ONNXLogo,
    },
    executorch: {
        type: 'executorch',
        label: 'ExecuTorch',
        description: 'Run inference on edge devices using ExecuTorch.',
        logo: ExecuTorchLogo,
    },
};
