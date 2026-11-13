// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, rs } from '@rstest/core';
import { screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import * as THREE from 'three';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { RobotModelsProvider } from '../robot-models-context';
import { RobotViewer } from './robot-viewer';

type Load = (
    manager: THREE.LoadingManager,
    url: string,
    onLoad: (model: unknown) => void,
    onProgress: unknown,
    onError: (error: Error) => void
) => void;

const { load } = rs.hoisted(() => ({ load: rs.fn<Load>() }));

// The 3D canvas needs WebGL, which jsdom lacks; the model loading around it is what is tested here.
rs.mock('@react-three/fiber', () => ({
    Canvas: () => <div data-testid='robot-canvas' />,
}));
rs.mock('urdf-loader', () => ({
    default: class {
        manager: THREE.LoadingManager;
        packages: Record<string, string> = {};

        constructor(manager: THREE.LoadingManager) {
            this.manager = manager;
        }

        load(url: string, onLoad: (model: unknown) => void, onProgress: unknown, onError: (error: Error) => void) {
            load(this.manager, url, onLoad, onProgress, onError);
        }
    },
}));

const ROBOT_TYPE = 'MuJoCo_FrankaResearch3_Follower';

const mockCatalog = (urdfPath: string | null) =>
    server.use(
        http.get('/api/robots/catalog', () =>
            HttpResponse.json([
                {
                    type: ROBOT_TYPE,
                    display_name: 'MuJoCo Franka Research 3 Follower',
                    role: 'follower',
                    urdf_path: urdfPath,
                    package_map: urdfPath === null ? {} : { mujoco_franka_fr3_1: `/api/robots/catalog/${ROBOT_TYPE}` },
                    joint_map: urdfPath === null ? {} : { 'fr3_joint1.pos': ['fr3_joint1_channel'] },
                    zero_calibration: null,
                    simulation: null,
                },
            ])
        )
    );

const renderViewer = () =>
    render(
        <RobotModelsProvider>
            <RobotViewer robot={{ type: ROBOT_TYPE }} />
        </RobotModelsProvider>
    );

describe('RobotViewer', () => {
    beforeEach(() => {
        load.mockReset();
    });

    it('says the robot has no 3D model when its type has no URDF, without loading one', async () => {
        mockCatalog(null);

        renderViewer();

        expect(await screen.findByRole('status')).toHaveTextContent('No 3D model for this robot');
        expect(screen.queryByText('Loading robot model...')).not.toBeInTheDocument();
        expect(load).not.toHaveBeenCalled();
    });

    it('says the robot has no 3D model when its URDF fails to load', async () => {
        mockCatalog(`/api/robots/catalog/${ROBOT_TYPE}/urdf`);
        load.mockImplementation((manager, url, _onLoad, _onProgress, onError) => {
            manager.itemStart(url);
            onError(new Error(`URDFLoader: Failed to load url '${url}' with error code 404 : Not Found.`));
            manager.itemError(url);
            manager.itemEnd(url);
        });

        renderViewer();

        const alert = await screen.findByRole('alert');
        expect(alert).toHaveTextContent('No 3D model for this robot');
        expect(alert).toHaveTextContent('error code 404');
        expect(screen.queryByText('Loading robot model...')).not.toBeInTheDocument();
        expect(load).toHaveBeenCalledWith(
            expect.anything(),
            `/api/robots/catalog/${ROBOT_TYPE}/urdf`,
            expect.anything(),
            undefined,
            expect.anything()
        );
    });

    it('shows the model once its URDF and meshes have loaded', async () => {
        mockCatalog(`/api/robots/catalog/${ROBOT_TYPE}/urdf`);
        let finish = () => {};
        load.mockImplementation((manager, url, onLoad) => {
            manager.itemStart(url);
            onLoad(Object.assign(new THREE.Group(), { joints: {} }));
            finish = () => manager.itemEnd(url);
        });

        renderViewer();

        expect(await screen.findByText('Loading robot model...')).toBeInTheDocument();
        finish();
        await waitFor(() => expect(screen.queryByText('Loading robot model...')).not.toBeInTheDocument());
        expect(screen.getByTestId('robot-canvas')).toBeInTheDocument();
        expect(screen.queryByText('No 3D model for this robot')).not.toBeInTheDocument();
        expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    });
});
