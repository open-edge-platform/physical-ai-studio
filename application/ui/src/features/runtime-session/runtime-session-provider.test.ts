import { describe, expect, it } from '@rstest/core';

import { environmentDraft, parseObservation } from './runtime-session-provider';

describe('parseObservation', () => {
    it('splits the header from the jpegs it lists', async () => {
        const header = new TextEncoder().encode(
            JSON.stringify({
                type: 'observation',
                timestamp: 2,
                state: { 'gripper.pos': 1 },
                action: null,
                frames: [
                    { key: 'front', timestamp: 2, size: 2 },
                    { key: 'wrist', timestamp: 1, size: 3 },
                ],
            })
        );
        const length = new Uint8Array(4);
        new DataView(length.buffer).setUint32(0, header.length);
        const blob = new Blob([length, header, new Uint8Array([1, 2]), new Uint8Array([3, 4, 5])]);

        const { observation, frames } = await parseObservation(blob);

        expect(observation.state).toEqual({ 'gripper.pos': 1 });
        expect([...frames.keys()]).toEqual(['front', 'wrist']);
        expect(frames.get('wrist')?.type).toBe('image/jpeg');
        const wrist = frames.get('wrist') ?? new Blob();
        expect([...new Uint8Array(await wrist.arrayBuffer())]).toEqual([3, 4, 5]);
    });
});

describe('environmentDraft', () => {
    it('loads a saved environment by its device ids', () => {
        const draft = environmentDraft({
            id: 'env',
            name: 'Bench',
            robots: [
                {
                    robot: { id: 'follower', name: 'Follower', type: 'SO101_Follower', payload: {} },
                    tele_operator: { type: 'robot', robot_id: 'leader', robot: null },
                },
            ],
            cameras: [{ id: 'cam', name: 'Front' }, { name: 'not saved yet' }],
        } as unknown as Parameters<typeof environmentDraft>[0]);

        expect(draft).toEqual({
            id: 'env',
            name: 'Bench',
            robots: [{ robot_id: 'follower', tele_operator: { type: 'robot', robot_id: 'leader' } }],
            cameras: [{ camera_id: 'cam' }],
        });
    });
});
