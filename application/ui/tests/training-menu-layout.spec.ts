import { HttpResponse } from 'msw';

import { getMockedDataset } from '../src/test-utils/mocks/mock-dataset';
import { getMockedEpisode } from '../src/test-utils/mocks/mock-episode';
import { expect, http, test } from './fixtures';

const policies = [
    { id: 'act', name: 'ACT', lora: false, snapflow: false },
    { id: 'smolvla', name: 'SmolVLA', lora: false, snapflow: true },
    { id: 'pi05', name: 'Pi0.5', lora: true, snapflow: true },
    { id: 'molmoact2', name: 'MolmoAct2', lora: true, snapflow: false },
    { id: 'rldx1', name: 'RLDX-1', lora: false, snapflow: false },
    { id: 'xr0', name: 'XR0', lora: false, snapflow: false },
];

for (const policy of policies) {
    test(`training menu layout for ${policy.name}`, async ({ page, network }, testInfo) => {
        await page.setViewportSize({ width: 1280, height: 1100 });
        const projectId = 'b8b28d4f-e78f-48ad-afb8-03d060178a3c';
        const dataset = getMockedDataset({ project_id: projectId, name: 'Hackathon' });
        network.use(
            http.get('/api/projects/{project_id}', () =>
                HttpResponse.json({ id: projectId, name: 'Test project', datasets: [dataset] })
            ),
            http.get('/api/projects/{project_id}/models', () => HttpResponse.json([])),
            http.get('/api/jobs', () => HttpResponse.json([])),
            http.get('/api/remote-trainers', () => HttpResponse.json([])),
            http.get('/api/system/devices/training', () =>
                HttpResponse.json({ mode: 'local', remote_available: false, devices: [] })
            ),
            http.get('/api/policies/available', () =>
                HttpResponse.json(
                    policies.map(({ id, name }) => ({ id, name, description: `${name} policy`, min_vram: null }))
                )
            ),
            http.get('/api/policies/{policy}/huggingface-access', () => HttpResponse.json({ requirements: [] })),
            http.get('/api/dataset/{dataset_id}/episodes', () =>
                HttpResponse.json([getMockedEpisode({ episode_index: 0, length: 100 })])
            ),
            http.get('/api/dataset/{dataset_id}/episodes/{episode_index}', () =>
                HttpResponse.json({
                    episode_index: 0,
                    length: 100,
                    fps: 30,
                    tasks: [],
                    actions: [],
                    action_keys: [],
                    videos: { 'observation.images.top': { start: 0, end: 3, path: 'top.mp4' } },
                })
            )
        );
        await page.goto(`/projects/${projectId}/models`);
        await page.getByRole('button', { name: 'Train model', exact: true }).click();
        const dialog = page.getByRole('dialog');
        await dialog.getByLabel(`Select ${policy.name} policy`, { exact: true }).click();
        await dialog.getByRole('button', { name: 'Next', exact: true }).click();
        if (policy.id === 'smolvla') {
            await dialog.getByRole('button', { name: 'Next', exact: true }).click();
        }
        await expect(dialog.getByRole('heading', { name: 'Training settings', exact: true })).toBeVisible();
        const fields = [
            dialog.getByRole('textbox', { name: 'Max Epochs', exact: true }),
            dialog.getByRole('button', { name: 'Auto Data Workers', exact: true }),
            dialog.getByRole('button', { name: '32-bit Precision', exact: true }),
        ];
        const bounds = await Promise.all(fields.map((field) => field.boundingBox()));
        expect(bounds.every((box) => box !== null)).toBe(true);
        expect(Math.max(...bounds.map((box) => box!.y)) - Math.min(...bounds.map((box) => box!.y))).toBeLessThan(2);
        const lora = dialog.getByRole('checkbox', { name: 'LoRA fine-tuning', exact: true });
        const snapflow = dialog.getByRole('checkbox', { name: 'SnapFlow distillation', exact: true });
        await expect(lora).toHaveCount(policy.lora ? 1 : 0);
        await expect(snapflow).toHaveCount(policy.snapflow ? 1 : 0);
        await dialog.screenshot({ path: testInfo.outputPath('training-menu.png') });
        if (policy.lora) {
            await lora.check();
            await expect(dialog.getByRole('textbox', { name: 'LoRA rank', exact: true })).toBeVisible();
            await expect(dialog.getByRole('checkbox', { name: 'Use DoRA', exact: true })).toBeVisible();
        }
        if (policy.snapflow) {
            await expect(dialog.getByRole('textbox', { name: 'Distillation epochs', exact: true })).toHaveCount(0);
            await snapflow.check();
            await expect(dialog.getByRole('textbox', { name: 'Distillation epochs', exact: true })).toBeVisible();
        }
        if (policy.lora || policy.snapflow) {
            await dialog.screenshot({ path: testInfo.outputPath('training-menu-expanded.png') });
        }
    });
}
