import { HttpResponse } from 'msw';

import { getMockedAwsAuthCapabilities } from '../src/test-utils/mocks/mock-aws-auth';
import { expect, http, test } from './fixtures';

test('AWS setup requires a named profile and explains backend configuration', async ({ page, network }) => {
    network.use(
        http.get('/api/jobs', () => HttpResponse.json([])),
        http.get('/api/remote-trainers', () => HttpResponse.json([])),
        http.get('/api/remote-servers/feature-status', () => HttpResponse.json({ network_exposed: false })),
        http.get('/api/remote-trainers/providers/aws/auth', () => HttpResponse.json(getMockedAwsAuthCapabilities())),
        http.post('/api/remote-trainers/providers/aws/configuration', () =>
            HttpResponse.json(
                {
                    id: 'credentials-expired',
                    title: 'AWS SSO session missing or expired',
                    detail: 'Renew the studio profile on the backend host, then retry in Studio.',
                    copy_text: 'aws sso login --profile=studio',
                },
                { status: 400 }
            )
        ),
        http.get('/api/remote-trainers/providers', () =>
            HttpResponse.json([
                {
                    id: 'aws',
                    title: 'AWS',
                    fields: [{ name: 'configuration_uri', title: 'S3 configuration file URI' }],
                },
            ])
        )
    );
    await page.goto('/settings/training-targets');
    await page.getByRole('button', { name: 'Add training target: AWS Provider' }).click();
    const configure = page.getByRole('dialog', { name: 'Deploy and configure AWS stack' });
    await expect(configure.getByRole('button', { name: 'Next' })).toBeDisabled();
    await expect(configure.getByRole('heading')).toHaveCount(1);
    await expect(configure.getByRole('button', { name: /AWS profile/ })).toHaveCount(0);
    await configure
        .getByRole('textbox', { name: 'S3 configuration file URI' })
        .fill('s3://your-bucket/studio-config.json');
    await configure.getByRole('button', { name: 'Next' }).click();
    const dialog = page.getByRole('dialog', { name: 'Connect to AWS remote training stack' });
    await expect(dialog.getByText(/Configure and authenticate this profile on the Studio backend host/)).toBeVisible();
    await expect(dialog.getByRole('button', { name: 'Connect', exact: true })).toBeDisabled();
    await expect(dialog.getByRole('radio')).toHaveCount(0);
    await dialog.getByRole('button', { name: 'Named profile help' }).click();
    const help = page.getByRole('dialog', { name: 'Named profile', exact: true });
    await expect(help).toBeVisible();
    await expect(help.getByText(/machine running the Studio backend/)).toBeVisible();
    await expect(help.getByText('aws sso login --profile studio', { exact: true })).toBeVisible();
    await page.keyboard.press('Escape');
    await dialog.getByRole('button', { name: /AWS profile/ }).click();
    await page.getByRole('option', { name: 'studio', exact: true }).click();
    await expect(dialog.getByRole('button', { name: 'Connect', exact: true })).toBeEnabled();
    const bounds = await dialog.boundingBox();
    const picker = dialog.getByRole('button', { name: /AWS profile/ });
    const fieldBounds = await picker.boundingBox();
    await dialog.getByRole('button', { name: 'Connect', exact: true }).click();
    await expect(page.getByText('Renew the studio profile on the backend host, then retry in Studio.')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Copy', exact: true })).toBeVisible();
    await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
    await page.getByRole('button', { name: 'Copy', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Copy', exact: true })).toHaveCount(0);
    expect(await page.evaluate(() => navigator.clipboard.readText())).toBe('aws sso login --profile=studio');
    await expect(page.getByText('Copied to clipboard.')).toHaveCount(0);
    await expect(page.getByRole('dialog')).toHaveCount(1);
    expect(await dialog.boundingBox()).toEqual(bounds);
    expect(await picker.boundingBox()).toEqual(fieldBounds);
    await expect(dialog.getByRole('alert')).toHaveCount(0);
    await dialog.getByRole('button', { name: 'Back' }).click();
    await expect(configure.getByRole('textbox', { name: 'S3 configuration file URI' })).toHaveValue(
        's3://your-bucket/studio-config.json'
    );
});
