import { expect, test } from './fixtures';

test.describe('Geti Action', () => {
    test('Opens the home page', async ({ page }) => {
        await page.goto('/');
        await expect(page.getByRole('button', { name: 'Add project' })).toBeVisible();
    });
});
