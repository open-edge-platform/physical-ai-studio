import { expect, test } from './fixtures';

test.describe('Physical AI Studio', () => {
    test('Opens the home page', async ({ page }) => {
        await page.goto('/');
        await expect(page.getByRole('button', { name: 'Add project' })).toBeVisible();
    });
});
