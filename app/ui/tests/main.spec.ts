import { expect, test } from './fixtures';

test.describe('Geti Action', () => {
    test('Opens the home page', async ({ page }) => {
        await page.goto('/');
        await expect(page.getByText('Geti Action')).toBeVisible();
    });
});
