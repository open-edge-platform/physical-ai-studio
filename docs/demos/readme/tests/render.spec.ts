import {execFile} from 'node:child_process';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {promisify} from 'node:util';
import {expect, test} from '@playwright/test';
import {fps, progress} from '../src/animation.ts';
import {demos, outputFrameCount} from '../src/config.ts';
import {clipDuration, guiClips, guiDuration, guiFrameUrl} from '../src/gui.ts';
import {stages} from '../src/workflow';

test('the timeline clamps progress and covers the committed recording without gaps', () => {
  expect(progress(-1, 0, 30)).toBe(0);
  expect(progress(15, 0, 30)).toBe(0.5);
  expect(progress(31, 0, 30)).toBe(1);
  expect(outputFrameCount('api')).toBe(250);
  expect(outputFrameCount('gui')).toBe(283);
  expect(guiClips[0].from).toBe(0);
  for (let index = 1; index < guiClips.length; index++) {
    expect(guiClips[index].from).toBe(guiClips[index - 1].to);
  }
  expect(guiDuration / fps).toBeCloseTo(guiClips.at(-1)!.to);
  expect(guiFrameUrl((282 * fps) / demos.gui.outputFps)).toBe('/gui-source/000282.jpg');
});

test.beforeEach(async ({page}) => {
  await page.goto('/?capture');
  await page.waitForFunction(() => typeof window.renderFrame === 'function');
});

test('API stage highlights, code reveals, and captures depend only on the requested frame', async ({page}) => {
  for (const stage of stages) {
    await page.evaluate((frame) => window.renderFrame('api', frame), stage.start + 24);
    await expect(page.locator('[aria-current="step"]')).toHaveAttribute('data-stage', stage.title);
  }
  await page.evaluate(() => window.renderFrame('api', 450));
  await expect(page.getByText('From dataset to deployable policy')).toBeVisible();
  await expect(page.locator('[data-stage]')).toHaveCount(4);
  expect(await page.locator('[data-stage]').allTextContents()).toEqual(expect.arrayContaining([
    expect.stringContaining('Run with Physical AI Runtime'),
  ]));
  expect(await page.locator('.token-line').evaluateAll((lines) =>
    lines.every((line) => getComputedStyle(line).opacity === '1'),
  )).toBe(true);
  const first = await page.screenshot();
  await page.evaluate(() => window.renderFrame('api', 60));
  await page.evaluate(() => window.renderFrame('api', 450));
  expect((await page.screenshot()).equals(first)).toBe(true);
  expect(await page.evaluate(() => document.fonts.check('20px "JetBrains Mono Variable"'))).toBe(true);
});

test('GUI stages seek to decoded source frames, including backwards and the last frame', async ({page}) => {
  let start = 0;
  for (const clip of guiClips) {
    const frame = start + clipDuration(clip) / 2;
    await page.evaluate((frame) => window.renderFrame('gui', frame), frame);
    await expect(page.locator('[aria-current="step"]')).toHaveAttribute('data-stage', clip.title);
    await expect(page.getByAltText('Physical AI Studio walkthrough')).toHaveAttribute('src', guiFrameUrl(frame));
    expect(await page.getByAltText('Physical AI Studio walkthrough').evaluate((image: HTMLImageElement) => image.naturalWidth)).toBeGreaterThan(0);
    start += clipDuration(clip);
  }
  await page.evaluate(() => window.renderFrame('gui', (282 * 30) / 8));
  const last = await page.screenshot();
  await page.evaluate(() => window.renderFrame('gui', 0));
  expect((await page.screenshot()).equals(last)).toBe(false);
  await page.evaluate(() => window.renderFrame('gui', (282 * 30) / 8));
  expect((await page.screenshot()).equals(last)).toBe(true);
});

test('capture rejects out-of-range frame requests', async ({page}) => {
  for (const frame of [-1, 500, Number.NaN]) {
    const error = await page.evaluate(async (frame) => {
      try {
        await window.renderFrame('api', frame);
      } catch (error) {
        return (error as Error).message;
      }
    }, frame);
    expect(error).toContain('Invalid api frame');
  }
});

test('preview supports switching demos, scrubbing, and playback', async ({page}) => {
  await page.goto('/');
  await page.getByRole('combobox', {name: 'Demo'}).selectOption('gui');
  await page.getByRole('slider', {name: 'Frame'}).fill('600');
  await expect(page.locator('[aria-current="step"]')).toHaveAttribute('data-stage', 'Train');
  await page.getByRole('button', {name: 'Play', exact: true}).click();
  await expect(page.getByRole('button', {name: 'Pause', exact: true})).toBeVisible();
  await expect.poll(() => page.getByRole('slider', {name: 'Frame'}).inputValue()).not.toBe('600');
  await page.getByRole('button', {name: 'Pause', exact: true}).click();
});

test('all locked npm packages declare reviewed open-source licenses', async () => {
  const lock = JSON.parse(await readFile(new URL('../package-lock.json', import.meta.url), 'utf8')) as {
    packages: Record<string, {license?: string}>;
  };
  for (const [name, dependency] of Object.entries(lock.packages)) {
    expect(['MIT', 'Apache-2.0', 'OFL-1.1'], name).toContain(dependency.license);
    expect(name).not.toMatch(/remotion/i);
  }
});

test('published GIFs have the expected dimensions, frame counts, durations, and infinite loops', async () => {
  for (const id of ['api', 'gui'] as const) {
    const path = new URL(`../../../assets/readme/${id}-demo.gif`, import.meta.url);
    const {stdout} = await promisify(execFile)('ffprobe', [
      '-v', 'error', '-show_entries', 'stream=codec_name,width,height,nb_frames',
      '-show_entries', 'format=duration', '-of', 'json', fileURLToPath(path),
    ]);
    const metadata = JSON.parse(stdout);
    expect(metadata.streams[0]).toMatchObject({
      codec_name: 'gif', width: demos[id].width, height: demos[id].height,
      nb_frames: String(outputFrameCount(id)),
    });
    expect(Math.abs(Number(metadata.format.duration) - demos[id].durationInFrames / fps)).toBeLessThan(0.03);
    const data = await readFile(path);
    expect(data.subarray(0, 6).toString()).toMatch(/^GIF8[79]a$/);
    const loop = data.indexOf('NETSCAPE2.0');
    expect(loop).toBeGreaterThan(0);
    expect(Array.from(data.subarray(loop + 11, loop + 16))).toEqual([3, 1, 0, 0, 0]);
  }
});

test('missing FFmpeg fails clearly without overwriting either published GIF', async () => {
  const paths = ['api', 'gui'].map((id) => new URL(`../../../assets/readme/${id}-demo.gif`, import.meta.url));
  const before = await Promise.all(paths.map((path) => readFile(path)));
  await expect(promisify(execFile)(process.execPath, [
    fileURLToPath(new URL('../scripts/render.ts', import.meta.url)), 'gui',
  ], {
    cwd: fileURLToPath(new URL('../', import.meta.url)),
    env: {...process.env, PATH: ''},
    timeout: 15_000,
  })).rejects.toMatchObject({code: 1, stderr: expect.stringContaining('Install FFmpeg')});
  for (let index = 0; index < paths.length; index++) {
    expect((await readFile(paths[index])).equals(before[index])).toBe(true);
  }
});
