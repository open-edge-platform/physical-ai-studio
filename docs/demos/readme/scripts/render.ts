// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {spawn} from 'node:child_process';
import {createServer} from 'node:http';
import {copyFile, mkdir, mkdtemp, readFile, readdir, rename, rm} from 'node:fs/promises';
import type {AddressInfo} from 'node:net';
import {extname, join, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {parseArgs} from 'node:util';
import {build} from 'esbuild';
import {chromium, type Browser} from '@playwright/test';
import {fps} from '../src/animation.ts';
import {demos, outputFrameCount, type DemoId} from '../src/config.ts';
import {guiSource} from '../src/gui.ts';

const project = fileURLToPath(new URL('../', import.meta.url));
const out = join(project, 'out');
const site = join(out, 'site');
const sourceFrames = join(out, 'gui-source');

const ffmpeg = async (args: string[]): Promise<void> => {
  await new Promise<void>((resolve, reject) => {
    const child = spawn('ffmpeg', ['-y', '-loglevel', 'error', ...args], {stdio: 'inherit'});
    child.once('error', (error) => reject(new Error('Install FFmpeg and ensure ffmpeg is on PATH.', {cause: error})));
    child.once('exit', (code) => (code === 0 ? resolve() : reject(new Error(`FFmpeg exited with ${code}`))));
  });
};

const prepareSite = async (): Promise<void> => {
  await mkdir(site, {recursive: true});
  await build({
    absWorkingDir: project,
    entryPoints: ['src/index.tsx'],
    outfile: join(site, 'demo.js'),
    bundle: true,
    format: 'esm',
    platform: 'browser',
    target: 'es2022',
    jsx: 'automatic',
    define: {'process.env.NODE_ENV': '"production"'},
    loader: {'.woff2': 'file'},
  });
  await copyFile(join(project, 'index.html'), join(site, 'index.html'));
};

const prepareGui = async (): Promise<void> => {
  // Decode once; browser screenshots must not depend on video playback/seek timing.
  const directory = await mkdtemp(join(out, 'gui-source-'));
  const count = outputFrameCount('gui');
  try {
    await ffmpeg([
      '-i', resolve(project, guiSource),
      '-vf', `fps=${demos.gui.outputFps},scale=1140:-2`,
      '-frames:v', String(count),
      '-q:v', '2',
      '-start_number', '0',
      join(directory, '%06d.jpg'),
    ]);
    if ((await readdir(directory)).length !== count) {
      throw new Error('The Studio recording is shorter than the GUI timeline. Check src/gui.ts.');
    }
    await rm(sourceFrames, {recursive: true, force: true});
    await rename(directory, sourceFrames);
  } finally {
    await rm(directory, {recursive: true, force: true});
  }
};

const serve = async (port: number) => {
  const mime: Record<string, string> = {
    '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
    '.jpg': 'image/jpeg', '.woff2': 'font/woff2',
  };
  const server = createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url ?? '/', 'http://localhost').pathname);
      const sourceFrame = /^\/gui-source\/\d{6}\.jpg$/.test(pathname);
      const base = sourceFrame ? sourceFrames : site;
      const file = sourceFrame
        ? join(base, pathname.split('/').at(-1)!)
        : resolve(base, `.${pathname === '/' ? '/index.html' : pathname}`);
      if (relative(base, file).startsWith('..')) {
        response.writeHead(403).end();
        return;
      }
      const body = await readFile(file);
      response.writeHead(200, {'Content-Type': mime[extname(file)] ?? 'application/octet-stream'});
      response.end(body);
    } catch {
      response.writeHead(404).end();
    }
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', resolve);
  });
  return {
    url: `http://127.0.0.1:${(server.address() as AddressInfo).port}`,
    close: async () => {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    },
  };
};

const render = async (browser: Browser, url: string, id: DemoId): Promise<void> => {
  const config = demos[id];
  const directory = join(out, `${id}-demo`);
  await rm(directory, {recursive: true, force: true});
  await mkdir(directory, {recursive: true});
  const page = await browser.newPage({viewport: {width: config.width, height: config.height}, deviceScaleFactor: 1});
  try {
    await page.goto(`${url}/?capture`);
    await page.waitForFunction(() => typeof window.renderFrame === 'function');
    for (let index = 0; index < outputFrameCount(id); index++) {
      await page.evaluate(async ({id, frame}) => window.renderFrame(id, frame), {
        id, frame: (index * fps) / config.outputFps,
      });
      await page.screenshot({path: join(directory, `${String(index).padStart(6, '0')}.png`), animations: 'disabled'});
    }
  } finally {
    await page.close();
  }

  const input = ['-framerate', String(config.outputFps), '-i', join(directory, '%06d.png')];
  const palette = join(directory, 'palette.png');
  const gif = join(directory, `${id}-demo.gif`);
  await ffmpeg([...input, '-vf', 'palettegen=stats_mode=diff', palette]);
  await ffmpeg([
    ...input, '-i', palette,
    '-lavfi', 'paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle',
    '-loop', '0', gif,
  ]);
  // Replace a published asset only after every frame and the GIF encode succeeded.
  const destination = resolve(project, `../../assets/readme/${id}-demo.gif`);
  await rename(gif, destination);
  console.log(`Rendered ${relative(project, destination)} (${outputFrameCount(id)} frames).`);
};

const main = async (): Promise<void> => {
  const {positionals, values} = parseArgs({
    allowPositionals: true,
    options: {port: {type: 'string', default: '4173'}},
  });
  const [command] = positionals;
  if (positionals.length !== 1 || !['api', 'gui', 'all', 'preview'].includes(command)) {
    throw new Error('Usage: node scripts/render.ts api|gui|all|preview [--port 4173]');
  }
  const preview = command === 'preview';
  const port = Number(values.port);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid preview port.');
  const targets: DemoId[] = command === 'all' ? ['api', 'gui'] : command === 'gui' ? ['gui'] : ['api'];
  await prepareSite();
  if (preview || targets.includes('gui')) await prepareGui();
  const server = await serve(preview ? port : 0);
  try {
    if (preview) {
      console.log(`Preview: ${server.url} (Ctrl-C to stop)`);
      await new Promise<void>((resolve) => {
        process.once('SIGINT', () => resolve());
        process.once('SIGTERM', () => resolve());
      });
    } else {
      const browser = await chromium.launch();
      try {
        for (const id of targets) await render(browser, server.url, id);
      } finally {
        await browser.close();
      }
    }
  } finally {
    await server.close();
  }
};

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
