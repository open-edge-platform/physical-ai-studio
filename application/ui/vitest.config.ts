import react from '@vitejs/plugin-react';
import svgr from 'vite-plugin-svgr';
import { defineConfig } from 'vitest/config';

export default defineConfig({
    plugins: [
        react(),
        svgr({
            svgrOptions: {
                svgo: false,
                exportType: 'named',
            },
            include: '**/*.svg',
        }),
    ],
    test: {
        // Set PUBLIC_API_BASE_URL before any module is evaluated so that
        // api/utils.ts and api/client.ts read the correct base URL when they
        // are first imported.
        env: {
            PUBLIC_API_BASE_URL: 'http://localhost:7860',
        },
        environment: 'jsdom',
        environmentOptions: {
            jsdom: {
                // Match the base URL that MSW handlers are registered under so that
                // relative fetch calls (baseUrl: '') resolve to the same origin.
                url: 'http://localhost:7860/',
            },
        },
        // This is needed to use globals like describe or expect
        globals: true,
        include: ['./src/**/*.test.{ts,tsx}'],
        setupFiles: './src/setup-tests.ts',
        watch: false,
        server: {
            deps: {
                inline: [/@react-spectrum\/.*/, /@spectrum-icons\/.*/, /@adobe\/react-spectrum\/.*/, /@geti-ui\/.*/],
            },
        },
    },
});
