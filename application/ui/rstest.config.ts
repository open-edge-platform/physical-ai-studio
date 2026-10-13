import { withRsbuildConfig } from '@rstest/adapter-rsbuild';
import { defineConfig } from '@rstest/core';

export default defineConfig({
    extends: withRsbuildConfig({
        modifyRsbuildConfig(config) {
            const define = { ...config.source?.define };

            // The application build replaces process.env with an empty object
            // because all public values are compiled in. Tests set feature flags
            // through process.env at runtime, so preserve Node's real process.
            delete define['process.env'];
            delete define['process.env.PUBLIC_API_BASE_URL'];
            delete define['import.meta.env.PUBLIC_API_BASE_URL'];

            return {
                ...config,
                source: {
                    ...config.source,
                    define,
                },
            };
        },
    }),
    env: {
        PUBLIC_API_BASE_URL: 'http://localhost:7860',
    },
    globals: true,
    include: ['./src/**/*.test.{ts,tsx}'],
    setupFiles: './src/setup-tests.ts',
    testEnvironment: {
        name: 'jsdom',
        options: {
            url: 'http://localhost:7860/',
        },
    },
});
