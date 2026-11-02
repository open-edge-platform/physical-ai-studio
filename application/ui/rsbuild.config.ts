import { defineConfig, loadEnv } from '@rsbuild/core';
import { pluginReact } from '@rsbuild/plugin-react';
import { pluginSvgr } from '@rsbuild/plugin-svgr';

const { publicVars, rawPublicVars } = loadEnv({ prefixes: ['PUBLIC_'] });
const apiProxyTarget = rawPublicVars.PUBLIC_API_PROXY_TARGET ?? 'http://localhost:7860';

export default defineConfig({
    plugins: [
        pluginReact({
            reactCompiler: true,
        }),

        pluginSvgr({
            svgrOptions: {
                exportType: 'named',
            },
        }),
    ],

    source: {
        define: {
            ...publicVars,
            'import.meta.env.PUBLIC_API_BASE_URL':
                publicVars['import.meta.env.PUBLIC_API_BASE_URL'] ?? '"http://localhost:3000"',
            'process.env.PUBLIC_API_BASE_URL':
                publicVars['process.env.PUBLIC_API_BASE_URL'] ?? '"http://localhost:3000"',
            // Needed to prevent an issue with spectrum's picker
            // eslint-disable-next-line max-len
            // https://github.com/adobe/react-spectrum/blob/6173beb4dad153aef74fc81575fd97f8afcf6cb3/packages/%40react-spectrum/overlays/src/OpenTransition.tsx#L40
            'process.env': {},
        },
    },
    html: {
        title: 'Physical AI Studio',
        favicon: './src/assets/icons/physicalai-studio-logo.svg',
    },
    tools: {
        rspack: {
            // Scalar intentionally imports user-provided plugins at runtime. Rspack
            // cannot statically resolve that expression, but it is not bundled and
            // does not affect the API reference's normal operation.
            ignoreWarnings: [
                {
                    module: /@scalar[\\/]api-reference[\\/]dist[\\/]standalone[\\/]lib[\\/]load-plugins-from-urls\.js$/,
                    message: /Critical dependency: the request of a dependency is an expression/,
                },
            ],
            watchOptions: {
                ignored: ['**/src-tauri/**'],
            },
        },
    },
    server: {
        proxy: {
            '/api': {
                target: apiProxyTarget,
                changeOrigin: true,
                ws: true,
                //pathRewrite: { '^/api': '' }, // strip the /api prefix
            },
        },
    },
});
