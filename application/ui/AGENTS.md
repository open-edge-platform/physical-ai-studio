# UI Development Guide

## Quick start

- Node `>=24.2.0`, npm `>=11.14.0` (enforced by `engines` in `package.json`).
- Work from `application/ui/`.
- Install or refresh dependencies with `npm install`.
- Start the dev server with `npm run start` (proxies `/api` to `http://localhost:7860`).

## Source layout

```
src/
├── api/            OpenAPI spec, generated types, fetch client, MSW utils, error helpers
├── assets/         Images, SVGs, illustrations
├── components/     Shared, reusable UI components
├── features/       Domain feature modules (cameras/, datasets/, jobs/, models/, projects/, robots/, …)
├── query-client/   TanStack QueryClient factory and mutation-meta type definitions
├── routes/         Route-level page components (thin shells — logic lives in features/)
├── test-utils/     Custom RTL render utility for unit tests
├── index.tsx       App entry point
├── providers.tsx   Global providers (QueryClientProvider, ThemeProvider, RouterProvider)
├── router.tsx      createBrowserRouter config + exported `paths` map
└── utils.ts        Shared utility functions
```

Each directory under `features/` owns the components, hooks, and tests for one domain. Shared
hooks live co-located with the feature that owns them; there is no top-level `src/hooks/` folder.

## Tech stack

| Layer                | Library                                   |
| -------------------- | ----------------------------------------- |
| UI framework         | React 19                                  |
| Language             | TypeScript 5 (strict)                     |
| Build                | RSBuild                                   |
| Routing              | React Router v6 (`createBrowserRouter`)   |
| Server state         | TanStack Query v5 + `openapi-react-query` |
| Component library    | `@geti-ui/ui`                             |
| Styling              | CSS Modules (`.module.css`)               |
| Linting / formatting | ESLint + Prettier                         |

## Data fetching and API types

### The generated client

The backend's OpenAPI spec drives all API access. **Never call `fetch` directly from a component.**

```
src/api/openapi-spec.json   ← the spec (do not hand-edit)
src/api/openapi-spec.d.ts   ← generated TypeScript types (do not hand-edit)
```

`src/api/client.ts` exports two objects:

- **`$api`** — `openapi-react-query` client; use this in components for queries and mutations.
- **`fetchClient`** — raw `openapi-fetch` instance; use this outside React (e.g. route loaders, one-off calls).

```ts
import { $api } from '../../api/client';

// Suspending query — component suspends until data is available
const { data } = $api.useSuspenseQuery('get', '/api/projects/{project_id}/robots', {
    params: { path: { project_id } },
});

// Non-suspending query — returns { data, isPending, isError }
const { data } = $api.useQuery('get', '/api/projects/{project_id}/robots', {
    params: { path: { project_id } },
});

// Mutation — specify meta.invalidates to keep related queries fresh
const mutation = $api.useMutation('post', '/api/projects/{project_id}/environments', {
    meta: {
        invalidates: [['get', '/api/projects/{project_id}/environments', { params: { path: { project_id } } }]],
    },
});
```

`useSuspenseQuery` requires a `<Suspense>` ancestor. The root `<Outlet>` in `src/router.tsx` provides
one; the test render utility (`src/test-utils/render.tsx`) provides one for unit tests.

### Mutation cache auto-invalidation

`createQueryClient()` in `src/query-client/query-client.ts` wires a `MutationCache` that automatically
calls `queryClient.invalidateQueries` on mutation success. Use `meta.invalidates` for fire-and-forget
invalidation or `meta.awaits` for awaited invalidation — no manual `onSuccess` needed in most cases.

### Regenerating types

After the backend contract changes:

```bash
# Download the live spec from a running backend and rebuild types
npm run build:api:download && npm run build:api
```

Keep the updated `openapi-spec.json`, `openapi-spec.d.ts`, and the consuming UI changes in one commit.

## State management

| Kind of state           | Approach                        |
| ----------------------- | ------------------------------- |
| Server / async          | TanStack Query via `$api`       |
| Local UI state          | `useState` / `useReducer`       |
| Shared non-server state | React Context (`createContext`) |

Do not introduce Redux, Zustand, or any other state management library.

## Navigation and routing

Routes are declared in `src/router.tsx`. All path helpers are exported from there as `paths`:

```ts
import { paths } from '../../router';

// Type-safe URL construction
const url = paths.project.robots.show({ project_id, robot_id });
```

Route files under `src/routes/` should be thin shells. Extract non-trivial logic into the relevant
`src/features/` module. Use `NavLink` / `Link` for declarative navigation and `useNavigate` for
programmatic navigation.

Route params are read with `useParams`. Wrap them in a typed selector hook (e.g. `useProjectId`)
rather than reading `params.project_id` inline in components.

## Conventions

### TypeScript

- No `any`. Use `unknown` and narrow, or import a type from the OpenAPI-generated spec.
- Prefer `interface` for object shapes that benefit from `extends`-based composition — interfaces flatten
  to a single object type, catch property conflicts at definition time, and display more clearly in
  IDE tooltips.
- Use `type` for everything else: unions, tuples, function signatures, mapped/conditional types, and
  computed aliases that need caching.

### Components and hooks

- Function components and hooks only — no class components.
- Co-locate a component's hook with it in the same feature directory; hooks are not gathered into a
  top-level `src/hooks/` folder.
- New code goes under `src/features/<domain>/` for domain-specific work, or `src/components/` for
  things genuinely shared across features.
- Keep route files (`src/routes/`) minimal — extract complex rendering or data-fetching into the
  feature module.
- Export named, not default, exports.

### Imports

ESLint enforces these restrictions — the linter will error if violated:

```ts
// Wrong — use @geti-ui/ui instead

// Correct

import { Button, Icon } from '@geti-ui/ui';
import { Button } from '@react-spectrum/button';
import { SpectrumButtonProps } from '@react-types/button';
import Edit from '@spectrum-icons/workflow/Edit';
```

## Verification

Run these before opening a PR. All must pass.

```bash
npm run format:check       # Prettier
npm run lint               # ESLint (zero warnings allowed)
npm run cyclic-deps-check  # Circular dependency check (madge)
npm run type-check         # TypeScript (tsc --noEmit)
npm run test:unit          # Vitest unit tests
npm run build              # Production build (catch any bundler errors)
```

## Commands reference

| Task                         | Command                      |
| ---------------------------- | ---------------------------- |
| Install dependencies         | `npm install`                |
| Dev server                   | `npm run start`              |
| Production build             | `npm run build`              |
| Preview production build     | `npm run preview`            |
| Format (write)               | `npm run format`             |
| Format (check only)          | `npm run format:check`       |
| Lint                         | `npm run lint`               |
| Lint + auto-fix              | `npm run lint:fix`           |
| Circular-deps check          | `npm run cyclic-deps-check`  |
| Type-check                   | `npm run type-check`         |
| Unit tests (one-shot)        | `npm run test:unit`          |
| Unit tests (watch)           | `npm run test:unit:watch`    |
| Component tests (Playwright) | `npm run test:component`     |
| Download spec from backend   | `npm run build:api:download` |
| Regenerate types from spec   | `npm run build:api`          |

---

# Unit Testing Guide

## Stack

| Tool                                                                             | Role                                             |
| -------------------------------------------------------------------------------- | ------------------------------------------------ |
| [Vitest](https://vitest.dev/)                                                    | Test runner and assertions                       |
| [Testing Library](https://testing-library.com/docs/react-testing-library/intro/) | Component rendering and queries                  |
| [MSW v2](https://mswjs.io/)                                                      | API mocking via `setupServer`                    |
| [openapi-msw](https://github.com/christoph-fricke/openapi-msw)                   | Type-safe MSW handlers matching the OpenAPI spec |

## Running tests

```bash
# All unit tests (one-shot)
npm run test:unit

# Watch mode
npm run test:unit:watch

# Single file
npx vitest run src/features/robots/environment-form/submit-new-environment-button.test.tsx
```

## File conventions

- Co-locate test files with the source: `foo.tsx` → `foo.test.tsx`
- Use `.test.tsx` for files that render JSX, `.test.ts` otherwise.

## Key test infrastructure files

| File                        | Purpose                                                                    |
| --------------------------- | -------------------------------------------------------------------------- |
| `src/setup-tests.ts`        | Global setup: env vars, MSW server lifecycle                               |
| `src/msw-node-setup.ts`     | Exports the MSW `server` instance                                          |
| `src/test-utils/render.tsx` | Custom `render` / `renderHook` wrapping Router, QueryClient, ThemeProvider |
| `src/api/utils.ts`          | Exports `http` — the type-safe MSW handler builder                         |
| `src/api/client.ts`         | Exports `$api` — the openapi-react-query client used by components         |

## Writing a component test

### 1. Use the custom render utility

Always import `render` from `src/test-utils/render` instead of `@testing-library/react` directly. It wires up:

- A fresh `QueryClient` per test (no cache bleed between tests)
- `ThemeProvider` (required by `@geti-ui/ui` components)
- `<Suspense>` boundary (required when components use `$api.useSuspenseQuery`)
- A memory router with configurable route and path pattern

```tsx
import { render } from '../../../test-utils/render';

render(<MyComponent />, {
    route: '/projects/abc-123/environments/new',
    path: '/projects/:project_id/environments/new',
});
```

Provide `route` + `path` whenever the component (or any hook it calls) reads route params via `useParams` / `useProjectId`.

### 2. Mock API calls with MSW — never `vi.mock` the API client

Use `server.use()` to override handlers for a single test. Import `http` from `src/api/utils` for type-safe OpenAPI path handlers.

```tsx
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';

it('shows an empty state when no robots exist', async () => {
    server.use(http.get('/api/projects/{project_id}/robots', () => HttpResponse.json([])));

    render(<MyComponent />, { route: '/projects/p1/...', path: '/projects/:project_id/...' });

    expect(await screen.findByText('No robots')).toBeInTheDocument();
});
```

`server.resetHandlers()` runs automatically after each test (wired in `setup-tests.ts`), so per-test overrides are cleaned up without any extra work.

### 3. Wait for async queries

Components that use `$api.useSuspenseQuery` start in a suspended state. Use `findBy*` queries (which await DOM updates) rather than `getBy*`:

```tsx
// Good — waits for the component to finish loading
expect(await screen.findByRole('button', { name: /add environment/i })).toBeDisabled();

// Bad — query runs synchronously before the component has rendered
expect(screen.getByRole('button', { name: /add environment/i })).toBeDisabled();
```

### 4. Share a QueryClient across calls in one test

When you need multiple renders or actions within a single test to share the same query cache:

```tsx
import { createQueryClient } from '../../../query-client/query-client';

it('invalidates the list after creating an item', async () => {
    const queryClient = createQueryClient();
    const { user } = render(<Form />, { queryClient, route: '...', path: '...' });
    // ...
});
```

## Complete example

```tsx
import { screen } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { render } from '../../../test-utils/render';
import { EnvironmentFormProvider } from './provider';
import { SubmitNewEnvironmentButton } from './submit-new-environment-button';

const PROJECT_ID = 'test-project-id';
const ROBOTS_PATH = '/api/projects/{project_id}/robots';

const renderButton = (environment = {}) =>
    render(
        <EnvironmentFormProvider environment={{ name: '', robots: [], cameras: [], ...environment }}>
            <SubmitNewEnvironmentButton />
        </EnvironmentFormProvider>,
        {
            route: `/projects/${PROJECT_ID}/environments/new`,
            path: '/projects/:project_id/environments/new',
        }
    );

describe('SubmitNewEnvironmentButton', () => {
    it('is disabled when the project has robots but none were added to the environment', async () => {
        server.use(http.get(ROBOTS_PATH, () => HttpResponse.json([{ id: 'r1', type: 'SO101_Follower', name: 'Arm' }])));

        renderButton({ name: 'My Environment', robots: [] });

        expect(await screen.findByRole('button', { name: /add environment/i })).toBeDisabled();
    });
});
```

## Critical gotchas

### MSW must be started at module level in setup-tests.ts

`openapi-fetch` captures `globalThis.fetch` into a local variable the moment `createFetchClient` is called — which happens when `src/api/client.ts` is first imported. If `server.listen()` runs later (e.g. inside `beforeAll`), MSW patches `globalThis.fetch` too late and component fetches bypass the mock server entirely, hitting the real backend.

`src/setup-tests.ts` therefore calls `server.listen()` at the top level (not inside a hook). **Do not move it inside `beforeAll`.**

### Do not override `global.fetch` or `global.Request` with `node-fetch`

Node.js 24 + jsdom 29 provide native `fetch` and `Request` with proper `AbortSignal` support. `node-fetch`'s `Request` sets `signal = null`, which crashes MSW v2's internal `request.signal.addEventListener(...)` call. The native globals work correctly with MSW — no polyfills are needed.

### Use `http` from `src/api/utils`, not `msw`'s `http` directly

`src/api/utils` exports an `openapi-msw`-wrapped `http` helper. It uses the same `baseUrl` (`process.env.PUBLIC_API_BASE_URL`, set to `http://localhost:7860` in tests) that the production `fetchClient` uses. This ensures handler URLs match request URLs exactly.

```ts
// Correct — URLs match the fetchClient base URL

// Wrong — MSW's plain http uses a different base and may not match

import { http } from 'msw';

import { http } from '../../../api/utils';

server.use(http.get('/api/projects/{project_id}/robots', handler));

server.use(http.get('http://localhost:7860/api/projects/:project_id/robots', handler));
```

### `$api.useSuspenseQuery` requires a `<Suspense>` boundary

The custom `render` utility already wraps children in `<Suspense>`. If you bypass it and use RTL's `render` directly, you must add `<Suspense>` yourself or the component will throw during the initial suspended render.

---

# Component Tests (Playwright)

Component tests live under `tests/` and run with Playwright (`npm run test:component`). They exercise
rendered browser behaviour — use them when a unit test cannot adequately cover a user interaction or
visual state. Prefer unit tests for logic and data-fetching; prefer Playwright only when the change
meaningfully affects what appears in the browser.

---

# Guardrails

- Do not hand-edit `src/api/openapi-spec.json` or `src/api/openapi-spec.d.ts` — regenerate them.
- Do not import from `@react-spectrum/*`, `@react-types/*`, or `@spectrum-icons/*` — use `@geti-ui/ui`.
- Do not call `fetch` directly from components — use `$api`.
- Do not introduce a second data-fetching library, CSS-in-JS library, or global state manager.
- Do not use `any` in TypeScript — use `unknown` and narrow, or import a generated type.
