import type { TestingLibraryMatchers } from '@testing-library/jest-dom/matchers';

declare module '@rstest/core' {
    // eslint-disable-next-line @typescript-eslint/no-empty-object-type
    interface Assertion<T = unknown> extends TestingLibraryMatchers<unknown, T> {}

    // eslint-disable-next-line @typescript-eslint/no-empty-object-type
    interface AsymmetricMatchersContaining extends TestingLibraryMatchers<unknown, unknown> {}
}
