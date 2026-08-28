/**
 * Build-time feature flags.
 *
 * The env value is baked into the bundle at build time, so it can't be
 * changed from the browser console. For local testing, a `localStorage`
 * override takes precedence over the build-time value — from devtools:
 *
 *   setFeatureFlag('someFlag', true)   // then reload the page
 *   setFeatureFlag('someFlag', false)  // force-disable
 *   setFeatureFlag('someFlag')         // clear override, use build default
 */

type FeatureFlagName = 'ipcam';

const STORAGE_KEY_PREFIX = 'physicalai:featureFlags:';

const getEnvValue = (value: string | undefined): boolean => value === 'true' || value === '1';

const getStorageOverride = (name: FeatureFlagName): boolean | undefined => {
    if (typeof window === 'undefined') {
        return undefined;
    }

    try {
        const raw = window.localStorage.getItem(`${STORAGE_KEY_PREFIX}${name}`);

        if (raw === 'true') return true;
        if (raw === 'false') return false;
        return undefined;
    } catch {
        // localStorage can throw in locked-down/private browsing contexts.
        return undefined;
    }
};

const resolveFlag = (name: FeatureFlagName, envValue: string | undefined): boolean => {
    const override = getStorageOverride(name);

    return override ?? getEnvValue(envValue);
};

void resolveFlag;

/**
 * Add a getter here for each new flag, e.g.:
 *
 *   get someFlag(): boolean {
 *       const envValue = typeof process !== 'undefined' ? process.env.PUBLIC_ENABLE_SOME_FLAG : undefined;
 *       return resolveFlag('someFlag', envValue);
 *   },
 */
export const featureFlags = {
    get ipCamera(): boolean {
        return resolveFlag('ipcam', typeof process !== 'undefined' ? process.env.PUBLIC_ENABLE_IP_CAM : undefined);
    },
};

/**
 * Overrides a feature flag at runtime via `localStorage`, without needing a
 * rebuild. Persists across reloads until cleared. Call with no value (or
 * `undefined`) to remove the override and fall back to the build-time env var.
 */
export const setFeatureFlag = (name: FeatureFlagName, value?: boolean): void => {
    if (typeof window === 'undefined') {
        return;
    }

    const key = `${STORAGE_KEY_PREFIX}${name}`;

    if (value === undefined) {
        window.localStorage.removeItem(key);
    } else {
        window.localStorage.setItem(key, String(value));
    }
};

declare global {
    interface Window {
        setFeatureFlag?: typeof setFeatureFlag;
    }
}

if (typeof window !== 'undefined') {
    // Exposed so it's callable directly from devtools without an import:
    // window.setFeatureFlag('someFlag', true)
    window.setFeatureFlag = setFeatureFlag;
}
