/**
 * Build-time feature flags.
 *
 * The env value is baked into the bundle at build time, so it can't be
 * changed from the browser console. For local testing, a `localStorage`
 * override takes precedence over the build-time value — from devtools:
 *
 *   setFeatureFlag('remoteTrainers', true)   // then reload the page
 *   setFeatureFlag('remoteTrainers', false)  // force-disable
 *   setFeatureFlag('remoteTrainers')         // clear override, use build default
 */

type FeatureFlagName = 'remoteTrainers';

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

export const featureFlags = {
    /**
     * Remote Trainers page and its nav tab, for offloading training jobs to
     * remote SSH-managed hosts. Disabled by default; set
     * `PUBLIC_ENABLE_REMOTE_TRAINERS=true` to enable it, or use
     * `setFeatureFlag('remoteTrainers', true)` in the browser console.
     */
    get remoteTrainers(): boolean {
        return resolveFlag(
            'remoteTrainers',
            typeof process !== 'undefined' ? process.env.PUBLIC_ENABLE_REMOTE_TRAINERS : undefined
        );
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
    // window.setFeatureFlag('remoteTrainers', true)
    window.setFeatureFlag = setFeatureFlag;
}
