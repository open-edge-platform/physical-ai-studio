export const getMainPageInProjectUrl = (pathname: string) => {
    const regexp = /\/projects\/[\w-]*\/([\w-]*)/g;
    const found = [...pathname.matchAll(regexp)];
    if (found.length) {
        const [, main] = found[0];
        if (main === 'cameras' || main === 'environments') {
            return 'robots';
        }
        return main;
    }

    return 'datasets';
};
