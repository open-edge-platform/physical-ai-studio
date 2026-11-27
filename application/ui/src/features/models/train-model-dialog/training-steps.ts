export const getEstimatedTrainingSteps = (
    episodeLengths: readonly number[],
    batchSize: number,
    epochs: number,
    valSplit = 0.1
): number | undefined => {
    if (episodeLengths.length === 0 || batchSize < 1 || epochs < 1) {
        return undefined;
    }

    const validationEpisodes = valSplit === 0 ? 0 : Math.max(1, Math.floor(episodeLengths.length * valSplit));
    const totalSamples = episodeLengths.reduce((sum, length) => sum + length, 0);
    const expectedTrainingSamples = totalSamples * (1 - validationEpisodes / episodeLengths.length);

    return Math.floor(expectedTrainingSamples / batchSize) * epochs;
};
