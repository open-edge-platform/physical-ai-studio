export const TRAINING_VAL_SPLIT = 0.1;

export const getEstimatedTrainingSteps = (
    episodeLengths: readonly number[],
    batchSize: number,
    epochs: number
): number | undefined => {
    if (episodeLengths.length === 0 || batchSize < 1 || epochs < 1) {
        return undefined;
    }

    const validationEpisodes = Math.max(1, Math.floor(episodeLengths.length * TRAINING_VAL_SPLIT));
    const totalSamples = episodeLengths.reduce((sum, length) => sum + length, 0);
    const expectedTrainingSamples = totalSamples * (1 - validationEpisodes / episodeLengths.length);

    return Math.floor(expectedTrainingSamples / batchSize) * epochs;
};
