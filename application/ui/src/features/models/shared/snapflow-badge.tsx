import { SNAPFLOW_BADGE_COLOR, SNAPFLOW_BADGE_TEXT, SNAPFLOW_BADGE_TITLE } from './snapflow';
import { SingleBadge } from './split-badge';

/**
 * Marks a training job or model whose checkpoint is SnapFlow-distilled.
 *
 * Renders nothing when it is not, so callers can drop it into a row
 * unconditionally instead of repeating the guard.
 */
export const SnapflowBadge = ({ isEnabled }: { isEnabled: boolean | undefined }) => {
    if (!isEnabled) {
        return null;
    }

    return (
        <SingleBadge
            color={SNAPFLOW_BADGE_COLOR}
            text={SNAPFLOW_BADGE_TEXT}
            title={SNAPFLOW_BADGE_TITLE}
            preserveCase
        />
    );
};
