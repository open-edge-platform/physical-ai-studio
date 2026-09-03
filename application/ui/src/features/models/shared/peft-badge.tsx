import { DORA_BADGE_TEXT, DORA_BADGE_TITLE, LORA_BADGE_COLOR, LORA_BADGE_TEXT, LORA_BADGE_TITLE } from './peft';
import { SingleBadge } from './split-badge';

/**
 * Marks a training job or model fine-tuned with LoRA/DoRA.
 *
 * Renders nothing when it is not, so callers can drop it into a row
 * unconditionally instead of repeating the guard.
 */
export const PeftBadge = ({ isEnabled, isDora }: { isEnabled: boolean | undefined; isDora: boolean | undefined }) => {
    if (!isEnabled) {
        return null;
    }

    return (
        <SingleBadge
            color={LORA_BADGE_COLOR}
            text={isDora ? DORA_BADGE_TEXT : LORA_BADGE_TEXT}
            title={isDora ? DORA_BADGE_TITLE : LORA_BADGE_TITLE}
            preserveCase
        />
    );
};
