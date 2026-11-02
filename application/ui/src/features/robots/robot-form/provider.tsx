import { createContext, ReactNode, RefObject, useContext, useMemo, useRef, useState } from 'react';

import { TextFieldRef } from '@geti-ui/ui';

import type { SchemaRobotInput } from '../robot-types';

export type RobotPayload = Record<string, unknown>;
export type RobotFormData = { name: string; payload: RobotPayload };

type RobotFormState = {
    activeType: string | undefined;
    name: string;
    payload: RobotPayload;
};

type RobotFormContextValue = RobotFormState & {
    setActiveType: (type: string) => void;
    setName: (name: string) => void;
    setPayload: (payload: RobotPayload) => void;
    updatePayloadField: (field: string, value: unknown) => void;
    nameFieldRef: RefObject<TextFieldRef | null>;
};

const RobotFormContext = createContext<RobotFormContextValue | null>(null);

export const RobotFormProvider = ({
    children,
    robot,
}: {
    children: ReactNode;
    robot?: { type: string; name: string; payload: unknown };
}) => {
    const [state, setState] = useState<RobotFormState>(() => ({
        activeType: robot?.type,
        name: robot?.name ?? '',
        payload: isPayload(robot?.payload) ? robot.payload : {},
    }));

    const nameFieldRef = useRef<TextFieldRef | null>(null);

    const value = useMemo<RobotFormContextValue>(
        () => ({
            ...state,
            setActiveType: (activeType) => setState((previous) => ({ ...previous, activeType, payload: {} })),
            setName: (name) => setState((previous) => ({ ...previous, name })),
            setPayload: (payload) => setState((previous) => ({ ...previous, payload })),
            updatePayloadField: (field, fieldValue) =>
                setState((previous) => ({ ...previous, payload: { ...previous.payload, [field]: fieldValue } })),
            nameFieldRef,
        }),
        [nameFieldRef, state]
    );

    return <RobotFormContext.Provider value={value}>{children}</RobotFormContext.Provider>;
};

const isPayload = (value: unknown): value is RobotPayload =>
    typeof value === 'object' && value !== null && !Array.isArray(value);

export const useRobotForm = () => {
    const context = useContext(RobotFormContext);
    if (context === null) {
        throw new Error('useRobotForm was used outside of RobotFormProvider');
    }
    return { ...context, robotForm: { name: context.name, payload: context.payload } as RobotFormData };
};

export const useRobotFormBody = (robot_id: string): SchemaRobotInput | null => {
    const { activeType, name, payload } = useRobotForm();
    if (activeType === undefined || name.trim() === '') {
        return null;
    }

    return { id: robot_id, name, type: activeType, payload } as unknown as SchemaRobotInput;
};
