import { createContext, ReactNode, useCallback, useContext, useMemo, useState } from 'react';

import { SchemaRobot, SchemaRobotInput, SchemaRobotType } from '../robot-types';
import { getInitialSO101FormData } from './catalog/so101';
import { getInitialWidowxFormData } from './catalog/widowxai';
import { getInitialBimanualFormData } from './catalog/widowxai-bimanual';
import { buildRobotBody, type AnyRobotFormData, type FormDataForSchema } from './form-data';

type RobotFormState = {
    activeType: SchemaRobotType;
    SO101_Follower: FormDataForSchema['SO101_Follower'];
    SO101_Leader: FormDataForSchema['SO101_Leader'];
    Trossen_WidowXAI_Follower: FormDataForSchema['Trossen_WidowXAI_Follower'];
    Trossen_WidowXAI_Leader: FormDataForSchema['Trossen_WidowXAI_Leader'];
    Trossen_Bimanual_WidowXAI_Follower: FormDataForSchema['Trossen_Bimanual_WidowXAI_Follower'];
    Trossen_Bimanual_WidowXAI_Leader: FormDataForSchema['Trossen_Bimanual_WidowXAI_Leader'];
};

const RobotFormContext = createContext<RobotFormState | null>(null);

type SetRobotFormContextType = {
    setActiveType: (type: SchemaRobotType) => void;
    updateFormData: <K extends SchemaRobotType>(
        schemaType: K,
        update: Partial<FormDataForSchema[K]> | ((prev: FormDataForSchema[K]) => FormDataForSchema[K])
    ) => void;
};

const SetRobotFormContext = createContext<SetRobotFormContextType | null>(null);

const FORM_DATA_FAMILY: Record<SchemaRobotType, string> = {
    SO101_Follower: 'so101',
    SO101_Leader: 'so101',
    Trossen_WidowXAI_Follower: 'widowx',
    Trossen_WidowXAI_Leader: 'widowx',
    Trossen_Bimanual_WidowXAI_Follower: 'bimanual',
    Trossen_Bimanual_WidowXAI_Leader: 'bimanual',
};

const getInitialState = (robot?: SchemaRobot): RobotFormState => ({
    activeType: robot?.type ?? 'SO101_Follower',
    SO101_Follower: getInitialSO101FormData(robot?.type === 'SO101_Follower' ? robot : undefined),
    SO101_Leader: getInitialSO101FormData(robot?.type === 'SO101_Leader' ? robot : undefined),
    Trossen_WidowXAI_Follower: getInitialWidowxFormData(
        robot?.type === 'Trossen_WidowXAI_Follower' ? robot : undefined
    ),
    Trossen_WidowXAI_Leader: getInitialWidowxFormData(robot?.type === 'Trossen_WidowXAI_Leader' ? robot : undefined),
    Trossen_Bimanual_WidowXAI_Follower: getInitialBimanualFormData(
        robot?.type === 'Trossen_Bimanual_WidowXAI_Follower' ? robot : undefined
    ),
    Trossen_Bimanual_WidowXAI_Leader: getInitialBimanualFormData(
        robot?.type === 'Trossen_Bimanual_WidowXAI_Leader' ? robot : undefined
    ),
});

export const RobotFormProvider = ({ children, robot }: { children: ReactNode; robot?: SchemaRobot }) => {
    const [state, setState] = useState(() => getInitialState(robot));

    const setActiveType = useCallback(<T extends SchemaRobotType>(type: T) => {
        setState((prev) => {
            if (prev.activeType === type) return prev;

            const oldSlice = prev[prev.activeType] as AnyRobotFormData;
            const sameFamily = FORM_DATA_FAMILY[prev.activeType] === FORM_DATA_FAMILY[type];

            if (sameFamily) {
                return { ...prev, activeType: type, [type]: { ...oldSlice } as FormDataForSchema[T] };
            }

            return { ...prev, activeType: type };
        });
    }, []);

    const updateFormData = useCallback(
        <K extends SchemaRobotType>(
            schemaType: K,
            update: Partial<FormDataForSchema[K]> | ((prev: FormDataForSchema[K]) => FormDataForSchema[K])
        ) => {
            setState((prev) => ({
                ...prev,
                [schemaType]:
                    typeof update === 'function'
                        ? (update as (prev: FormDataForSchema[K]) => FormDataForSchema[K])(
                              prev[schemaType] as FormDataForSchema[K]
                          )
                        : { ...(prev[schemaType] as FormDataForSchema[K]), ...update },
            }));
        },
        []
    );

    const setContextValue = useMemo(() => ({ setActiveType, updateFormData }), [setActiveType, updateFormData]);

    return (
        <RobotFormContext.Provider value={state}>
            <SetRobotFormContext.Provider value={setContextValue}>{children}</SetRobotFormContext.Provider>
        </RobotFormContext.Provider>
    );
};

export function useRobotForm(): { activeType: SchemaRobotType; robotForm: AnyRobotFormData };
export function useRobotForm<K extends SchemaRobotType>(
    schemaType: K
): { activeType: SchemaRobotType; robotForm: FormDataForSchema[K] };
export function useRobotForm(schemaType?: SchemaRobotType) {
    const context = useContext(RobotFormContext);

    if (context === null) {
        throw new Error('useRobotForm was used outside of RobotFormProvider');
    }

    const rt = schemaType ?? context.activeType;

    return { activeType: context.activeType, robotForm: context[rt] as AnyRobotFormData };
}

export const useSetRobotForm = () => {
    const context = useContext(SetRobotFormContext);

    if (context === null) {
        throw new Error('useSetRobotForm was used outside of RobotFormProvider');
    }

    return context;
};

export const useRobotFormFields = <T extends AnyRobotFormData = AnyRobotFormData>() => {
    const context = useContext(RobotFormContext);
    const { updateFormData } = useSetRobotForm();

    if (context === null) {
        throw new Error('useRobotFormFields was used outside of RobotFormProvider');
    }

    const formData = context[context.activeType] as T;

    const updateField = <K extends keyof T>(field: K, value: T[K]) => {
        updateFormData(context.activeType, { [field]: value } as Partial<FormDataForSchema[typeof context.activeType]>);
    };

    return { formData, activeType: context.activeType, updateField };
};

export const useRobotFormBody = (robot_id: string): SchemaRobotInput | null => {
    const state = useContext(RobotFormContext);

    if (state === null) {
        throw new Error('useRobotFormBody was used outside of RobotFormProvider');
    }

    const formData = state[state.activeType] as AnyRobotFormData;

    return buildRobotBody(formData, state.activeType, robot_id);
};
