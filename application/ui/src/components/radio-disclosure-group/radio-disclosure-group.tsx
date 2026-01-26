import { ReactNode } from 'react';

import { Disclosure, DisclosurePanel, DisclosureTitle, Flex, Radio, RadioGroup, Text, View } from '@geti/ui';
import { clsx } from 'clsx';

import classes from './radio-disclosure-group.module.scss';

export const RadioDisclosure = <ValueType extends string>({
    value,
    setValue,
    items,
    ariaLabel,
}: {
    value: ValueType;
    setValue: (value: ValueType) => void;
    items: Array<{
        value: ValueType;
        label: ReactNode;
        icon?: ReactNode;
        content: ReactNode;
    }>;
    ariaLabel?: string;
}) => {
    return (
        <RadioGroup
            orientation='vertical'
            width='100%'
            onChange={(newValue) => {
                setValue(newValue as ValueType);
            }}
            aria-label={ariaLabel}
            value={value}
        >
            <Flex direction='column' gap='size-100'>
                {items.map((item) => {
                    const isExpanded = item.value === value;

                    return (
                        <Disclosure
                            key={item.value}
                            onExpandedChange={(expanded) => expanded && setValue(item.value)}
                            isExpanded={isExpanded}
                            UNSAFE_className={clsx(classes.disclosure, { [classes.selected]: isExpanded })}
                        >
                            <DisclosureTitle UNSAFE_className={classes.disclosureTitleContainer}>
                                <View>
                                    <Radio value={item.value} UNSAFE_className={classes.radio}>
                                        <Flex alignItems='center' gap='size-200'>
                                            {item.icon}

                                            <Text UNSAFE_className={classes.disclosureTitle}>{item.label}</Text>
                                        </Flex>
                                    </Radio>
                                </View>
                            </DisclosureTitle>
                            <DisclosurePanel marginX={'size-100'}>{item.content}</DisclosurePanel>
                        </Disclosure>
                    );
                })}
            </Flex>
        </RadioGroup>
    );
};
