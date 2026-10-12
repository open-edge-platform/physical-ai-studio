import { ReactNode } from 'react';

import {
    AriaListBox,
    AriaListBoxItem,
    DimensionValue,
    GridLayout,
    GridLayoutOptions,
    Selection,
    View,
    Virtualizer,
} from '@geti-ui/ui';

import classes from './virtualized-grid-layout.module.css';

interface VirtualizedGridLayoutProps<Item> {
    items: Item[];
    ariaLabel: string;
    layoutOptions: GridLayoutOptions;
    idFormatter: (item: Item) => string;
    textValueFormatter: (item: Item) => string;
    renderItem: (item: Item, index: number) => ReactNode;
    selectionMode?: 'none' | 'single' | 'multiple';
    selectedKeys?: Selection;
    onSelectionChange?: (keys: Selection) => void;
    disallowEmptySelection?: boolean;
    containerHeight?: DimensionValue;
}

export const VirtualizedGridLayout = <Item,>({
    items,
    ariaLabel,
    layoutOptions,
    idFormatter,
    textValueFormatter,
    renderItem,
    selectionMode = 'none',
    selectedKeys,
    onSelectionChange,
    disallowEmptySelection,
    containerHeight = '100%',
}: VirtualizedGridLayoutProps<Item>) => {
    return (
        <View
            height={containerHeight}
            UNSAFE_style={{
                width: '100%',
            }}
        >
            <Virtualizer layout={GridLayout} layoutOptions={layoutOptions}>
                <AriaListBox
                    layout='grid'
                    aria-label={ariaLabel}
                    className={classes.listBox}
                    selectionMode={selectionMode}
                    selectedKeys={selectedKeys}
                    onSelectionChange={onSelectionChange}
                    disallowEmptySelection={disallowEmptySelection}
                >
                    {items.map((item, index) => (
                        <AriaListBoxItem
                            id={idFormatter(item)}
                            key={idFormatter(item)}
                            textValue={textValueFormatter(item)}
                        >
                            {renderItem(item, index)}
                        </AriaListBoxItem>
                    ))}
                </AriaListBox>
            </Virtualizer>
        </View>
    );
};
