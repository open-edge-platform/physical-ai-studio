import { Children, createContext, CSSProperties, ReactNode, useContext, useId, useState } from 'react';

import { ActionButton, Icon } from '@geti-ui/ui';
import { ChevronRightSmallLight } from '@geti-ui/ui/icons';
import { clsx } from 'clsx';

import classes from './table.module.css';

export type TableColumn = {
    width: string;
    header?: ReactNode;
    align?: 'start' | 'center' | 'end';
};

export type TableProps = {
    columns: TableColumn[];
    /** Whether an expanded row shows the energy-blue accent bar. Defaults to `false`. */
    isEmphasized?: boolean;
    children: ReactNode;
};

type TableContextValue = {
    columns: TableColumn[];
    isEmphasized: boolean;
};

const TableContext = createContext<TableContextValue>({ columns: [], isEmphasized: false });

const alignStyle = (align: TableColumn['align']): CSSProperties | undefined =>
    align !== undefined && align !== 'start' ? { justifySelf: align } : undefined;

const renderCells = (children: ReactNode, columns: TableColumn[]) => {
    const cells = Children.toArray(children);

    if (cells.length !== columns.length) {
        console.error(
            `Table row received ${cells.length} cell(s) but ${columns.length} column(s) were declared. ` +
                'Every row must supply exactly one cell per column.'
        );
    }

    return cells.map((cell, index) => {
        const column = columns[index];
        const style = alignStyle(column?.align);

        if (style === undefined) {
            return cell;
        }

        return (
            <div key={index} style={style}>
                {cell}
            </div>
        );
    });
};

export type TableRowBaseProps = {
    children: ReactNode;
    after?: ReactNode;
    id?: string;
};

export type TableRowProps = TableRowBaseProps;

const Row = ({ children, after, id }: TableRowProps) => {
    const { columns } = useContext(TableContext);

    return (
        <>
            <div className={classes.row} data-testid={id}>
                {renderCells(children, columns)}
            </div>
            {Boolean(after) && (
                <div className={classes.after} data-testid={id ? `${id}-after` : 'after'}>
                    {after}
                </div>
            )}
        </>
    );
};

Row.displayName = 'Table.Row';

export type ExpandableRowBaseProps = TableRowBaseProps & {
    detail: ReactNode;
    label: string;
};

export type ControlledExpandableRowProps = ExpandableRowBaseProps & {
    isExpanded: boolean;
    onExpandedChange: (isExpanded: boolean) => void;
    defaultExpanded?: never;
};

export type UncontrolledExpandableRowProps = ExpandableRowBaseProps & {
    defaultExpanded?: boolean;
    isExpanded?: never;
    onExpandedChange?: never;
};

export type TableExpandableRowProps = ControlledExpandableRowProps | UncontrolledExpandableRowProps;

const ExpandableRow = ({
    children,
    after,
    id,
    detail,
    label,
    onExpandedChange,
    isExpanded: isExpandedProp,
    defaultExpanded,
}: TableExpandableRowProps) => {
    const { columns, isEmphasized } = useContext(TableContext);
    const contentId = useId();

    const isControlled = onExpandedChange !== undefined;
    const [internalExpanded, setInternalExpanded] = useState(defaultExpanded ?? false);
    const isExpanded = isControlled ? Boolean(isExpandedProp) : internalExpanded;

    const toggle = () => {
        if (isControlled) {
            onExpandedChange(!isExpanded);
        } else {
            setInternalExpanded((current) => !current);
        }
    };

    return (
        <>
            <div
                className={clsx(classes.row, classes.rowClickable, {
                    [classes.rowExpanded]: isExpanded,
                    [classes.accentBar]: isExpanded && isEmphasized,
                })}
                data-testid={id}
                onClick={toggle}
            >
                <ActionButton
                    isQuiet
                    aria-expanded={isExpanded}
                    aria-controls={contentId}
                    aria-label={`Show details for ${label}`}
                    onPress={toggle}
                    UNSAFE_className={classes.disclosureButton}
                >
                    <Icon>
                        <ChevronRightSmallLight />
                    </Icon>
                </ActionButton>
                {renderCells(children, columns.slice(1))}
            </div>
            {isExpanded && (
                <div id={contentId} className={clsx(classes.panel, { [classes.accentBar]: isEmphasized })}>
                    {detail}
                </div>
            )}
            {Boolean(after) && (
                <div className={classes.after} data-testid={id ? `${id}-after` : 'after'}>
                    {after}
                </div>
            )}
        </>
    );
};

ExpandableRow.displayName = 'Table.ExpandableRow';

const TableRoot = ({ columns, isEmphasized = false, children }: TableProps) => {
    const gridTemplateColumns = columns.map((column) => column.width).join(' ');

    return (
        <div className={classes.table} style={{ gridTemplateColumns }} data-testid='table'>
            <div className={classes.header} data-testid='table-header'>
                {columns.map((column, index) => (
                    <div key={index} style={alignStyle(column.align)}>
                        {column.header}
                    </div>
                ))}
            </div>
            <TableContext.Provider value={{ columns, isEmphasized }}>{children}</TableContext.Provider>
        </div>
    );
};

TableRoot.displayName = 'Table';

export const Table = Object.assign(TableRoot, { Row, ExpandableRow });
