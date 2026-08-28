import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';

import { render } from '../../test-utils/render';
import { Table, TableColumn } from './table';

import classes from './table.module.css';

const COLUMNS: TableColumn[] = [
    { width: '1fr', header: 'Name' },
    { width: '1fr', header: 'Status' },
    { width: 'auto', align: 'end' },
];

const EXPANDABLE_COLUMNS: TableColumn[] = [{ width: 'max-content' }, ...COLUMNS];

describe('Table', () => {
    it('renders one header cell per column entry, in order, with an empty cell for a spacer column', () => {
        render(
            <Table columns={COLUMNS}>
                <Table.Row>
                    <div>Alpha</div>
                    <div>Ready</div>
                    <div>Action</div>
                </Table.Row>
            </Table>
        );

        const headerCells = screen.getByText('Name').parentElement?.children;

        expect(headerCells).toBeDefined();
        // Name, Status, spacer (trailing action column has no header)
        expect(headerCells?.[0]).toHaveTextContent('Name');
        expect(headerCells?.[1]).toHaveTextContent('Status');
        expect(headerCells?.[2]).toHaveTextContent('');
    });

    it('derives grid-template-columns directly from the columns prop, with no hidden tracks', () => {
        render(
            <Table columns={COLUMNS}>
                <Table.Row>
                    <div>Alpha</div>
                    <div>Ready</div>
                    <div>Action</div>
                </Table.Row>
            </Table>
        );

        const table = screen.getByTestId('table');

        expect(table.style.gridTemplateColumns).toBe('1fr 1fr auto');
    });

    it('gives header and row containers matching subgrid declarations without explicit tracks', () => {
        render(
            <Table columns={COLUMNS}>
                <Table.Row id='row-1'>
                    <div>Alpha</div>
                    <div>Ready</div>
                    <div>Action</div>
                </Table.Row>
            </Table>
        );

        const header = screen.getByTestId('table-header');
        const row = screen.getByTestId('row-1');

        expect(header.style.gridTemplateColumns).toBe('');
        expect(row.style.gridTemplateColumns).toBe('');
    });

    it('applies align to both the header cell and the body cell of that column', () => {
        render(
            <Table columns={COLUMNS}>
                <Table.Row>
                    <div>Alpha</div>
                    <div>Ready</div>
                    <div>Action</div>
                </Table.Row>
            </Table>
        );

        const header = screen.getByTestId('table-header');
        const headerActionCell = header.lastElementChild as HTMLElement;
        const bodyActionCell = screen.getByText('Action').parentElement;

        expect(bodyActionCell).toHaveStyle({ justifySelf: 'end' });
        expect(headerActionCell).toHaveStyle({ justifySelf: 'end' });
    });

    it('starts an uncontrolled expandable row collapsed and toggles it via the disclosure button', async () => {
        const user = userEvent.setup();

        render(
            <Table columns={EXPANDABLE_COLUMNS}>
                <Table.ExpandableRow label='Alpha' detail={<div>Alpha details</div>}>
                    <div>Alpha</div>
                    <div>Ready</div>
                    <div>Action</div>
                </Table.ExpandableRow>
            </Table>
        );

        const toggle = screen.getByRole('button', { name: 'Show details for Alpha' });

        expect(toggle).toHaveAttribute('aria-expanded', 'false');
        expect(screen.queryByText('Alpha details')).not.toBeInTheDocument();

        await user.click(toggle);

        expect(toggle).toHaveAttribute('aria-expanded', 'true');
        expect(screen.getByText('Alpha details')).toBeInTheDocument();

        await user.click(toggle);

        expect(toggle).toHaveAttribute('aria-expanded', 'false');
        expect(screen.queryByText('Alpha details')).not.toBeInTheDocument();
    });

    it('starts expanded when defaultExpanded is set', () => {
        render(
            <Table columns={EXPANDABLE_COLUMNS}>
                <Table.ExpandableRow label='Alpha' detail={<div>Alpha details</div>} defaultExpanded>
                    <div>Alpha</div>
                    <div>Ready</div>
                    <div>Action</div>
                </Table.ExpandableRow>
            </Table>
        );

        expect(screen.getByRole('button', { name: 'Show details for Alpha' })).toHaveAttribute('aria-expanded', 'true');
        expect(screen.getByText('Alpha details')).toBeInTheDocument();
    });

    it('allows multiple uncontrolled rows to be expanded simultaneously', async () => {
        const user = userEvent.setup();

        render(
            <Table columns={EXPANDABLE_COLUMNS}>
                <Table.ExpandableRow label='Alpha' detail={<div>Alpha details</div>}>
                    <div>Alpha</div>
                    <div>Ready</div>
                    <div>Action</div>
                </Table.ExpandableRow>
                <Table.ExpandableRow label='Bravo' detail={<div>Bravo details</div>}>
                    <div>Bravo</div>
                    <div>Ready</div>
                    <div>Action</div>
                </Table.ExpandableRow>
            </Table>
        );

        await user.click(screen.getByRole('button', { name: 'Show details for Alpha' }));
        await user.click(screen.getByRole('button', { name: 'Show details for Bravo' }));

        expect(screen.getByText('Alpha details')).toBeInTheDocument();
        expect(screen.getByText('Bravo details')).toBeInTheDocument();
    });

    it('lets isExpanded drive rendering in controlled mode without changing on its own', async () => {
        const user = userEvent.setup();
        const onExpandedChange = vi.fn();

        const { rerender } = render(
            <Table columns={EXPANDABLE_COLUMNS}>
                <Table.ExpandableRow
                    label='Alpha'
                    detail={<div>Alpha details</div>}
                    isExpanded={false}
                    onExpandedChange={onExpandedChange}
                >
                    <div>Alpha</div>
                    <div>Ready</div>
                    <div>Action</div>
                </Table.ExpandableRow>
            </Table>
        );

        const toggle = screen.getByRole('button', { name: 'Show details for Alpha' });

        await user.click(toggle);

        expect(onExpandedChange).toHaveBeenCalledWith(true);
        expect(screen.queryByText('Alpha details')).not.toBeInTheDocument();

        rerender(
            <Table columns={EXPANDABLE_COLUMNS}>
                <Table.ExpandableRow
                    label='Alpha'
                    detail={<div>Alpha details</div>}
                    isExpanded={true}
                    onExpandedChange={onExpandedChange}
                >
                    <div>Alpha</div>
                    <div>Ready</div>
                    <div>Action</div>
                </Table.ExpandableRow>
            </Table>
        );

        expect(screen.getByText('Alpha details')).toBeInTheDocument();
    });

    it('renders no disclosure button and does not respond to clicks for a static row', async () => {
        const user = userEvent.setup();

        render(
            <Table columns={COLUMNS}>
                <Table.Row id='static-row'>
                    <div>Alpha</div>
                    <div>Ready</div>
                    <div>Action</div>
                </Table.Row>
            </Table>
        );

        expect(screen.queryByRole('button', { name: /show details for/i })).not.toBeInTheDocument();

        const row = screen.getByTestId('static-row');

        expect(row).not.toHaveAttribute('aria-expanded');
        expect(row).not.toHaveAttribute('aria-controls');

        await user.click(row);

        expect(screen.queryByText('Alpha details')).not.toBeInTheDocument();
    });

    it('names the disclosure button, exposes aria-expanded, and links aria-controls to the panel id', () => {
        render(
            <Table columns={EXPANDABLE_COLUMNS}>
                <Table.ExpandableRow label='Alpha' detail={<div>Alpha details</div>} defaultExpanded>
                    <div>Alpha</div>
                    <div>Ready</div>
                    <div>Action</div>
                </Table.ExpandableRow>
            </Table>
        );

        const toggle = screen.getByRole('button', { name: 'Show details for Alpha' });
        const controlsId = toggle.getAttribute('aria-controls');

        expect(controlsId).toBeTruthy();
        expect(toggle).toHaveAttribute('aria-expanded', 'true');
        expect(document.getElementById(controlsId as string)).toHaveTextContent('Alpha details');
    });

    it('can be expanded and collapsed via keyboard', async () => {
        const user = userEvent.setup();

        render(
            <Table columns={EXPANDABLE_COLUMNS}>
                <Table.ExpandableRow label='Alpha' detail={<div>Alpha details</div>}>
                    <div>Alpha</div>
                    <div>Ready</div>
                    <div>Action</div>
                </Table.ExpandableRow>
            </Table>
        );

        await user.tab();
        const toggle = screen.getByRole('button', { name: 'Show details for Alpha' });

        expect(toggle).toHaveFocus();

        await user.keyboard('{Enter}');
        expect(toggle).toHaveAttribute('aria-expanded', 'true');

        await user.keyboard(' ');
        expect(toggle).toHaveAttribute('aria-expanded', 'false');
    });

    it('toggles an expandable row on a whole-row click', async () => {
        const user = userEvent.setup();

        render(
            <Table columns={EXPANDABLE_COLUMNS}>
                <Table.ExpandableRow label='Alpha' detail={<div>Alpha details</div>} id='alpha-row'>
                    <div>Alpha</div>
                    <div>Ready</div>
                    <div>Action</div>
                </Table.ExpandableRow>
            </Table>
        );

        await user.click(screen.getByText('Alpha'));

        expect(screen.getByText('Alpha details')).toBeInTheDocument();
    });

    it('renders the after slot whether the row is expanded or collapsed', async () => {
        const user = userEvent.setup();

        render(
            <Table columns={EXPANDABLE_COLUMNS}>
                <Table.ExpandableRow
                    label='Alpha'
                    detail={<div>Alpha details</div>}
                    after={<div>Alpha progress</div>}
                    id='alpha-row'
                >
                    <div>Alpha</div>
                    <div>Ready</div>
                    <div>Action</div>
                </Table.ExpandableRow>
            </Table>
        );

        expect(screen.getByTestId('alpha-row-after')).toBeInTheDocument();
        expect(screen.getByText('Alpha progress')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Show details for Alpha' }));

        expect(screen.getByTestId('alpha-row-after')).toBeInTheDocument();
        expect(screen.getByText('Alpha progress')).toBeInTheDocument();
    });

    it('does not render the after wrapper when after is a falsy conditional expression', () => {
        render(
            <Table columns={EXPANDABLE_COLUMNS}>
                <Table.ExpandableRow
                    label='Alpha'
                    detail={<div>Alpha details</div>}
                    after={false && <div>Alpha progress</div>}
                    id='alpha-row'
                >
                    <div>Alpha</div>
                    <div>Ready</div>
                    <div>Action</div>
                </Table.ExpandableRow>
            </Table>
        );

        expect(screen.queryByTestId('alpha-row-after')).not.toBeInTheDocument();
    });

    it('omits the accent bar by default (isEmphasized defaults to false)', async () => {
        const user = userEvent.setup();

        render(
            <Table columns={EXPANDABLE_COLUMNS}>
                <Table.ExpandableRow label='Alpha' detail={<div>Alpha details</div>} id='alpha-row'>
                    <div>Alpha</div>
                    <div>Ready</div>
                    <div>Action</div>
                </Table.ExpandableRow>
            </Table>
        );

        await user.click(screen.getByRole('button', { name: 'Show details for Alpha' }));

        expect(screen.getByTestId('alpha-row').className).not.toContain(classes.accentBar);
    });

    it('shows the accent bar when isEmphasized is true', async () => {
        const user = userEvent.setup();

        render(
            <Table columns={EXPANDABLE_COLUMNS} isEmphasized>
                <Table.ExpandableRow label='Alpha' detail={<div>Alpha details</div>} id='alpha-row'>
                    <div>Alpha</div>
                    <div>Ready</div>
                    <div>Action</div>
                </Table.ExpandableRow>
            </Table>
        );

        await user.click(screen.getByRole('button', { name: 'Show details for Alpha' }));

        expect(screen.getByTestId('alpha-row').className).toContain(classes.accentBar);
    });

    it('logs a console.error when the cell count differs from the column count', () => {
        const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined);

        render(
            <Table columns={COLUMNS}>
                <Table.Row>
                    <div>Alpha</div>
                </Table.Row>
            </Table>
        );

        expect(consoleError).toHaveBeenCalledWith(expect.stringContaining('1 cell(s)'));

        consoleError.mockRestore();
    });
});
