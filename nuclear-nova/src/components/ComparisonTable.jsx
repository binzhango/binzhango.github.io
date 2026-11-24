import { Table, TableBody, TableCell, TableHeader, TableHeaderCell, TableRow } from '@astryxdesign/core/Table';

export default function ComparisonTable({ headers, rows, label, hasRowHeaders = false }) {
  return (
    <Table aria-label={label} density="compact" dividers="rows" isStriped verticalAlign="top">
      <TableHeader>
        <TableRow>
          {headers.map((header, index) => <TableHeaderCell key={index} scope="col">{header}</TableHeaderCell>)}
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((row, index) => (
          <TableRow key={index}>
            {row.map((cell, column) => hasRowHeaders && column === 0
              ? <TableHeaderCell key={column} scope="row">{cell}</TableHeaderCell>
              : <TableCell key={column}>{cell}</TableCell>)}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
