import { MetadataList, MetadataListItem } from '@astryxdesign/core/MetadataList';

export default function Metadata({ items, columns }) {
  return (
    <MetadataList columns={columns}>
      {items.map(({ label, value }) => (
        <MetadataListItem key={label} label={label}>{value}</MetadataListItem>
      ))}
    </MetadataList>
  );
}
