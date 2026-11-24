import { TopNav, TopNavItem } from '@astryxdesign/core/TopNav';

export default function PrimaryNav({ links }) {
  return (
    <TopNav label="Primary" className="primary-nav">
      {links.map(({ label, href, selected }) => (
        <TopNavItem key={href} label={label} href={href} isSelected={selected} />
      ))}
    </TopNav>
  );
}
