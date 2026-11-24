import { SideNav, SideNavItem } from '@astryxdesign/core/SideNav';
import { Icon } from '@astryxdesign/core/Icon';
import { useHoverCard } from '@astryxdesign/core/HoverCard';

function PostLink({ href, title, current, published }) {
  const preview = useHoverCard({
    placement: 'end',
    alignment: 'start',
    delay: 450,
    hideDelay: 250,
    touchTrigger: 'none',
  });

  return (
    <>
      <SideNavItem
        ref={preview.ref}
        href={href}
        label={title}
        size="sm"
        isSelected={current}
        aria-describedby={preview.describedBy}
      />
      {preview.renderHoverCard(
        <>
          <strong aria-hidden="true">{title}</strong>
          {published && <time dateTime={published.iso}>{published.label}</time>}
        </>,
        { className: 'sidebar-post-preview' },
      )}
    </>
  );
}

export default function PostSideNav({ groups }) {
  return (
    <SideNav id="blog-posts-nav" aria-label="Posts by year" className="post-side-nav">
      {groups.map(({ year, entries, open }) => (
        <details key={year} className="post-year" data-year={year} open={open}>
          <summary>
            <Icon icon="calendar" size="sm" color="inherit" />
            <span>{year}</span>
            <span className="post-year-count" aria-label={`${entries.length} posts`}>{entries.length}</span>
            <span className="post-year-chevron"><Icon icon="chevronRight" size="sm" color="inherit" /></span>
          </summary>
          <div className="post-year-entries">
            {entries.map(entry => <PostLink key={entry.href} {...entry} />)}
          </div>
        </details>
      ))}
    </SideNav>
  );
}
