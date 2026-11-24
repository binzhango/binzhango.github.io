import { List, ListItem } from '@astryxdesign/core/List';

export default function InterviewTopics({ topics, total }) {
  return (
    <nav className="interview-rail" aria-label="Question topics">
      <p>{total} questions</p>
      <List density="compact" edge="flush" hasDividers>
        {topics.map(({ id, title, count }) => (
          <ListItem
            key={id}
            label={<a href={`#topic-${id}`}><span>{title}</span><small>{count}</small></a>}
          />
        ))}
      </List>
    </nav>
  );
}
