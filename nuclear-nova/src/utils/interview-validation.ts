import { interviewTopicIds } from './interview-topics';

export interface InterviewRecord {
  id: string;
  body?: string;
  data: { questionId: string; aliases?: string[]; topic: string; order: number; status: string; reviewedOn: string | null };
}

/** Fail a production build rather than silently publish missing/unreviewed answers. */
export function validateInterviewBank(records: InterviewRecord[], requireReviewed = true) {
  const ids = new Set<string>();
  const orders = new Set<number>();
  const errors: string[] = [];
  const canonicalIds = new Set(records.map((record) => record.id));
  const aliases = new Set<string>();
  for (const record of records) {
    const { id, body = '', data } = record;
    if (ids.has(id) || data.questionId !== id) errors.push(`${id}: duplicate or mismatched permanent ID`);
    ids.add(id);
    for (const alias of data.aliases ?? []) {
      if (canonicalIds.has(alias) || aliases.has(alias)) errors.push(`${id}: duplicate question alias ${alias}`);
      aliases.add(alias);
    }
    if (orders.has(data.order)) errors.push(`${id}: duplicate editorial order`);
    orders.add(data.order);
    if (!interviewTopicIds.includes(data.topic)) errors.push(`${id}: unknown topic`);
    const names = ['Question', 'Brief answer', 'Detailed answer'];
    const parts = body.split(/^## (Question|Brief answer|Detailed answer)\s*$/m);
    const found: string[] = [];
    for (let i = 1; i < parts.length; i += 2) {
      found.push(parts[i]);
      if (!parts[i + 1]?.trim()) errors.push(`${id}: empty ${parts[i]}`);
    }
    if (found.join('|') !== names.join('|')) errors.push(`${id}: expected Question, Brief answer, Detailed answer exactly once in order`);
    if (requireReviewed && (data.status !== 'reviewed' || !/^\d{4}-\d{2}-\d{2}$/.test(data.reviewedOn ?? ''))) errors.push(`${id}: answer not reviewed`);
  }
  if (!records.length) errors.push('Question bank is empty');
  if (errors.length) throw new Error(`Interview content validation failed (${errors.length} issues):\n${errors.slice(0, 15).join('\n')}`);
}
