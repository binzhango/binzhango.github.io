export type Rating = 'again' | 'got-it';
export interface QuestionIndex { id: string; topic: string; text: string }
export interface Progress { version: 1; ratings: Record<string, Rating>; lastQuestion: string | null }
export const storageKey = 'bz:interview:progress:v1';
export const emptyProgress = (): Progress => ({ version: 1, ratings: {}, lastQuestion: null });

export function parseProgress(raw: string | null, validIds: Set<string>): Progress {
  try {
    const value = JSON.parse(raw ?? 'null');
    if (!value || value.version !== 1) return emptyProgress();
    const ratings: Record<string, Rating> = {};
    if (value.ratings && typeof value.ratings === 'object' && !Array.isArray(value.ratings)) {
      for (const [id, rating] of Object.entries(value.ratings)) {
        if (validIds.has(id) && (rating === 'again' || rating === 'got-it')) ratings[id] = rating;
      }
    }
    return { version: 1, ratings, lastQuestion: validIds.has(value.lastQuestion) ? value.lastQuestion : null };
  } catch { return emptyProgress(); }
}

export function filterQuestions(questions: QuestionIndex[], topic: string, query: string): QuestionIndex[] {
  const terms = query.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
  return questions.filter((question) => (!topic || question.topic === topic) && terms.every((term) => question.text.toLocaleLowerCase().includes(term)));
}

export function summarize(ids: string[], ratings: Progress['ratings']) {
  const again = ids.filter((id) => ratings[id] === 'again').length;
  const gotIt = ids.filter((id) => ratings[id] === 'got-it').length;
  return { again, gotIt, reviewed: again + gotIt, unreviewed: ids.length - again - gotIt };
}
