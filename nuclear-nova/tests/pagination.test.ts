import { describe, expect, it } from 'vitest';
import { getBlogPagePath, getPageLinks, paginatePosts } from '../src/utils/pagination';

describe('blog pagination', () => {
  it('keeps all posts in their original order, exactly once', () => {
    const posts = Array.from({ length: 64 }, (_, index) => `post-${index}`);
    const pages = paginatePosts(posts);
    expect(pages.map((page) => page.posts.length)).toEqual([10, 10, 10, 10, 10, 10, 4]);
    expect(pages.flatMap((page) => page.posts)).toEqual(posts);
    expect(pages[6]).toMatchObject({ currentPage: 7, totalPages: 7, totalPosts: 64, start: 61, end: 64 });
  });
  it('handles an empty blog and exact page boundaries without extra pages', () => {
    expect(paginatePosts([])).toEqual([{ posts: [], currentPage: 1, totalPages: 1, totalPosts: 0, start: 0, end: 0 }]);
    expect(paginatePosts(Array(20).fill('post'))).toHaveLength(2);
  });
  it('keeps the original index URL for the first page', () => {
    expect(getBlogPagePath(1)).toBe('/posts/');
    expect(getBlogPagePath(2)).toBe('/posts/page/2/');
  });
  it('keeps navigation bounded as the blog grows', () => {
    expect(getPageLinks(50, 100)).toEqual([1, 'ellipsis', 49, 50, 51, 'ellipsis', 100]);
    expect(getPageLinks(1, 100)).toEqual([1, 2, 'ellipsis', 100]);
    expect(getPageLinks(100, 100)).toEqual([1, 'ellipsis', 99, 100]);
    expect(getPageLinks(3, 7)).toEqual([1, 2, 3, 4, 5, 6, 7]);
  });
});
