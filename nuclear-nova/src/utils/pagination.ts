export const POSTS_PER_PAGE = 10;

export function getBlogPagePath(page: number): string {
  return page === 1 ? '/posts/' : `/posts/page/${page}/`;
}

export function paginatePosts<T>(posts: T[]) {
  const totalPages = Math.max(1, Math.ceil(posts.length / POSTS_PER_PAGE));
  return Array.from({ length: totalPages }, (_, index) => ({
    posts: posts.slice(index * POSTS_PER_PAGE, (index + 1) * POSTS_PER_PAGE),
    currentPage: index + 1,
    totalPages,
    totalPosts: posts.length,
    start: posts.length ? index * POSTS_PER_PAGE + 1 : 0,
    end: Math.min((index + 1) * POSTS_PER_PAGE, posts.length),
  }));
}

export function getPageLinks(currentPage: number, totalPages: number): (number | 'ellipsis')[] {
  const pages = totalPages <= 7
    ? Array.from({ length: totalPages }, (_, index) => index + 1)
    : [...new Set([1, currentPage - 1, currentPage, currentPage + 1, totalPages])]
      .filter((page) => page >= 1 && page <= totalPages).sort((a, b) => a - b);
  return pages.flatMap((page, index): (number | 'ellipsis')[] =>
    index > 0 && page - pages[index - 1] > 1 ? ['ellipsis', page] : [page]);
}
