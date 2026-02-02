import { getCollection, type CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'docs'>;

export interface PostsByYear {
    [year: string]: Post[];
}

export interface PostsByCategory {
    [category: string]: Post[];
}

export interface PostsByTag {
    [tag: string]: Post[];
}

/**
 * Get all posts sorted by date (newest first)
 * Filters for posts in the /posts/ directory and sorts by date
 */
export async function getAllPosts(): Promise<Post[]> {
    const allDocs = await getCollection('docs');
    
    // Filter for posts (entries in posts/ directory)
    // Use id instead of slug
    const posts = allDocs.filter(doc => doc.id && doc.id.startsWith('posts/'));
    
    // Sort by date (newest first)
    return posts.sort((a, b) => {
        const dateA = a.data.date ? new Date(a.data.date).getTime() : 0;
        const dateB = b.data.date ? new Date(b.data.date).getTime() : 0;
        return dateB - dateA;
    });
}

/**
 * Group posts by year for archive
 * Returns an object with years as keys and arrays of posts as values
 */
export async function getPostsByYear(): Promise<PostsByYear> {
    const posts = await getAllPosts();
    const postsByYear: PostsByYear = {};
    
    for (const post of posts) {
        if (post.data.date) {
            const year = new Date(post.data.date).getFullYear().toString();
            if (!postsByYear[year]) {
                postsByYear[year] = [];
            }
            postsByYear[year].push(post);
        }
    }
    
    return postsByYear;
}

/**
 * Group posts by category
 * Returns an object with categories as keys and arrays of posts as values
 */
export async function getPostsByCategory(): Promise<PostsByCategory> {
    const posts = await getAllPosts();
    const postsByCategory: PostsByCategory = {};
    
    for (const post of posts) {
        if (post.data.categories) {
            for (const category of post.data.categories) {
                if (!postsByCategory[category]) {
                    postsByCategory[category] = [];
                }
                postsByCategory[category].push(post);
            }
        }
    }
    
    return postsByCategory;
}

/**
 * Group posts by tag
 * Returns an object with tags as keys and arrays of posts as values
 */
export async function getPostsByTag(): Promise<PostsByTag> {
    const posts = await getAllPosts();
    const postsByTag: PostsByTag = {};
    
    for (const post of posts) {
        if (post.data.tags) {
            for (const tag of post.data.tags) {
                if (!postsByTag[tag]) {
                    postsByTag[tag] = [];
                }
                postsByTag[tag].push(post);
            }
        }
    }
    
    return postsByTag;
}

/**
 * Generate URL from date (yyyy/MM format)
 * @param date - The post date
 * @param slug - The post slug (without posts/ prefix)
 * @returns URL in format /posts/yyyy/MM/slug
 */
export function generatePostUrl(date: Date, slug: string): string {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    
    // Remove 'posts/' prefix if present
    const cleanSlug = slug.replace(/^posts\//, '');
    
    return `/posts/${year}/${month}/${cleanSlug}`;
}

/**
 * Get previous and next posts for navigation
 * @param currentSlug - The slug/id of the current post
 * @returns Object with prev and next posts (null if at boundaries)
 */
export async function getAdjacentPosts(currentSlug: string): Promise<{
    prev: Post | null;
    next: Post | null;
}> {
    const posts = await getAllPosts();
    const currentIndex = posts.findIndex(post => post.id === currentSlug || post.slug === currentSlug);
    
    if (currentIndex === -1) {
        return { prev: null, next: null };
    }
    
    return {
        // Previous post is the one before in the sorted array (newer)
        prev: currentIndex > 0 ? posts[currentIndex - 1] : null,
        // Next post is the one after in the sorted array (older)
        next: currentIndex < posts.length - 1 ? posts[currentIndex + 1] : null,
    };
}
