#!/usr/bin/env node

import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');
const postsRoot = path.join(projectRoot, 'src', 'content', 'docs', 'posts');

function parseArgs(argv) {
  const args = {
    extension: 'md',
    dryRun: false,
    title: '',
    slug: '',
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === '--mdx') {
      args.extension = 'mdx';
      continue;
    }

    if (arg === '--md') {
      args.extension = 'md';
      continue;
    }

    if (arg === '--dry-run') {
      args.dryRun = true;
      continue;
    }

    if (arg === '--title') {
      args.title = argv[index + 1] ?? '';
      index += 1;
      continue;
    }

    if (arg === '--slug') {
      args.slug = argv[index + 1] ?? '';
      index += 1;
      continue;
    }

    if (!arg.startsWith('--') && !args.title) {
      args.title = arg;
    }
  }

  return args;
}

function pad(value) {
  return String(value).padStart(2, '0');
}

function slugify(value) {
  return value
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/-{2,}/g, '-');
}

function titleFromSlug(slug) {
  return slug
    .split('-')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function buildTemplate({ title, isoDate }) {
  return `---
title: "${title}"
date: ${isoDate}
author: BZ
description: ""
categories: []
tags: []
---

<!-- more -->

## Draft

`;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const now = new Date();
  const year = String(now.getFullYear());
  const month = pad(now.getMonth() + 1);
  const day = pad(now.getDate());
  const isoDate = `${year}-${month}-${day}`;
  const rawSlug = options.slug || slugify(options.title || 'new-post');

  if (!rawSlug) {
    console.error('Could not derive a filename slug. Provide a title or use --slug.');
    process.exitCode = 1;
    return;
  }

  const title = options.title || titleFromSlug(rawSlug);
  const fileName = `${month}-${day}-${rawSlug}.${options.extension}`;
  const yearDir = path.join(postsRoot, year);
  const targetPath = path.join(yearDir, fileName);
  const content = buildTemplate({ title, isoDate });

  if (options.dryRun) {
    console.log(targetPath);
    return;
  }

  await mkdir(yearDir, { recursive: true });
  await writeFile(targetPath, content, { flag: 'wx' });

  console.log(`Created ${targetPath}`);
}

main().catch((error) => {
  if (error && typeof error === 'object' && 'code' in error && error.code === 'EEXIST') {
    console.error('Post file already exists. Use a different slug or date.');
    process.exitCode = 1;
    return;
  }

  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
