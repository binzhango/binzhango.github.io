#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
POSTS_ROOT="${PROJECT_ROOT}/src/content/docs/posts"

extension="md"
dry_run="false"
title=""
slug=""

usage() {
  cat <<'EOF'
Usage:
  init-blog-post.sh "post title" [--md|--mdx] [--dry-run]
  init-blog-post.sh --title "post title" [--slug custom-name] [--md|--mdx]

Options:
  --title <title>   Set the post title.
  --slug <slug>     Override the generated filename slug.
  --md              Create a Markdown file. Default.
  --mdx             Create an MDX file.
  --dry-run         Print the target path without creating the file.
  -h, --help        Show this help message.

Examples:
  bash ./scripts/init-blog-post.sh "my new post"
  bash ./scripts/init-blog-post.sh "my new post" --mdx
  bash ./scripts/init-blog-post.sh --title "my new post" --slug custom-name
  bash ./scripts/init-blog-post.sh "my new post" --dry-run
EOF
}

slugify() {
  local value="$1"
  value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
  value="$(printf '%s' "$value" | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-{2,}/-/g')"
  printf '%s' "$value"
}

title_from_slug() {
  local value="$1"
  printf '%s' "$value" \
    | tr '-' '\n' \
    | sed '/^$/d' \
    | awk '{ printf("%s%s", toupper(substr($0,1,1)) substr($0,2), ORS) }' \
    | paste -sd ' ' -
}

while (($# > 0)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --mdx)
      extension="mdx"
      shift
      ;;
    --md)
      extension="md"
      shift
      ;;
    --dry-run)
      dry_run="true"
      shift
      ;;
    --title)
      title="${2-}"
      shift 2
      ;;
    --slug)
      slug="${2-}"
      shift 2
      ;;
    --*)
      usage >&2
      printf 'Unknown option: %s\n' "$1" >&2
      exit 1
      ;;
    *)
      if [[ -z "${title}" ]]; then
        title="$1"
      else
        title="${title} $1"
      fi
      shift
      ;;
  esac
done

year="$(date +%Y)"
month="$(date +%m)"
day="$(date +%d)"
iso_date="${year}-${month}-${day}"

if [[ -n "${slug}" ]]; then
  slug="$(slugify "${slug}")"
else
  slug="$(slugify "${title:-new-post}")"
fi

if [[ -z "${slug}" ]]; then
  printf 'Could not derive a filename slug. Provide a title or use --slug.\n' >&2
  exit 1
fi

if [[ -z "${title}" ]]; then
  title="$(title_from_slug "${slug}")"
fi

year_dir="${POSTS_ROOT}/${year}"
target_path="${year_dir}/${month}-${day}-${slug}.${extension}"

if [[ "${dry_run}" == "true" ]]; then
  printf '%s\n' "${target_path}"
  exit 0
fi

mkdir -p "${year_dir}"

if [[ -e "${target_path}" ]]; then
  printf 'Post file already exists. Use a different slug or date.\n' >&2
  exit 1
fi

cat > "${target_path}" <<EOF
---
title: "${title}"
date: ${iso_date}
author: BZ
description: ""
categories: []
tags: []
---

<!-- more -->

## Draft

EOF

printf 'Created %s\n' "${target_path}"
