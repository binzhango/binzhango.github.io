TOKENS = ("<|repo_name|>", "<|file_sep|>", "<|fim_prefix|>",
          "<|fim_suffix|>", "<|fim_middle|>", "<|endoftext|>")

def repo_fim(repo, files, target, start, end):
    if target not in files or not 0 <= start <= end <= len(files[target]):
        raise ValueError("invalid target span")
    values = [repo, *files.keys(), *files.values()]
    if any(t in value for t in TOKENS for value in values):
        raise ValueError("reserved marker in source")
    context = f"<|repo_name|>{repo}\n"
    for path in sorted(files):
        if path != target:
            context += f"<|file_sep|>{path}\n{files[path]}\n"
    text = files[target]
    prompt = (context + f"<|file_sep|>{target}\n<|fim_prefix|>"
              + text[:start] + "<|fim_suffix|>" + text[end:]
              + "<|fim_middle|>")
    return prompt, text[start:end] + "<|endoftext|>"

prompt, answer = repo_fim("demo", {"a.py": "x=12\n"}, "a.py", 2, 4)
assert "12" not in prompt and answer == "12<|endoftext|>"
assert prompt.endswith("<|fim_middle|>")
assert repo_fim("r", {"x": ""}, "x", 0, 0)[1] == "<|endoftext|>"
