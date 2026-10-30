from collections import defaultdict

def fertility(rows, encode, segment):
    totals = defaultdict(lambda: [0, 0, 0])
    for language, text in rows:
        words = segment(language, text)
        if not words:
            continue
        ids = encode(text)
        counts = totals[language]
        counts[0] += len(ids)
        counts[1] += len(words)
        counts[2] += 1
    return {lang: {"tokens_per_word": t / w, "documents": n}
            for lang, (t, w, n) in totals.items()}

result = fertility([("en", "ab cd"), ("en", ""), ("hi", "एक दो")],
                   lambda text: list(text.replace(" ", "")),
                   lambda lang, text: text.split())
assert result["en"] == {"tokens_per_word": 2.0, "documents": 1}
assert result["hi"]["tokens_per_word"] == 2.0
assert fertility([], list, lambda lang, text: text.split()) == {}
