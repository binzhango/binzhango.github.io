def divide(a, b):
    if b == 0:
        raise ZeroDivisionError
    negative = (a < 0) != (b < 0)
    x, y = abs(a), abs(b)
    quotient = 0
    for shift in range(max(0, x.bit_length() - y.bit_length()), -1, -1):
        if (y << shift) <= x:
            x -= y << shift
            quotient |= 1 << shift
    return -quotient if negative else quotient

def components(n, edges):
    if n < 0:
        raise ValueError("negative vertex count")
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if not (0 <= a < n and 0 <= b < n):
            raise ValueError("invalid vertex")
        adj[a].append(b); adj[b].append(a)
    seen, count = set(), 0
    for start in range(n):
        if start in seen:
            continue
        count += 1
        stack = [start]; seen.add(start)
        while stack:
            for nxt in adj[stack.pop()]:
                if nxt not in seen:
                    seen.add(nxt); stack.append(nxt)
    return count

def balanced(text):
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for char in text:
        if char in '([{':
            stack.append(char)
        elif char in pairs and (not stack or stack.pop() != pairs[char]):
            return False
    return not stack

assert divide(-17, 3) == -5 and divide(0, 2) == 0
assert divide(2, 7) == 0 and divide(-8, -2) == 4
assert components(4, [(0, 1), (1, 1)]) == 3
assert components(0, []) == 0
assert balanced('a({[]})') and not balanced('([)]')
assert not balanced('(') and balanced('')
