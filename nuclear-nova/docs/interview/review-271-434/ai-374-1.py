import numpy as np

def nms(boxes, scores, threshold=0.5):
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    s = np.asarray(scores, dtype=float)
    if s.shape != (len(b),) or not 0 <= threshold <= 1:
        raise ValueError("invalid scores or threshold")
    if not np.isfinite(b).all() or not np.isfinite(s).all():
        raise ValueError("non-finite inputs")
    if np.any(b[:, 2:] < b[:, :2]):
        raise ValueError("inverted box")
    areas = np.prod(b[:, 2:] - b[:, :2], axis=1)
    order = np.argsort(-s, kind='stable')
    keep = []
    while order.size:
        i = int(order[0]); keep.append(i)
        rest = order[1:]
        wh = np.maximum(0, np.minimum(b[i, 2:], b[rest, 2:])
                           - np.maximum(b[i, :2], b[rest, :2]))
        intersection = wh[:, 0] * wh[:, 1]
        union = areas[i] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(union),
                        where=union > 0)
        order = rest[iou <= threshold]
    return keep

boxes = [[0, 0, 2, 2], [0, 0, 2, 2], [3, 3, 4, 4]]
assert nms(boxes, [0.9, 0.8, 0.7]) == [0, 2]
assert nms([], []) == []
assert nms([[0, 0, 0, 0]], [1]) == [0]
assert nms(boxes, [1, 1, 1], 1) == [0, 1, 2]
