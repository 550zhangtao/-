"""Prototype of the coarse-to-fine target detector (V13 idea).

Stage 1: find the large dark printed board (the whole target).
Stage 2: inside that board, find the bright 10-ring disc at the board's known
relative position/size, then validate it with the V12 geometry proofs.

Run on the board; it reuses main.py's helpers (_validate, _l_at, ...).
"""
import image

# Geometry measured from the real target photo (612x782 px):
#   board width/height = 0.925
#   disc radius / board height = 0.092
#   disc centre from board top = 0.545 * board height
BOARD_ASPECT = 0.925
DISC_R_RATIO = 0.092
DISC_DY = 0.545
TARGET_DARK_L = 40
BOARD_MIN_FRAC = 0.012
BOARD_MAX_FRAC = 0.92
BOARD_MERGE_MARGIN = 3


def find_boards(img, roi=None, dark_l=TARGET_DARK_L):
    width = img.width()
    height = img.height()
    if roi is None:
        rx, ry, rw, rh = 0, 0, width, height
    else:
        rx, ry, rw, rh = roi
        rx = max(0, rx)
        ry = max(0, ry)
        rw = min(rw, width - rx)
        rh = min(rh, height - ry)
    if rw < 24 or rh < 24:
        return []
    try:
        blobs = img.find_blobs(
            [(0, dark_l, -40, 40, -40, 40)],
            roi=(rx, ry, rw, rh),
            pixels_threshold=120, area_threshold=120,
            merge=True, margin=BOARD_MERGE_MARGIN)
    except Exception:
        return []
    frame = float(width * height)
    out = []
    for b in blobs:
        w = b.w()
        h = b.h()
        if w < 24 or h < 24:
            continue
        area = w * h
        if area < BOARD_MIN_FRAC * frame or area > BOARD_MAX_FRAC * frame:
            continue
        aspect = float(w) / h
        if aspect < 0.50 or aspect > 1.60:
            continue
        if b.density() < 0.30:
            continue
        board_h = w / BOARD_ASPECT
        if board_h > h * 1.10:
            board_h = h * 1.10
        out.append((b, w, h, board_h))
    out.sort(key=lambda t: t[0].pixels(), reverse=True)
    return out


def disc_in_board(img, b, w, h, board_h):
    """Search the bright 10 disc at the board's known relative position."""
    import main
    cx = int(b.x() + w * 0.5)
    cy = int(b.y() + board_h * DISC_DY)
    r_exp = DISC_R_RATIO * board_h
    half = int(max(12, r_exp * 3.4))
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(img.width(), cx + half)
    y1 = min(img.height(), cy + half)
    if x1 - x0 < 14 or y1 - y0 < 14:
        return None
    try:
        hist = img.get_histogram(roi=(x0, y0, x1 - x0, y1 - y0))
        p50 = hist.get_percentile(0.50).value()
        p90 = hist.get_percentile(0.90).value()
    except Exception:
        return None
    lmin = int(p50 + 0.55 * (p90 - p50))
    if lmin < 18:
        lmin = 18
    if lmin > 80:
        lmin = 80
    try:
        blobs = img.find_blobs(
            [(lmin, 100, -40, 40, -40, 40)],
            roi=(x0, y0, x1 - x0, y1 - y0),
            pixels_threshold=12, area_threshold=12, merge=False)
    except Exception:
        return None
    best = None
    extra = []
    for blob in blobs:
        bw = blob.w()
        bh = blob.h()
        if bw < 4 or bh < 4:
            continue
        aspect = float(bw) / bh
        if aspect < 0.55 or aspect > 1.80:
            continue
        if blob.density() < 0.50 or blob.density() > 0.95:
            continue
        r = 0.5 * (bw if bw > bh else bh)
        if r < r_exp * 0.55 or r > r_exp * 1.70:
            continue
        dx = blob.cx() - cx
        dy = blob.cy() - cy
        if dx * dx + dy * dy > (0.55 * board_h) * (0.55 * board_h):
            continue
        res = main._validate(img, blob, r, blob.density(), blob.pixels(), True)
        if res is None:
            continue
        size = res[8]
        if not (r_exp * 1.1 <= size <= r_exp * 3.4):
            continue
        if best is None or res[0] > best[0]:
            best = res
        extra.append((blob.cx(), blob.cy(), size, res[4]))
    if best is None:
        return None
    halfb = best[8] // 2
    return (best[6], best[7],
            (best[6] - halfb, best[7] - halfb, best[8], best[8]),
            best[4] * 100.0), (cx, cy, r_exp, lmin, extra)


def detect_coarse(img, roi=None):
    """Return (disc_tuple, board_info) or (None, board_info)."""
    boards = find_boards(img, roi)
    for b, w, h, bh in boards[:4]:
        result = disc_in_board(img, b, w, h, bh)
        if result is not None:
            disc, info = result
            return disc, (b.x(), b.y(), w, h, bh, info)
    return None, None
