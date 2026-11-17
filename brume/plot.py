import numpy as np

SHADES = " .:-=+*#%@"


def density(points, width=48, height=20, lim=2.6):
    """Render a 2D point cloud as an ascii density map (list of strings)."""
    h, _, _ = np.histogram2d(
        points[:, 1], points[:, 0], bins=[height, width], range=[[-lim, lim], [-lim, lim]]
    )
    h = h[::-1]  # rows go top to bottom, y goes up
    if h.max() > 0:
        # sqrt so the sparse tails still show up instead of everything
        # being either blank or '@'
        h = np.sqrt(h / h.max())
    idx = np.minimum((h * (len(SHADES) - 1)).round().astype(int), len(SHADES) - 1)
    return ["".join(SHADES[i] for i in row) for row in idx]


def side_by_side(panels, titles, gap="  |  "):
    width = max(len(r) for p in panels for r in p)
    head = gap.join(t.center(width) for t in titles)
    body = [gap.join(row.ljust(width) for row in rows) for rows in zip(*panels)]
    return "\n".join([head] + body)
