from scipy.spatial import ConvexHull
import numpy as np

def convex(data, region):
    data = data.loc[data['Tectonic Belt'] == region, :]
    data.index = range(len(data.index))
    points = [[data.loc[i, 'Latitude'], data.loc[i, 'Longitude']] for i in range(len(data.index))]
    points = np.array(points)
    hull = ConvexHull(points)
    print(type(hull))
    return hull

def is_point_inside_polygon(pt, poly):
    # 判断点是否在多边形内部
    x, y = pt
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    data = pd.read_csv('data.csv')
    data = data.loc[data['Tectonic Belt'] == 'Peri-Laurentia', :]
    data.index = range(len(data.index))
    hull = convex(data, 'Peri-Laurentia')
    vertices = hull.points[hull.vertices]


