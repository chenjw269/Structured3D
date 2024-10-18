import numpy as np

def convert_lines_to_vertices(lines):
    """convert line representation to polygon vertices
    """
    polygons = []
    lines = np.array(lines)

    polygon = None
    while len(lines) != 0:
        # 多边形中的第一条边，直接取线集的第一条线
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, 0)

        # 从线集中，查找
        lineID, juncID = np.where(lines == polygon[-1])
        vertex = lines[lineID[0], 1 - juncID[0]]
        lines = np.delete(lines, lineID, 0)

        if vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)

    return polygons