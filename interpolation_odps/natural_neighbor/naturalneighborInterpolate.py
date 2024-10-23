from odps.udf import annotate
import sys
import json
sys.path.insert(0, "work/numpy-1.21.6.zip")
sys.path.insert(1, "work/scipy-1.7.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.zip")
from scipy.spatial import ConvexHull

@annotate("String, String, String, String, String, Double, Double ->Double")
class naturalneighborInterpolate(object):
    # def __init__(self):
    def evaluate(self, neighbors, neighbors_tri_simplices, neighbors_tri_points, neighbors_point_value, neighbors_circumcenters, insert_point_lat, insert_point_lon):
        if neighbors is None:
            return None

        neighbors = json.loads(neighbors)
        tri_simplices = self.convert_dict(json.loads(neighbors_tri_simplices))
        tri_points = self.convert_dict(json.loads(neighbors_tri_points))
        points_value = self.convert_dict(json.loads(neighbors_point_value))
        circumcenters = self.convert_dict(json.loads(neighbors_circumcenters))

        edges = self.find_local_boundary(tri_simplices, neighbors)
        edge_vertices = [segment[0] for segment in self.order_edges(edges)]
        num_vertices = len(edge_vertices)

        p1 = edge_vertices[0]
        p2 = edge_vertices[1]

        try:
            c1 = self.circumcenter(insert_point_lon, insert_point_lat, tri_points[p1], tri_points[p2])
        except ZeroDivisionError as e:
            return None
        polygon = [c1]


        area_list = []
        total_area = 0.0
        for i in range(num_vertices):

            p3 = edge_vertices[(i + 2) % num_vertices]

            try:

                c2 = self.circumcenter(insert_point_lon, insert_point_lat, tri_points[p3], tri_points[p2])
                polygon.append(c2)

                for check_tri in neighbors:
                    if p2 in tri_simplices[check_tri]:
                        polygon.append(tuple(circumcenters[check_tri]))

                pts = [polygon[i] for i in ConvexHull(polygon).vertices] #耗时1ms左右

                value = points_value[p2]

                cur_area = self.area(pts)
                total_area += cur_area
                area_list.append(cur_area * value)

            except Exception as e:
                return None

            polygon = [c2]
            p2 = p3


        return sum(x / total_area for x in area_list) if total_area != 0.0 else None


    def convert_dict(self, original_dict):
        return {int(key): value for key, value in original_dict.items()}

    def find_local_boundary(self, tri_simplices, triangles):
        edges = []

        for triangle in triangles:

            for i in range(3):

                pt1 = tri_simplices[triangle][i]
                pt2 = tri_simplices[triangle][(i + 1) % 3]

                if (pt1, pt2) in edges:
                    edges.remove((pt1, pt2))

                elif (pt2, pt1) in edges:
                    edges.remove((pt2, pt1))

                else:
                    edges.append((pt1, pt2))

        return edges

    def order_edges(self, edges):
        edge = edges[0]
        edges = edges[1:]

        ordered_edges = [edge]

        num_max = len(edges)
        while len(edges) > 0 and num_max > 0:

            match = edge[1]

            for search_edge in edges:
                vertex = search_edge[0]
                if match == vertex:
                    edge = search_edge
                    edges.remove(edge)
                    ordered_edges.append(search_edge)
                    break
            num_max -= 1

        return ordered_edges

    def circumcenter(self, a_x, a_y, pt1, pt2):

        b_x = pt1[0]
        b_y = pt1[1]
        c_x = pt2[0]
        c_y = pt2[1]

        bc_y_diff = b_y - c_y
        ca_y_diff = c_y - a_y
        ab_y_diff = a_y - b_y
        cb_x_diff = c_x - b_x
        ac_x_diff = a_x - c_x
        ba_x_diff = b_x - a_x

        d_div = (a_x * bc_y_diff + b_x * ca_y_diff + c_x * ab_y_diff)

        if d_div == 0:
            raise ZeroDivisionError

        d_inv = 0.5 / d_div

        a_mag = a_x**2 + a_y**2
        b_mag = b_x**2 + b_y**2
        c_mag = c_x**2 + c_y**2

        cx = (a_mag * bc_y_diff + b_mag * ca_y_diff + c_mag * ab_y_diff) * d_inv
        cy = (a_mag * cb_x_diff + b_mag * ac_x_diff + c_mag * ba_x_diff) * d_inv

        return cx, cy

    def area(self, poly):
        a = 0.0
        n = len(poly)

        for i in range(n):
            a += poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]

        return abs(a) / 2.0


if __name__ == '__main__':
    pass