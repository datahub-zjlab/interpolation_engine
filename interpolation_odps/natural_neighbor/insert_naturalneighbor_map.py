import json
import numpy as np
from scipy.spatial import Delaunay, ConvexHull, QhullError
from metpy.interpolate import geometry
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from utils.odps_util import Odps

o = Odps().get_odps_instance()

def get_origin_data(datasetname, gl, aggr_func, name):
    sql = f'select index, lat, lon, value from meanorigindata where name="{name}" and datasetName="{datasetname}" and grid_length={gl} and aggregate_func="{aggr_func}";'
    with o.execute_sql(sql).open_reader(
            tunnel=True) as reader:
        pd_df = reader.to_pandas()
    return pd_df

def get_insert_data(datasetname, gl):
    sql = f'select box_index, box_lat, box_lon from insertdata where datasetName="{datasetname}" and grid_length={gl};'
    with o.execute_sql(sql).open_reader(
            tunnel=True) as reader:
        pd_df = reader.to_pandas()
    return pd_df

import math
def can_form_triangle(p1, p2, p3):
    """
    判断三个点是否能构成一个三角形。

    参数:
        p1, p2, p3: 分别是三个点的坐标元组 (x, y)

    返回:
        True 如果三个点可以构成三角形，否则返回 False
    """
    if (p1[0] == p2[0] and p1[1] == p2[1]) or (p2[0] == p3[0] and p2[1] == p3[1]) or (p1[0] == p3[0] and p1[1] == p3[1]):
        return False
    # 计算三边的长度
    a = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    b = math.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
    c = math.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
    # print(a,b,c)
    # 检查是否满足三角形的条件
    return (a + b > c) and (a + c > b) and (b + c > a)


def insert_into_origindatamapnaturalneighbor(datasetname, gl, aggr_func, feature, insert_to_odps = False):
    df = get_origin_data(datasetname, gl, aggr_func, feature)
    points = np.array(df[['lon', 'lat']])
    values = np.array(df['value'])


    insert_data = get_insert_data(datasetname, gl)
    points_tobe_interpolated = insert_data[['box_lon', 'box_lat']]
    id_relation_of_insert_data = {i: insert_data.loc[i, 'box_index'] for i in range(len(insert_data.box_index))}

    starttime = time.time()

    # 构建三角网
    tri = Delaunay(points[:, [0, 1]])
    print("tri.simplices length before process:  ", len(tri.simplices))
    new_simplices = list()
    for i, indices in enumerate(tri.simplices):
        triangle = tri.points[indices]
        if can_form_triangle(*triangle):
            new_simplices.append(indices)
    tri.simplices = np.array(new_simplices)
    print("tri.simplices length after process:  ", len(tri.simplices))

    # 给插值点寻找近邻点，获取三角网的三角形的外接圆圆心列表
    members, circumcenters = geometry.find_natural_neighbors(tri, points_tobe_interpolated)
    print(f"{len(members), len(circumcenters)}")

    map_ = []
    j = 0

    for grid, neighbors in members.items():
        box_index = id_relation_of_insert_data[grid]

        if len(neighbors) > 0:
            j += 1
            neighbors_tri_simplices = dict()
            neighbors_tri_points = dict()
            neighbors_point_value = dict()
            neighbors_circumcenters = dict()
            for nb in neighbors:
                neighbors_tri_simplices[nb] = tri.simplices[nb].tolist()
                neighbors_circumcenters[nb] = circumcenters[nb].tolist()
                for vertice in tri.simplices[nb]:
                    neighbors_tri_points[int(vertice)] = tri.points[vertice].tolist()
                    neighbors_point_value[int(vertice)] = values[vertice].tolist()

            map_.append([box_index,
                         json.dumps(neighbors),
                         json.dumps(neighbors_tri_simplices),
                         json.dumps(neighbors_tri_points),
                         json.dumps(neighbors_point_value),
                         json.dumps(neighbors_circumcenters),
                         datasetname,
                         gl,
                         feature])
        # else:
        #     map_.append([box_index,
        #                  None,
        #                  None,
        #                  None,
        #                  None,
        #                  None,
        #                  datasetname,
        #                  gl,
        #                  feature])

        if grid % 1000 == 0:
            print("grid>>>", grid)

    # print(map_)
    if insert_to_odps:
        o.write_table('naturalneighbor_map', map_)
    print("cal time:", time.time() - starttime)


if __name__ == "__main__":
    datasetname = '纽芬兰'
    gl = 1
    aggr_func ='mean'
    name = 'SiO2'
    insert_into_origindatamapnaturalneighbor(datasetname, gl, aggr_func, name, insert_to_odps=True)