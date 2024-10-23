###插入三角形以及map关系
import time
import numpy as np
from scipy.spatial import Delaunay
from matplotlib.tri import Triangulation, LinearTriInterpolator, TrapezoidMapTriFinder
import matplotlib.pyplot as plt
from utils.odps_util import Odps

o = Odps().get_odps_instance()

#o.write_table('Triangle', [[1, 54808, 54810, 54812]])

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

def get_start_triindex():
    sql = f'select max(tri_index) from Triangle;'
    with o.execute_sql(sql).open_reader(
            tunnel=True) as reader:
        pd_df = reader.to_pandas()
        pd_df.columns = ['maxtriindex']
    print(pd_df)
    return pd_df.loc[0, 'maxtriindex'] + 1 if pd_df.loc[0, 'maxtriindex']!=None else 0

def insert_into_triangle(datasetname, gl, aggr_func, feature, insert_to_odps = False):
    df = get_origin_data(datasetname, gl, aggr_func, feature)
    id_relation = {i : df.loc[i, 'index'] for i in range(len(df.index))}
    points = np.array(df[['lon', 'lat', 'value']])
    tri = Delaunay(points[:, [0,1]])
    start_index = get_start_triindex()
    print('----------------')
    print(start_index)
    # tri.simplices 包含 Delaunay 三角剖分中的三角形列表(在此 2D 情况下)。每个三角形表示为三个整数:每个值表示原始点数组中的一个索引。 比如
    # points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
    # tri.simplices = array([[3, 2, 0], [3, 1, 0]], dtype=int32)
    # 所以 [3,2,0] 是顶点 3 (1,1)、顶点 2 (1,0) 和顶点 0 (0,0) 之间的三角形。
    simplices = tri.simplices
    triangle_r = [[start_index + i, id_relation[simplices[i][0]], id_relation[simplices[i][1]], id_relation[simplices[i][2]], datasetname, feature] for i in range(len(simplices))]
    #print(points)
    #print(points[:, 0])
    triang = Triangulation(points[:, 0], points[:, 1], tri.simplices)
    tri_find = TrapezoidMapTriFinder(triang)
    insert_data = get_insert_data(datasetname, gl)
    gridx = list(set(insert_data['box_lon']))
    gridy = list(set(insert_data['box_lat']))
    gridx.sort()
    gridy.sort()
    len_x = len(gridx)
    gridx = np.array([[i for i in gridx] for _ in gridy])
    gridy = np.array([[i] * len_x for i in gridy])
    z = tri_find(gridx, gridy)
    print(z.shape)
    row, col = z.shape
    triMap_r = [[start_index + z[i][j], i * col + j, datasetname, gl, feature] for i in range(row) for j in range(col) if z[i][j]!=-1] # 加start_index
    if insert_to_odps:
        o.write_table('Triangle', triangle_r)
        o.write_table('OrigindataMapTriangle', triMap_r)
    #print(triangle_r)

def tri_plot(datasetname, gl, aggr_func, feature):
    df = get_origin_data(datasetname, gl, aggr_func, feature)
    points = np.array(df[['lon', 'lat', 'value']])
    tri = Delaunay(points[:, [0, 1]])
    triang = Triangulation(points[:, 0], points[:, 1], tri.simplices)
    interpolator = LinearTriInterpolator(triang, np.array(points[:, 2]))
    insert_data = get_insert_data(datasetname, gl)
    gridx = list(set(insert_data['box_lon']))
    gridy = list(set(insert_data['box_lat']))
    gridx.sort()
    gridy.sort()
    len_x = len(gridx)
    gridx = np.array([[i for i in gridx] for _ in gridy])
    gridy = np.array([[i] * len_x for i in gridy])
    gridz = interpolator(gridx, gridy)
    print(gridz.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(gridx, gridy, gridz, levels=20, cmap='viridis')
    plt.colorbar(label='Interpolated values')
    plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], edgecolor='k', cmap='viridis')
    plt.triplot(points[:, 0], points[:, 1], tri.simplices, color='black', lw=0.5)
    plt.title('Linear Interpolation using Triangulated Irregular Network')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()




if __name__ == '__main__':
    datasetname = '纽芬兰'
    gl = 1
    aggr_func ='mean'
    # name = 'SiO2'
    # insert_into_triangle(datasetname, gl, aggr_func, name, insert_to_odps=True)
    #tri_plot(datasetname, gl, aggr_func, name)


    # all_feats = ['SiO2', 'Au']
    all_feats = ['Na2O', 'U', 'Zn', 'Mn', 'Mo', '87Rb/86Sr(m)', 'P', 'CaO', 'Eu', 'Hf', '87Sr/86Sr(m)',
                 'TDM2 (Ma)', 'Te', 'MnO', 'H2O-', 'Nd', 'Br', 'FeOT', 'Cd', 'Er', 'Al2O3', '207Pb/204Pb(m)',
                 'CL', 'Sr(m)', '147Sm/144Nd(m)', 'As', 'La', 'Hg', 'Gd', 'Cu', 'Pt', 'Rb', 'εNd(t)', 'FeO',
                 '208Pb/204Pb(m)', 'Bi', 'Fe2O3', 'Pd', '143Nd/144Nd(m)', 'Th', 'Pr', 'Sr', 'Zr', 'P2O5', 'MgO',
                 'Rb(m)', 'F', 'Ba', 'Ce', 'Cr', 'Recalcul TDM2-Nd (Ma)', 'Ir', 'S', 'CO2', 'Sb', 'Ho',
                 'Recalcul εNd(t)', 'f(Sm/Nd)', 'TDM1 (Ma)', 'Rb/Sr', '206Pb/204Pb(m)', 'Cs', 'Be', 'Recalcul εNd(0)',
                 'Ag', 'Ga', 'In', 'Y', 'Se', 'Ta', 'LOI', 'W', 'K2O', 'Fe2O3T', 'Pb', 'Tm', 'Nb', 'Nd(m)',
                 'Ti', 'Sm(m)', 'K2O/Na2O', 'Ge', 'δ18O', 'Co', 'Tb', 'Li', 'TiO2', '87Sr/86Sr(i)',
                 'log10(Rb/Sr)', 'Tl', 'Sm', 'Recalcul TDM1-Nd(Ma)', 'Sc', 'V', 'Lu', 'Dy', 'Sn', 'Ni']
    for name in all_feats:
        insert_into_triangle(datasetname, gl, aggr_func, name, insert_to_odps=True)
