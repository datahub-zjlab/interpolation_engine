# Minimum Curvature interpolation
import time, os
from odps import options
import multiprocessing
import numpy as np
import pandas as pd
from scipy.interpolate import CloughTocher2DInterpolator
from interpolation_odps.yjy_interpolation import Method
from utils.odps_util import Odps

o = Odps().get_odps_instance()

env_settings = {
    'odps.sql.allow.cartesian': True,
    # 'in_predicate_conversion_threshold': 2,
    'odps.sql.python.version': 'cp37',
    'odps.sql.mapper.memory': 2048,
    # 'odps.sql.mapper.cpu': 100,
    'odps.sql.mapper.split.size': 1,
    'odps.sql.reducer.memory': 2048,
    # 'odps.sql.reducer.cpu': 100,
    'odps.sql.reducer.instances': 80,
    # 'options.tunnel.limit_instance_tunnel': True,
}
options.sql.settings = env_settings
n_process = multiprocessing.cpu_count()


class MinimumCurvatureInterpolation:
    def __init__(self, method):
        self.method = method
        # self.aggre_func = method.aggre_func
    def get_interpolation_sql(self):
        sql = f"""
            with partOrigin as (
              select index, lat, lon, value from meanorigindata where
              datasetname="{self.method.dataset}" and
              grid_length={self.method.resolution} and name="{self.method.feature}" and
              aggregate_func="{self.method.aggre_func}"
            ),
            m as (
              select tri_index, insert_point_index from OrigindataMapTriangle where datasetname="{self.method.dataset}" and grid_length={self.method.resolution} and name="{self.method.feature}"
            ),
            ins as (
              select box_index, box_lat, box_lon from insertdata where datasetname="{self.method.dataset}" and grid_length={self.method.resolution}
            )
            insert into result(box_index, box_lat, box_lon, value, method_id)
            select box_index, box_lat, box_lon, triInterpolate(array(point1_lat, point1_lon, point1_value, point2_lat, point2_lon, point2_value, point3_lat, point3_lon, point3_value), box_lat, box_lon), {self.method.m_id} from
            (
              select
                  t.tri_index,
                  t1.lat AS point1_lat, t1.lon AS point1_lon, t1.value AS point1_value,
                  t2.lat AS point2_lat, t2.lon AS point2_lon, t2.value AS point2_value,
                  t3.lat AS point3_lat, t3.lon AS point3_lon, t3.value AS point3_value
              FROM
                  (select tri_index, point1, point2, point3 from Triangle where datasetname="{self.method.dataset}" and name="{self.method.feature}") t
              JOIN
                  partOrigin t1 ON t.point1 = t1.index
              JOIN
                  partOrigin t2 ON t.point2 = t2.index
              JOIN
                  partOrigin t3 ON t.point3 = t3.index
            ) tri, ins join m on m.tri_index=tri.tri_index and ins.box_index=m.insert_point_index
            ;
            """
        return sql
    
    def run(self):
        sql = self.get_interpolation_sql()
        print(sql)
        start = time.time()
        o.execute_sql(sql)
        end = time.time()
        print(f'cal time: {end - start}')
        return end - start

if __name__ == '__main__':

    _method_name = "mincurvature"
    _save_path = "./results/"
    _dataset = "纽芬兰"
    _resolution = 1
    _aggre_func = "mean"

    all_feats = ["SiO2", 'Au', 'Na2O', 'U', 'Zn', 'Mn', 'Mo', '87Rb/86Sr(m)', 'P', 'CaO', 'Eu', 'Hf', '87Sr/86Sr(m)',
                 'TDM2 (Ma)', 'Te', 'MnO', 'H2O-', 'Nd', 'Br', 'FeOT', 'Cd', 'Er', 'Al2O3', '207Pb/204Pb(m)',
                 'CL', 'Sr(m)', '147Sm/144Nd(m)', 'As', 'La', 'Hg', 'Gd', 'Cu', 'Pt', 'Rb', 'εNd(t)', 'FeO',
                 '208Pb/204Pb(m)', 'Bi', 'Fe2O3', 'Pd', '143Nd/144Nd(m)', 'Th', 'Pr', 'Sr', 'Zr', 'P2O5', 'MgO',
                 'Rb(m)', 'F', 'Ba', 'Ce', 'Cr', 'Recalcul TDM2-Nd (Ma)', 'Ir', 'S', 'CO2', 'Sb', 'Ho',
                 'Recalcul εNd(t)', 'f(Sm/Nd)', 'TDM1 (Ma)', 'Rb/Sr', '206Pb/204Pb(m)', 'Cs', 'Be', 'Recalcul εNd(0)',
                 'Ag', 'Ga', 'In', 'Y', 'Se', 'Ta', 'LOI', 'W', 'K2O', 'Fe2O3T', 'Pb', 'Tm', 'Nb', 'Nd(m)',
                 'Ti', 'Sm(m)', 'K2O/Na2O', 'Ge', 'δ18O', 'Co', 'Tb', 'Li', 'TiO2', '87Sr/86Sr(i)',
                 'log10(Rb/Sr)', 'Tl', 'Sm', 'Recalcul TDM1-Nd(Ma)', 'Sc', 'V', 'Lu', 'Dy', 'Sn', 'Ni']
    all_feats = ['SiO2']

    time_records = pd.DataFrame(columns=['Method_id', 'Feature', 'Elapsed_time'])
    result_filename = f"{_method_name}_{len(all_feats)}-feats.csv"
    uniq_fn = 0
    while os.path.exists(os.path.join(_save_path, result_filename)):
        result_filename = f"{_method_name}_{len(all_feats)}-feats_{uniq_fn}.csv"
        uniq_fn += 1
    failed = []

    for feat in all_feats:
        method = Method(method_name=_method_name, params={}, thresh=0,
                        dataset=_dataset, resolution=_resolution, feature=feat, aggr_func=_aggre_func)
        print("------------------")
        method_id = method.insert_to_insertMethod_if_not_exists()
        a = MinimumCurvatureInterpolation(method)
        try:
            elapsed_time = a.run()
            time_records.loc[len(time_records)] = [method_id, feat, elapsed_time]
        except Exception as e:
            print(f"Error: {e}")
            failed.append(feat)

    time_records.to_csv(os.path.join(_save_path, result_filename))
    # print(time_records.head())
    print(f"All {len(all_feats)} features done. Saved to {result_filename}")
    if len(failed):
        print(f"Failed {len(failed)}/{len(all_feats)}:")
        print(failed)



def get_origin_data(datasetname, gl, aggr_func, name):
    sql = f'select index, lat, lon, value from meanorigindata where name="{name}" and datasetName="{datasetname}" and grid_length={gl} and aggregate_func="{aggr_func}";'
    with o.execute_sql(sql).open_reader(
            tunnel=True) as reader:
        pd_df = reader.to_pandas(n_process=n_process)
    return pd_df

def get_insert_data(datasetname, gl):
    sql = f'select box_index, box_lat, box_lon from insertdata where datasetName="{datasetname}" and grid_length={gl};'
    with o.execute_sql(sql).open_reader(
            tunnel=True) as reader:
        pd_df = reader.to_pandas(n_process=n_process)
    return pd_df

def get_start_triindex():
    sql = f'select max(tri_index) from Triangle;'
    with o.execute_sql(sql).open_reader(
            tunnel=True) as reader:
        pd_df = reader.to_pandas(n_process=n_process)
        pd_df.columns = ['maxtriindex']
    print(pd_df)
    return pd_df.loc[0, 'maxtriindex'] + 1 if pd_df.loc[0, 'maxtriindex']!=None else 0

def ct2d(datasetname, gl, aggr_func, feature, insert_to_odps = False):
    df = get_origin_data(datasetname, gl, aggr_func, feature)
    id_relation = {i : df.loc[i, 'index'] for i in range(len(df.index))}
    points = np.array(df[['lon', 'lat', 'value']])

    func = CloughTocher2DInterpolator( list(zip(lat_sta, lon_sta)), t2m_sta)

    start_index = get_start_triindex()
    print('----------------')
    print(start_index)
    # tri.simplices 包含 Delaunay 三角剖分中的三角形列表(在此 2D 情况下)。每个三角形表示为三个整数:每个值表示原始点数组中的一个索引。 比如
    # points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
    # tri.simplices = array([[3, 2, 0], [3, 1, 0]], dtype=int32)
    # 所以 [3,2,0] 是顶点 3 (1,1)、顶点 2 (1,0) 和顶点 0 (0,0) 之间的三角形。
    simplices = tri.simplices
    triangle_r = [[start_index + i, id_relation[simplices[i][0]], id_relation[simplices[i][1]], id_relation[simplices[i][2]], datasetname, feature] for i in range(len(simplices))]
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

    all_feats = ['Na2O', 'U', 'Zn', 'Mn', 'Mo', '87Rb/86Sr(m)', 'P', 'CaO', 'Eu', 'Hf', '87Sr/86Sr(m)',
                 'TDM2 (Ma)', 'Te', 'MnO', 'H2O-', 'Nd', 'Br', 'FeOT', 'Cd', 'Er', 'Al2O3', '207Pb/204Pb(m)',
                 'CL', 'Sr(m)', '147Sm/144Nd(m)', 'As', 'La', 'Hg', 'Gd', 'Cu', 'Pt', 'Rb', 'εNd(t)', 'FeO',
                 '208Pb/204Pb(m)', 'Bi', 'Fe2O3', 'Pd', '143Nd/144Nd(m)', 'Th', 'Pr', 'Sr', 'Zr', 'P2O5', 'MgO',
                 'Rb(m)', 'F', 'Ba', 'Ce', 'Cr', 'Recalcul TDM2-Nd (Ma)', 'Ir', 'S', 'CO2', 'Sb', 'Ho',
                 'Recalcul εNd(t)', 'f(Sm/Nd)', 'TDM1 (Ma)', 'Rb/Sr', '206Pb/204Pb(m)', 'Cs', 'Be', 'Recalcul εNd(0)',
                 'Ag', 'Ga', 'In', 'Y', 'Se', 'Ta', 'LOI', 'W', 'K2O', 'Fe2O3T', 'Pb', 'Tm', 'Nb', 'Nd(m)',
                 'Ti', 'Sm(m)', 'K2O/Na2O', 'Ge', 'δ18O', 'Co', 'Tb', 'Li', 'TiO2', '87Sr/86Sr(i)',
                 'log10(Rb/Sr)', 'Tl', 'Sm', 'Recalcul TDM1-Nd(Ma)', 'Sc', 'V', 'Lu', 'Dy', 'Sn', 'Ni']
    all_feats = ['SiO2']
    for name in all_feats:
        ct2d(datasetname, gl, aggr_func, name, insert_to_odps=True)
