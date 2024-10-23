import multiprocessing

from utils.odps_util import Odps

o = Odps().get_odps_instance()


n_process = multiprocessing.cpu_count()
method_name = 'krige'
_method_func_map = {'krige':'kriging', 'rbf':'rbf', 'nn':'nearest_neighbour', 'idw':'idw'}

def get_rec_data(method_id, rec):
    lat_down, lat_up, lon_left, lon_right = rec
    with o.execute_sql(f'select * from result where method_id={method_id} and box_lat between {lat_down} and {lat_up} and box_lon between {lon_left} and {lon_right};').open_reader(
            tunnel=True) as reader:
        pd_df = reader.to_pandas()
    return pd_df

def get_result_data_from_method_id(method_id):
    print(method_id)
    with o.execute_sql(f'select * from Result_{_method_func_map.get(method_name)} where method_id = {method_id}').open_reader(
            tunnel=True) as reader:
        data = reader.to_pandas()
    
    data.to_csv('SiO2-thresh-3.csv', index = False)
    print('done')
    # return data

def get_grid_data_by_feature(feature, aggr_func='mean'):
    print(feature)
    sql = f'select * from meanorigindata where name="{feature}" and aggregate_func="{aggr_func}" and grid_length=1' # only have res=1 
    # sql = f'select * from origindata where name="{feature}"'
    with o.execute_sql(sql).open_reader(
            tunnel=True) as reader:
        data = reader.to_pandas(n_process=n_process)
    
    data.to_csv(f'{feature}-data.csv', index = False)
    print('done')

def get_combined_data_by_feature_and_resolution_odps(feature, res, save_file=False, aggr_func='mean', dataset='纽芬兰'):
    print(feature, res)
    sql = f"""
with grid as (
  select box_x, box_y, box_index, box_lat, box_lon
  from insertdata
  where datasetname = '{dataset}' and grid_length = {res}),
orig as (
  select index, name, value
  from meanorigindata
  where datasetname = '{dataset}'
  and name='{feature}'
  and grid_length = {res}
)
select grid.box_index, grid.box_lat, grid.box_lon, orig.value
from grid left join orig
  on grid.box_index = orig.index;"""

    with o.execute_sql(sql).open_reader(
            tunnel=True) as reader:
        data = reader.to_pandas(n_process=n_process)
    print(data.shape)

    if save_file:
        fn = f'{feature}_{res}_data.csv'
        data.to_csv(fn, index = False)
        print(fn, '\ndone')
    else:
        print('done')
    return data


from scipy.spatial import KDTree
import numpy as np


class DM:

    def __call__(self,data):
        return getattr(self,self.type)(data)

    def median(self,data):
        return  np.median(data)

    def max(self,data):
        return np.max(data)

    def min(self,data):
        return np.min(data)

    def Q1(self,data):
        return np.percentile(data,25)

    def Q3(self,data):
        return np.percentile(data,75)

    def mean(self,data):
        return np.mean(data)

    def var(self,data):
        return np.var(data, ddof=1)

    def std(self,data):
        return np.std(data, ddof=1)

    def CV(self,data):
        mean = np.mean(data)

        # 计算标准差
        std_deviation = np.std(data)

        # 计算变异系数
        cv = (std_deviation / mean) * 100

        return  cv

    def local_comput(self,data):
        return getattr(self, self.type)(data)

    def __init__(self,x,y,v,type="median"):
        self.points = np.array([x,y])
        self.points = self.points.T
        self.v = np.array(v)
        self.tree = KDTree(self.points)
        self.type = type

    def compute(self,cell_x,cell_y,k = 5):
        points = np.array([cell_x,cell_y])
        if points.shape[0] >1:
            points = points.T

        _,values = self.tree.query(points,k=k)
        values = self.v[values]

        values = np.array(values)
        reval = []
        if len(values.shape) >1:
            for i in values:
                t = self.local_comput(i)
                reval.append(t)
        else:
            reval.append(self.local_comput(values))

        return reval


import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os

def get_df(filename):
    path = "./source_data/"+filename +".csv"
    df = pd.read_csv(path)
    return df
    

def funcs(features,func_names,lens = 1):

    for func_name in func_names:
        lat = []
        lon = []
        re_value = []
        re_method = []
        re_index = []
        for j in features:
            df = get_df(j)
            df_value = df[df['value'].notnull()]
            df_nan = df[df['value'].isnull()]
            func = DM(df_value['box_lat'].values,df_value['box_lon'].values, df_value['value'].values,type=func_name)
            try:
                values = func.compute(df_nan['box_lat'].values,df_nan['box_lon'].values)
            except:
                print(j,"error")
                continue

            #存储计算的插值
            lat.extend(list(df_nan['box_lat'].values))
            lon.extend(list(df_nan['box_lon'].values))
            re_index.extend(list(df_nan['box_index'].values))
            re_value.extend(values)
            re_method.extend([j]*len(values))

            #存储已存在的插值
            lat.extend(list(df_value['box_lat'].values))
            lon.extend(list(df_value['box_lon'].values))
            re_index.extend(list(df_value['box_index'].values))
            re_value.extend(list(df_value['value'].values))
            re_method.extend([j]*len(df_value))
            
            print(j,lens,"ok")

        output={"box_index":re_index,
                'method_id':re_method,
            "box_lat":lat,
            "box_lon":lon,
            "value":re_value}

        t = pd.DataFrame(output)
        store_dir = "./data/dm/dataset4/"
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        t.to_csv(f"{store_dir}{func_name}_{lens}.csv",index=False)
        print(f"{store_dir}{func_name}_{lens}.csv"," ok")



all_feats = ['Au', 'Na2O', 'U', 'Zn', 'Mn', 'Mo', '87Rb/86Sr(m)', 'P', 'CaO', 'Eu', 'Hf', '87Sr/86Sr(m)',
                 'TDM2 (Ma)', 'Te', 'MnO', 'H2O-', 'Nd', 'Br', 'FeOT', 'Cd', 'Er', 'Al2O3', '207Pb/204Pb(m)',
                 'CL', 'Sr(m)', '147Sm/144Nd(m)', 'As', 'La', 'Hg', 'Gd', 'Cu', 'Pt', 'Rb', 'εNd(t)', 'FeO',
                 '208Pb/204Pb(m)', 'Bi', 'Fe2O3', 'Pd', '143Nd/144Nd(m)', 'Th', 'Pr', 'Sr', 'Zr', 'P2O5', 'MgO',
                 'Rb(m)', 'F', 'Ba', 'Ce', 'Cr', 'Recalcul TDM2-Nd (Ma)', 'Ir', 'S', 'CO2', 'Sb', 'Ho',
                 'Recalcul εNd(t)', 'f(Sm/Nd)', 'TDM1 (Ma)', 'Rb/Sr', '206Pb/204Pb(m)', 'Cs', 'Be', 'Recalcul εNd(0)',
                 'Ag', 'Ga', 'In', 'Y', 'Se', 'Ta', 'LOI', 'W', 'K2O', 'Fe2O3T', 'Pb', 'Tm', 'Nb', 'Nd(m)',
                 'Ti', 'Sm(m)', 'K2O/Na2O', 'SiO2', 'Ge', 'δ18O', 'Co', 'Tb', 'Li', 'TiO2', '87Sr/86Sr(i)',
                 'log10(Rb/Sr)', 'Tl', 'Sm', 'Recalcul TDM1-Nd(Ma)', 'Sc', 'V', 'Lu', 'Dy', 'Sn', 'Ni']

func_name = ['Q1','Q3','max','mean','median','min','std','var']
all_feats = ["dataset4_Nd","dataset4_TDM"]

for lens in [10]:
    
    funcs(all_feats,func_name,lens)