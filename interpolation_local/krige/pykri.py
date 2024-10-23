from pykrige.ok import OrdinaryKriging
import numpy as np
import time
import pandas as pd
from memory_profiler import profile
import tracemalloc
import sys
import pickle
import math

def trans_gridlength_to_la(max_lat, min_lat, grid_length):
    lat_to_km = 111
    mid_lat = (max_lat + min_lat)/2
    lon_to_km = 111 * np.cos(math.pi * mid_lat/180)
    return grid_length/lat_to_km, grid_length/lon_to_km


data = pd.read_csv('niu_all_feature.csv')

f_list = list(set(list(data['name'])))
cols = ['lat', 'lon']
var_para = ['linear'] #, 'gaussian', 'hole-effect', 'power', 'spherical'
coo_para = ['euclidean']  #'geographic'
def cal_time(data, bounds, coor_type, vari, backend, grid_length = 1):
    #lat_min lat_max lon_min lon_max
    x_gap, y_gap = trans_gridlength_to_la(bounds[1], bounds[0], grid_length)
    gridx = np.arange(bounds[0], bounds[1] + x_gap, x_gap)
    gridy = np.arange(bounds[2], bounds[3] + y_gap, y_gap)
    start = time.time()
    ok3d = OrdinaryKriging(data['lat'], data['lon'], data['value'], variogram_model=vari, coordinates_type = coor_type) # 模型p
    k3d1, ss3d = ok3d.execute("grid", gridx, gridy, backend = backend, n_closest_points=5) # k3d1是结果，给出了每个网格点处对应的值
    end = time.time()
    g = end - start
    return k3d1, g, gridx, gridy

pd_ = pd.DataFrame(columns= ['特征', '已有点数', '插值点数', '距离计算', '变插值函数','backend', '计算时间', '内存占用'])
count = 0
tracemalloc.start()
saved = {}
z_all = []
gridx_all = []
gridy_all = []
var_all = []
feature_all = []
grid_length_all = []
bounds = [min(data['lat']), max(data['lat']), min(data['lon']), max(data['lon'])]
print(bounds)
for gl in [2]:
    for t in ['loop']:
        for var in var_para:
            for coo in coo_para:
                for f in f_list:
                    #cols2 = cols + [f]
                    cols2 = cols
                    d = data.loc[:, cols2 + ['name', 'value']]
                    d = d.loc[d['name'] == f,:]
                    print('#########################')
                    d = d.dropna()
                    d.index = range(len(d.index))
                    l = len(d.index)
                    if l < 10:
                        print(f'warning: feature {f} sample number less than 10')
                        continue
                    tracemalloc.clear_traces()
                    s1, p1 = tracemalloc.get_traced_memory()
                    p1 = (p1*1.0)/(10**6*1.0)

                    try:
                        z, g, gridx, gridy = cal_time(d, bounds, coo, var, t, grid_length=gl)
                        print(len(gridx) * len(gridy))
                        z_all.append(z)
                        gridx_all.append(gridx)
                        gridy_all.append(gridy)
                        var_all.append(var)
                        feature_all.append(f)
                        grid_length_all.append(gl)
                        print(z.shape)
                    except:
                        continue
                    # g = round(g, 2)
                    # s2, p2 = tracemalloc.get_traced_memory()
                    # p2 = (p2 * 1.0) / (10 ** 9 * 1.0)
                    # mem = p2 - p1
                    # num = 40000
                    # pd_.loc[count] = [f, l, num, coo, var, t, g, round(mem, 2)]
                    # count += 1
                    #pd_.to_csv('time.csv', index = False)

saved['data'] = z_all
saved['variogram'] = var_all
saved['feature'] = feature_all
saved['gridx'] = gridx_all
saved['gridy'] = gridy_all
saved['grid_length'] = grid_length_all

pickle.dump(saved, open('result_5_27_all_feature.pkl', 'wb'))

