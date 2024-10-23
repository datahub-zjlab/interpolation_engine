"""
线性三角网插值本地实现，以SiO2为特征，使用有SiO2的点集构建插值点网格。
"""

from collections import defaultdict
import time
import numpy as np
import pandas as pd
from metpy import interpolate

import visualization

# 读取数据，以SiO2为例。
# 问题：纽芬兰数据中不同数据贡献者、同一经纬度、同一个特征会有不同特征值。
df = pd.read_excel('纽芬兰数据.xlsx', dtype={
    'Longitude': 'float64', 'Latitude': 'float64', 'SiO2': 'float64'})

data_has_value = df[df['SiO2'].notna()]

# --------------------------------------------------
# 自定义网格范围进行插值。这里只用了有SiO2的点。
x = data_has_value.loc[:, 'Longitude'].values
y = data_has_value.loc[:, 'Latitude'].values
z = data_has_value.loc[:, 'SiO2'].values

# 分辨率hres，和点集同单位，(max(x) - min(x))/hres
gx, gy, img = interpolate.interpolate_to_grid(
    x, y, z, interp_type='linear', hres=0.05)

# 导出插值结果到CSV文件，参考K2O_Na2O_region.csv。
df_linear = pd.DataFrame({'box_lon': gx.flatten(),
                          'box_lat': gy.flatten(), 'value': img.flatten()})
df_linear.to_csv('SiO2_linear.csv', index=False)

# 画图
# df_linear = pd.read_csv('SiO2_linear.csv')
visualization.map(df, 'SiO2', df_linear)

# --------------------------------------------------
# 测速：不同分辨率，每组十次。
times_linear = defaultdict(list)
for hres in range(20, 0, -1):
    hres = hres/40
    for _ in range(10):
        start_t = time.time()
        gx, gy, img = interpolate.interpolate_to_grid(
            x, y, z, interp_type='linear', hres=hres)
        end_t = time.time()
        grid_num = img.shape[0]*img.shape[1]
        print(
            f'linear interpolation takes {end_t-start_t:.2f}s, hres: {hres}, grid_num: {grid_num}')
        times_linear[(hres, grid_num)].append(end_t-start_t)

times = [(hres, gird_num, np.mean(t))
         for (hres, gird_num), t in times_linear.items()]
times = pd.DataFrame(times, columns=['hres', 'grid_num', 'time'])
print(times)
times.to_csv('times_linear.csv')
