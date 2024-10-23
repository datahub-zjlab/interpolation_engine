import numpy as np
import pandas as pd
import math
from scipy.spatial import KDTree
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time

# 加载数据
data = pd.read_csv('../data.csv') 

# 创建网格点
def trans_gridlength_to_la(max_lat, min_lat, grid_length):
    lat_to_km = 111
    mid_lat = (max_lat + min_lat)/2
    lon_to_km = 111 * np.cos(math.pi * mid_lat/180)
    return grid_length/lat_to_km, grid_length/lon_to_km

bounds = [min(data['Latitude']), max(data['Latitude']), min(data['Longitude']), max(data['Longitude'])]
x_gap, y_gap = trans_gridlength_to_la(bounds[1], bounds[0], grid_length=2) # 2km
gridx = np.arange(bounds[2], bounds[3] + y_gap, y_gap)  
gridy = np.arange(bounds[0], bounds[1] + x_gap, x_gap) 
grid_x, grid_y = np.meshgrid(gridx, gridy)

# 插值
f_list = ['SiO2']
for f in f_list:
    print(f)
    
    # 特征预处理
    d = data.loc[:, ['Longitude', 'Latitude'] + [f]]
    d = d.dropna()
    l = len(d.index)
    if l < 16:
        print(f'warning: feature {f} sample number less than 16')
        continue
    
    # 准备KDTree用于快速查找最近邻
    points = d[['Longitude', 'Latitude']].values
    tree = KDTree(points)

    # 插值
    start = time.time()
    
    flat_grid_x = grid_x.ravel()
    flat_grid_y = grid_y.ravel()
    interpolated_values = []
    
    for x, y in zip(flat_grid_x, flat_grid_y):
        # 找到最近的16个邻居
        distances, indices = tree.query([x, y], k=16)
        neighbors = d.iloc[indices]
        
        # 多项式回归
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(neighbors[['Longitude', 'Latitude']])
        model = LinearRegression().fit(X_poly, neighbors[f])
        
        # 预测该网格点的值
        grid_point = poly.transform([[x, y]])
        interpolated_value = model.predict(grid_point)[0]
        interpolated_values.append(interpolated_value)
    
    end = time.time()
    g = end - start
    print("运行时间：", g)
    
    # 组合成 DataFrame
    interpolated_df = pd.DataFrame({
        'box_lon': flat_grid_x,
        'box_lat': flat_grid_y,
        'value': interpolated_values
    })
    
    # 存储结果
    interpolated_df.to_csv('poly_' + f + '.csv', index=False)
