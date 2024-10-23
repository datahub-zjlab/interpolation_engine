from pykrige.ok import OrdinaryKriging
import numpy as np
import time
import pandas as pd
# from memory_profiler import profile
import tracemalloc
import sys
import pickle
import math

def trans_gridlength_to_la(max_lat, min_lat, grid_length):
    lat_to_km = 111
    mid_lat = (max_lat + min_lat)/2
    lon_to_km = 111 * np.cos(math.pi * mid_lat/180)
    return grid_length/lat_to_km, grid_length/lon_to_km

def merge_arrays(arr1, arr2):
    merged = np.concatenate((arr1, arr2))
    return merged.reshape((2, len(arr1))).T


#data = pd.read_csv('data.csv')
data = pd.read_csv('IDW/niu_SiO2_mean.csv')

lon_sta = data['lon'].values
lat_sta = data['lat'].values
t2m_sta = data['value'].values
list_data=data[['lat','lon']].values.tolist()
y_data=data[['value']]
y_list_data=y_data.values.tolist()
latmax=data[['lat']].max()
latmin=data[['lat']].min()
lonmax=data[['lon']].max()
lonmin=data[['lon']].min()

lat_step=(latmax.values[0]-latmin.values[0])/475.0
lon_step=(lonmax.values[0]-lonmin.values[0])/491.0


t1 =  time.time()
print("begin create func")
ok = OrdinaryKriging(lon_sta, lat_sta, y_list_data)

print("begin create undefined data")
lat_res=np.arange(latmin.values[0]+lat_step,latmax.values[0]-lat_step,lat_step)
lon_res=np.arange(lonmin.values[0]+lon_step,lonmax.values[0]-lon_step,lon_step)
x,y = [],[]
for i in lat_res:
    for j in lon_res:
        x.append(i)
        y.append(j)
xy=merge_arrays(x,y)
print("bein create inserted data")
print("time=",time.time()-t1)
t2 = time.time()

k3d1, ss3d = ok.execute("grid", x, y, backend='loop', n_closest_points=5)
print(k3d1.data)

output={"box_lat":x,
        "box_lon":y,
        "value":k3d1.data}

print("done")
print("time2=",time.time()-t2)
print("total time = ",time.time()-t1)
csvout=pd.DataFrame(output)
csvout.to_csv('idw_niu_SiO2_mean-1.csv',index=False)

