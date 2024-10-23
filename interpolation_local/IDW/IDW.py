import numpy as np
import pandas as pd
#from scipy.interpolate import Rbf
from pandas.core.frame import DataFrame
import time
from photutils.utils import ShepardIDWInterpolator as idw

def merge_arrays(arr1, arr2):
    merged = np.concatenate((arr1, arr2))
    return merged.reshape((2, len(arr1))).T



data=pd.read_csv('niu_SiO2_mean.csv')
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

lat_step=(latmax.values[0]-latmin.values[0])/201.0
lon_step=(lonmax.values[0]-lonmin.values[0])/201.0

t1 =  time.time()
print("begin create func")
func = idw(list_data,y_list_data)

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

t2m_RBF = func(xy)
output={"box_lat":x,
        "box_lon":y,
        "value":t2m_RBF}

print("done")
print("time2=",time.time()-t2)
print("total time = ",time.time()-t1)
csvout=DataFrame(output)
csvout.to_csv('idw_niu_SiO2_mean.csv',index=False)