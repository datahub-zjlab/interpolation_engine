import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from pandas.core.frame import DataFrame
import time
from scipy.interpolate import CloughTocher2DInterpolator



data=pd.read_csv('niu_SiO2_mean.csv')
lon_sta = data['lon'].values
lat_sta = data['lat'].values
t2m_sta = data['value'].values

latmax=data[['lat']].max()
latmin=data[['lat']].min()
lonmax=data[['lon']].max()
lonmin=data[['lon']].min()

lat_step=(latmax.values[0]-latmin.values[0])/475.0
lon_step=(lonmax.values[0]-lonmin.values[0])/491.0

t1 =  time.time()
print("begin create CMI func")
func = CloughTocher2DInterpolator( list(zip(lat_sta, lon_sta)), t2m_sta)

print("begin create undefined data")
lat_res=np.arange(latmin.values[0]+lat_step,latmax.values[0]-lat_step,lat_step)
lon_res=np.arange(lonmin.values[0]+lon_step,lonmax.values[0]-lon_step,lon_step)
x,y = [],[]
for i in lat_res:
    for j in lon_res:
        x.append(i)
        y.append(j)

print("bein create inserted data")
print("time=",time.time()-t1)
t2 = time.time()

t2m_RBF = func(x,y)
output={"box_lat":x,
        "box_lon":y,
        "value":t2m_RBF}

print("done")
print("time2=",time.time()-t2)
print("total time = ",time.time()-t1)
csvout=DataFrame(output)
csvout.to_csv('cmi_niu_SiO2_mean.csv',index=False)