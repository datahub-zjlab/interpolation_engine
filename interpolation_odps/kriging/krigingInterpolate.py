# import math
from odps.udf import annotate
import sys
sys.path.insert(0, "work/PyKrige-1.7.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.zip")
sys.path.insert(1, "work/numpy-1.21.6.zip")
sys.path.insert(2, "work/scipy-1.7.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.zip")
import numpy as np
from pykrige.ok import OrdinaryKriging


# @annotate("Double, Double, Array<Array<Double>>, Bigint->Array<Double>")
@annotate("Double, Double, Array<Array<Double>>, Bigint, String, Array<Double>->Double")
class krigingInterpolate(object):
    # def __init__(self):

    def evaluate(self, cell_x, cell_y, neigh, thresh=10, model='gaussian', param=None):
        df = np.array([x for x in neigh if x[0] is not None])
        if df.shape[0] < thresh:
            return None
            # return [df.shape[0], float(np.mean(df, axis=0)[3])]
            # return [df.shape[0], float(df[0][4])]
        try:
            ok3d = OrdinaryKriging(df[:thresh, 1], df[:thresh, 2], df[:thresh, 3], variogram_model=model)
            # param = [1.08944327e-22, 4.48928370e+01]  # Zn @1
            # param = [2.47987308e-16, 2.17823897e+04] # Zn @2
            if param is not None:
                ok3d.update_variogram_model(variogram_model=model, variogram_parameters=param)
            k3d1, ss3d = ok3d.execute("grid", [cell_x], [cell_y])
        except ValueError:
            # print('errrrr')
            return float(np.mean(df, axis=0)[3])
            # raise ValueError(str(df))
        d = k3d1.data[0]

        return float(d[0])

if __name__ == '__main__':
    tt = krigingInterpolate()
    # grid box index: 11206
    nei = [[3.980404248447033, 46.90233358, -55.46402621, 26.8], [4.017836837497612, 46.90278052, -55.46514262, 19.8], [4.021106305674149, 46.90267765, -55.46619595, 27.3], [4.030942452581096, 46.90273954, -55.46678504, 37.1], [4.036726061546153, 46.90328078, -55.46352647, 13.0], [4.041477056960967, 46.90329665, -55.46403809, 22.1], [4.048330720762647, 46.90342083, -55.46390324, 23.2], [4.052548991310476, 46.90349226, -55.46386181, 20.5], [4.0556901289455505, 46.90352966, -55.46396578, 23.5], [4.0575157191017235, 46.9035891, -55.46370148, 8.1], [4.071415661494965, 46.90372455, -55.46440664, 43.6], [4.135925442813118, 46.90455353, -55.46582731, 15.8], [4.147553916415292, 46.90471738, -55.46596707, 23.6], [4.149971267935704, 46.90472055, -55.46620333, 39.5], [4.163427850237428, 46.90486182, -55.46667199, 16.8], [4.1917847794601935, 46.90532007, -55.46661953, 36.6], [4.192064608325633, 46.90532042, -55.46664578, 49.1], [4.195590872241614, 46.90537457, -55.46665737, 48.4], [4.198128444273725, 46.9054109, -55.46668259, 39.4], [4.198691903472494, 46.90541161, -55.4667351, 51.5]]
    res = tt.evaluate(46.84, -55.45, nei, model='gaussian', thresh=16)
    print(res)