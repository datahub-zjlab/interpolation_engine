from odps.udf import annotate
import sys
sys.path.insert(1, "work/numpy-1.21.6.zip")
sys.path.insert(2, "work/scipy-1.7.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.zip")
import numpy as np
from scipy.spatial import KDTree

@annotate("Double, Double, Array<Array<Double>>, Bigint, String, Array<Double>->Double")
class NearestNeighbor(object):
    # def __init__(self):

    def evaluate(self, cell_x, cell_y, neighbors, k=1, model='gaussian', param=None):
        df = np.array([x for x in neighbors if x[0] is not None])
        if df.shape[0] == 0:
            return None
        
        # # KDTree查找最近邻
        # target_point = np.array([cell_x, cell_y])
        # tree = KDTree(df[:, 1:3])  
        # dists, indices = tree.query(target_point, k)
        # if k == 1:
        #     nearest_neighbors = df[indices].reshape(1, -1)
        # else:
        #     nearest_neighbors = df[indices]

        # 基于现成距离查找最近邻
        nearest_neighbors = df[0:k]

        average_value = np.mean(nearest_neighbors[:, 3])
        return float(average_value)

if __name__ == '__main__':
    nn = NearestNeighbor()
    neighbors = [[3.980404248447033, 46.90233358, -55.46402621, 26.8], [4.017836837497612, 46.90278052, -55.46514262, 19.8], [4.021106305674149, 46.90267765, -55.46619595, 27.3], [4.030942452581096, 46.90273954, -55.46678504, 37.1], [4.036726061546153, 46.90328078, -55.46352647, 13.0], [4.041477056960967, 46.90329665, -55.46403809, 22.1], [4.048330720762647, 46.90342083, -55.46390324, 23.2], [4.052548991310476, 46.90349226, -55.46386181, 20.5], [4.0556901289455505, 46.90352966, -55.46396578, 23.5], [4.0575157191017235, 46.9035891, -55.46370148, 8.1], [4.071415661494965, 46.90372455, -55.46440664, 43.6], [4.135925442813118, 46.90455353, -55.46582731, 15.8], [4.147553916415292, 46.90471738, -55.46596707, 23.6], [4.149971267935704, 46.90472055, -55.46620333, 39.5], [4.163427850237428, 46.90486182, -55.46667199, 16.8], [4.1917847794601935, 46.90532007, -55.46661953, 36.6], [4.192064608325633, 46.90532042, -55.46664578, 49.1], [4.195590872241614, 46.90537457, -55.46665737, 48.4], [4.198128444273725, 46.9054109, -55.46668259, 39.4], [4.198691903472494, 46.90541161, -55.4667351, 51.5]]
    res = nn.evaluate(46.84, -55.45, neighbors, k=1)
    print(res)