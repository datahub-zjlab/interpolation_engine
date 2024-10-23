import math
from odps.udf import annotate
from odps.udf import BaseUDAF


@annotate("Double,Double,Double,Double,Double,Bigint->ARRAY<ARRAY<Double>>")
class FindNeighbours(BaseUDAF):
    def __init__(self):
        # self.size = 200  # grid: size*size
        self.thresh = 20
        self.alpha_dict = {0: 111.1949, 1: 111.178, 2: 111.1272, 3: 111.0425, 4: 110.9241, 5: 110.7718, 6: 110.5858,
                           7: 110.3661,
                           8: 110.1128, 9: 109.8259, 10: 109.5056, 11: 109.1519, 12: 108.765, 13: 108.3449,
                           14: 107.8919,
                           15: 107.406, 16: 106.8873, 17: 106.3361, 18: 105.7525, 19: 105.1367, 20: 104.4889,
                           21: 103.8092,
                           22: 103.098, 23: 102.3553, 24: 101.5814, 25: 100.7766, 26: 99.9411, 27: 99.0751, 28: 98.179,
                           29: 97.253, 30: 96.2973, 31: 95.3123, 32: 94.2983, 33: 93.2556, 34: 92.1844, 35: 91.0852,
                           36: 89.9582,
                           37: 88.8038, 38: 87.6224, 39: 86.4143, 40: 85.1798, 41: 83.9194, 42: 82.6335, 43: 81.3223,
                           44: 79.9864, 45: 78.6262, 46: 77.242, 47: 75.8342, 48: 74.4034, 49: 72.9499, 50: 71.4742,
                           51: 69.9767,
                           52: 68.4579, 53: 66.9182, 54: 65.3582, 55: 63.7782, 56: 62.1789, 57: 60.5606, 58: 58.9238,
                           59: 57.2691, 60: 55.5969, 61: 53.9078, 62: 52.2023, 63: 50.4809, 64: 48.7441, 65: 46.9925,
                           66: 45.2266, 67: 43.4469, 68: 41.6539, 69: 39.8483, 70: 38.0305, 71: 36.2011, 72: 34.3607,
                           73: 32.5099, 74: 30.6491, 75: 28.779, 76: 26.9002, 77: 25.0131, 78: 23.1184, 79: 21.2167,
                           80: 19.3086,
                           81: 17.3945, 82: 15.4752, 83: 13.5511, 84: 11.6229, 85: 9.6912, 86: 7.7565, 87: 5.8194,
                           88: 3.8806,
                           89: 1.9406, 90: 0.0}

    # TODO: thresh may change in iterate!
    def new_buffer(self):
        # [dist, lat, long, value]
        # return [[None]*4 for _ in range(self.thresh)]
        return list()

    def iterate(self, buffer, pnt_lon=0, pnt_lat=0, cell_lon=0, cell_lat=0, val=0, thresh=20):
        dist = self.distance2(pnt_lon, pnt_lat, cell_lon, cell_lat)
        self.thresh = thresh
        for i in range(len(buffer)):
            try:
                # store points in ascending order
                if dist < buffer[i][0]:
                    buffer.insert(i, [dist, pnt_lon, pnt_lat, val])
                    if len(buffer) > thresh:
                        buffer.pop()
                    return
                # if full or dist larger than any points, ignore
            except IndexError:
                raise IndexError(f'i:{i}, buffer:{buffer} thresh:{self.thresh}')
        if len(buffer) >= thresh:
            return
        buffer.append([dist, pnt_lon, pnt_lat, val])

    def merge(self, buffer, pbuffer):
        pb, pp = 0, 0
        while pb < len(buffer) and pp < len(pbuffer):
            try:
                if buffer[pb][0] > pbuffer[pp][0]:
                    buffer.insert(pb, pbuffer[pp])
                    pp += 1
                pb += 1
                if pb >= self.thresh:
                    # buffer.pop()
                    break
            except IndexError:
                raise IndexError(f'pb:{pb}, pp:{pp}, thresh:{self.thresh}')
        buffer += pbuffer[pp:]
        buffer = buffer[:self.thresh]
        # buffer += [x for x in pbuffer if x is not None]

    def terminate(self, buffer):
        return buffer
        # return [[len(buffer)]] + buffer
        # return [x for x in buffer if x[0] is not None]

    def distance2(self, lon1, lat1, lon2, lat2):
        eu_dis = math.sqrt(((lon2 - lon1) ** 2) + ((lat2 - lat1) ** 2))
        alpha = self.alpha_dict[abs(int(lat1))]
        return alpha * eu_dis
