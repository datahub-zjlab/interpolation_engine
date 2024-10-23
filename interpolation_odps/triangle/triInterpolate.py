# import math
from odps.udf import annotate

# @annotate("Double, Double, Array<Array<Double>>, Bigint->Array<Double>")
@annotate("Array<Double>, Double, Double ->Double")
class triInterpolate(object):
    # def __init__(self):
    def evaluate(self, tri_point_list, insert_point_lat, insert_point_lon):
        p1_lat, p1_lon, p1_val, p2_lat, p2_lon, p2_val, p3_lat, p3_lon, p3_val = tri_point_list
        s1 = self.cal_area(p1_lat, p1_lon, p2_lat, p2_lon, insert_point_lat, insert_point_lon)
        s2 = self.cal_area(p1_lat, p1_lon, p3_lat, p3_lon, insert_point_lat, insert_point_lon)
        s3 = self.cal_area(p2_lat, p2_lon, p3_lat, p3_lon, insert_point_lat, insert_point_lon)
        s = s1 + s2 + s3
        r1 = s1/s
        r2 = s2/s
        r3 = s3/s
        return 0.5 * (r1 * (p1_val + p2_val) + r2 * (p1_val + p2_val) + r3 * (p2_val + p3_val))
    def cal_area(self, x1,y1, x2,y2,x3,y3):
        return 0.5 * abs(x1 * (y2-y3) + x2 * (y3-y1) + x3 * (y1-y2))



if __name__ == '__main__':
    pass