from rec import *

def generate_region(x,y, region_dic):
    for r in region_dic.keys():
        hull = region_dic[r]
        vertices = hull.points[hull.vertices]
        if is_point_inside_polygon([x,y], vertices):
            return r
    return 'NULL'

def add_belt(data): ###data:from result table
    origindata = pd.read_csv('data.csv')
    belt = ['Avalonia', 'Ganderia', 'Laurentia', 'Peri-Laurentia']
    hull_dic = {}
    for b in belt:
        hull_dic[b] = convex(origindata, b)
    region = data.apply(lambda row:generate_region(row['box_lat'], row['box_lon'], hull_dic), axis = 1)
    data['region'] = region
    return data



if __name__ == '__main__':
    pass

