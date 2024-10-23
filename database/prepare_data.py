import pandas as pd
import numpy as np
import math
from utils.odps_util import Odps

o = Odps().get_odps_instance()

class Dataset():
    def __init__(self, data, dataName, feature_list):
        self.data = data
        self.dataName = dataName
        self.feature_list = feature_list
    def div(self, x, y):
        if y != 0:
            return x/y
        if x == 0:
            return 0
        return np.nan
    def add_feature_div(self, feature_name, col_name1, col_name2):
        self.data[feature_name] = self.data.apply(lambda row:self.div(row[col_name1], row[col_name2]), axis = 1)
        self.feature_list.append(feature_name)
    def _log_(self, x):
        if x <= 0:
            return np.nan
        return math.log10(x)
    def add_feature_log(self, feature_name, col_name):
        self.data[feature_name] = self.data.apply(lambda row : self._log_(row[col_name]), axis = 1)
        self.feature_list.append(feature_name)

####需要进行插值的点的相关参数,后续用于点生成
class InsertBox():
    def __init__(self, lat_min, lat_max, lon_min, lon_max, grid_length = 1.0, datasetName = 'global'):
        if type == 'global':
            lat_gap, lon_gap = trans_gridlength_to_la(0, 0, grid_length=grid_length)
        else:
            lat_gap, lon_gap = trans_gridlength_to_la(lat_max, lat_min, grid_length=grid_length)
        self.grid_length = grid_length
        x_num = math.ceil((lat_max - lat_min) / lat_gap)
        y_num = math.ceil((lon_max - lon_min) / lon_gap)
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_gap = lat_gap
        self.lon_gap = lon_gap
        self.x_num = x_num
        self.y_num = y_num
        self.allNum = x_num * y_num
        self.datasetName = datasetName
    def trans_box_index_to_lat_lon(self, box_index):
        #######根据box_index转换成经纬度#########
        x = box_index//self.y_num
        y = box_index%self.y_num
        lat = self.lat_min + self.lat_gap/2 + x * self.lat_gap
        lon = self.lon_min + self.lon_gap/2 + y * self.lon_gap
        return lat, lon
    def trans_lat_lon_to_box_index(self, lat, lon):
        #######根据经纬度转换成box_index#########
        x = (lat - self.lat_min)//self.lat_gap
        y = (lon - self.lon_min)//self.lon_gap
        return x * self.y_num + y

###########将分辨率km转变成经纬度,用于计算每个网格的间隔############
def trans_gridlength_to_la(max_lat, min_lat, grid_length):
    lat_to_km = 111
    mid_lat = (max_lat + min_lat)/2
    lon_to_km = 111 * np.cos(math.pi * mid_lat/180)
    return grid_length/lat_to_km, grid_length/lon_to_km

def insert_map_data(dataset, insertbox, insert_to_odps = False):
    #x纬度
    #y经度
    data = dataset.data
    datasetName = dataset.dataName
    map_r = []
    for i in range(len(data.index)):
        indx = (data.loc[i, 'Latitude'] - insertbox.lat_min)//insertbox.lat_gap
        indy = (data.loc[i, 'Longitude'] - insertbox.lon_min)//insertbox.lon_gap
        box_index = int(indx * insertbox.y_num + indy)
        map_r.append([datasetName, float(insertbox.grid_length), data.loc[i, 'index'], int(indx), int(indy), box_index])
    if insert_to_odps:
        o.write_table('OrigindataMapInsert', map_r)
    return map_r

def insert_box_data(insertbox, insert_to_odps = False):
    s_x = insertbox.lat_min + insertbox.lat_gap / 2
    count = 0
    box_r = []
    for i in range(insertbox.x_num):
        s_y = insertbox.lon_min + insertbox.lon_gap / 2
        for j in range(insertbox.y_num):
            box_r.append([insertbox.datasetName, float(insertbox.grid_length), i, j, count, s_x, s_y, 0])
            s_y += insertbox.lon_gap
            count += 1
        s_x += insertbox.lat_gap
    if insert_to_odps and len(box_r)!=0:
        o.write_table('InsertData', box_r)
    return box_r

def insert_origin_data(dataset, insert_to_odps = False):
    data = dataset.data
    dataName = dataset.dataName
    feature_list =dataset.feature_list
    origin_r = []
    for f in feature_list:
        for i in range(len(data.index)):
            lat = data.loc[i, 'Latitude']
            lon = data.loc[i, 'Longitude']
            value = data.loc[i, f]
            if pd.isnull(lat) or pd.isnull(lon) or pd.isnull(value) or type(value) is str:
                continue
            origin_r.append([dataName, i, lat, lon, f, value])
    if insert_to_odps:
        o.write_table('OriginData', origin_r)
    return origin_r

def prepare_data(data, datasetName, feature_list):
    lat = list(data['Latitude'])
    lon = list(data['Longitude'])
    lat_lon = [str(lat[i]) + ',' + str(lon[i]) for i in range(len(lat))]
    data['lat_lon'] = lat_lon
    data.index = range(len(data.index))
    data['index'] = data.index
    dataset = Dataset(data, datasetName, feature_list)
    return dataset

def prepare_mean_data(map_r, origin_r, insertbox, grid_length = 1,insert_to_odps = False, aggr_func = 'mean'):
    #####每个网格将网格内所有点求平均作为该点的插值
    map_r = pd.DataFrame(map_r, columns=['datasetName', 'grid_length', 'index', 'box_indx', 'box_indy', 'box_index'])
    origin_r = pd.DataFrame(origin_r, columns=['datasetName', 'index', 'lat', 'lon', 'name', 'value'])
    map_r = map_r.loc[:, ['index', 'box_index']]
    combine_data = pd.merge(origin_r, map_r, on = 'index')
    if aggr_func == 'mean':
        mean_data = combine_data.loc[:, ['datasetName', 'name', 'box_index', 'value']].groupby(['datasetName', 'name', 'box_index']).mean()
    else:
        mean_data = combine_data.loc[:, ['datasetName', 'name', 'box_index', 'value']].groupby(
            ['datasetName', 'name', 'box_index']).median()
    ind = mean_data.index
    mean_data['datasetName'] = [i[0] for i in ind]
    mean_data['name'] = [i[1] for i in ind]
    mean_data['box_index'] = [i[2] for i in ind]
    mean_data.index = range(len(mean_data['name']))
    lat_lon = mean_data['box_index'].apply(insertbox.trans_box_index_to_lat_lon)
    mean_data['lat'] = [i[0] for i in lat_lon]
    mean_data['lon'] = [i[1] for i in lat_lon]
    mean_data_r = [[mean_data.loc[i, 'datasetName'], mean_data.loc[i, 'box_index'], mean_data.loc[i, 'lat'], mean_data.loc[i, 'lon'], mean_data.loc[i, 'name'], mean_data.loc[i, 'value'], grid_length, aggr_func] for i in range(len(mean_data.index))]
    if insert_to_odps:
        o.write_table('MeanOriginData', mean_data_r)
    mean_data = mean_data.loc[~mean_data['box_index'].duplicated(), :]
    mean_data.columns = ['value', 'datasetName', 'name', 'index', 'Latitude', 'Longitude']
    mean_data.index = range(len(mean_data['name']))
    mean_data = Dataset(mean_data, mean_data.loc[0, 'datasetName'], feature_list=None)
    return mean_data

if __name__ == '__main__':
    # 数据加载
    df = pd.read_excel("3. Hf data of East Tianshan.xlsx")
    data = df[['Sample', 'Point', 'X', 'Y', 'eHft']]
    data.columns = ['Sample', 'Point', 'Longitude', 'Latitude', 'eHft']
    data = data.drop_duplicates()
    dataset = prepare_data(data, datasetName = '东天山', feature_list = ['eHft'])

    # 均值网格
    grid_length = 1
    insertbox = InsertBox(min(dataset.data['Latitude']), max(dataset.data['Latitude']), min(dataset.data['Longitude']), max(dataset.data['Longitude']), 
                          grid_length = grid_length, datasetName = '东天山')
    print("总网格数：", insertbox.allNum)
    box_r= insert_box_data(insertbox)
    box_df = pd.DataFrame(box_r, columns=['datasetName', 'grid_length', 'x_index', 'y_index', 'index', 'latitude', 'longitude', 'placeholder'])
    box_df = box_df.drop(columns=['placeholder'])
    map_r = insert_map_data(dataset, insertbox)
    origin_r = insert_origin_data(dataset)
    mean_data = prepare_mean_data(map_r, origin_r, insertbox, grid_length = grid_length)
    mean_df = mean_data.data
