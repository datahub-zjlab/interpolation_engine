#######插入insert表 origin表
import pandas as pd
import numpy as np
import math
from geopy import distance
from utils.odps_util import Odps

o = Odps().get_odps_instance()

import sys
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
    def region(self, name, type = 'belt'):
        #限制在一定区域
        ###区域：belt or domain
        map_dic = {'belt' : 'Tectonic Belt', 'domain':'Tectonic Domain'}
        self.data = self.data.loc[self.data[map_dic[type]] == name, :]
        self.data.index = range(len(self.data.index))
        self.dataName = name

####需要进行插值的点得的相关参数,后续用于点生成
class InsertBox():
    def __init__(self, lat_min, lat_max, lon_min, lon_max, grid_length = 1.0, datasetName = 'global'):
        if type == 'global':
            lat_gap, lon_gap = trans_gridlength_to_la(0, 0, grid_length=grid_length)
        else:
            lat_gap, lon_gap = trans_gridlength_to_la(lat_max, lat_min, grid_length=grid_length)
        self.grid_length = grid_length
        # x_num = int((lat_max + 0.01 - lat_min) / lat_gap)
        # y_num = int((lon_max + 0.01 - lon_min) / lon_gap)
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

##########计算最大分辨率##################
def cal_max_gl(lat_x1, lat_x2, lon_y1, lon_y2):
    x_mid = (lat_x1 + lat_x2) / 2
    max_grid_length = (distance.distance((x_mid, lon_y1), (x_mid, lon_y2)).km) / 2
    return max_grid_length

def insert_map_data(dataset, insertbox, insert_to_odps = False):
    #x维度
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
        print('Write to OrigindataMapInsert')
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
        '''if i % 10 == 0:
            #print(i)
            #print(count)
            if insert_to_odps:
                o.write_table('InsertData', box_r)
            box_r = []'''
    if insert_to_odps and len(box_r)!=0:
        print('Write to InsertData')
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
        #origin_r = []
    return origin_r

def prepare_data():
    data = pd.read_csv('data.csv')
    unused = 'ContributorSampleID,SampleNo,Longitude,Latitude,Way of Getting Location,Tectonic Domain,Tectonic Belt,Tectonic Region,Tectonic Event,Occurrence,Pluton Name,Lithology,Rock Texture,Rock Structure,Hornblende (Y/N),Biotite (Y/N),Sulfide (Y/N),Assumed Age (Ma),Era'
    unused2 = 'Year,Dataset Source,Contributor,owner,scope,update_time,Total,Garnet (Y/N),Tourmaline (Y/N),Beryl (Y/N),Fluorite (Y/N),Muscobite (Y/N)'
    unused = unused2.split(',') + unused.split(',') + ['First author', 'Age Analytical Method', 'Analyzed Age Error(+)', 'Volcanic Group', 'Measure Method', 'Measured Minerals', 'Mafics-percentage', 'Analyzed Age Error(-)', 'Analyzed Age (Ma)', 'Unnamed: 0']
    unused = unused + ['Reference-ALL', 'DOI', 'Yb', 'Pages', 'Volcanic Formation', '', 'Journal', 'Volume']
    feature_list = [col for col in data.columns if col not in unused]
    '''
        lat = list(data['Latitude'])
        lon = list(data['Longitude'])
        lat_lon = [str(lat[i]) + ',' + str(lon[i]) for i in range(len(lat))]
        data['lat_lon'] = lat_lon
        data = data.loc[~data['lat_lon'].duplicated(), :]
    '''
    postive_columns = ['K2O', 'Na2O', 'SiO2']
    for pc in postive_columns:
        data.loc[data[pc] < 0, pc] = 0
    data.index = range(len(data.index))
    data['index'] = data.index
    dataset = Dataset(data, '纽芬兰', feature_list)
    return dataset

def prepare_data_new(fn=None, datasetname=None):
    print(fn)
    df = pd.read_excel(fn, header=1)
    df = df.iloc[:-2]
    data = df[['Longitude', 'Latitude', 'Mapped εNd(t)', 'Mapped TDM (Ga)']]
    data.rename({'Mapped εNd(t)': 'εNd(t)', 'Mapped TDM (Ga)': 'TDM (Ga)'}, axis=1, inplace=True)
    data['εNd(t)'] = pd.to_numeric(data['εNd(t)'], errors='coerce')
    data['TDM (Ga)'] = pd.to_numeric(data['TDM (Ga)'], errors='coerce')
    data = data[data['Latitude'] <= 57]

    # feature_list = [col for col in data.columns]
    feature_list = ['εNd(t)', 'TDM (Ga)']
    # print(feature_list)
    lat = list(data['Latitude'])
    lon = list(data['Longitude'])
    lat_lon = [str(lat[i]) + ',' + str(lon[i]) for i in range(len(lat))]
    data['lat_lon'] = lat_lon
    data.index = range(len(data.index))
    data['index'] = data.index
    dataset = Dataset(data, datasetname, feature_list)
    return dataset

def prepare_data_global():
    data = pd.read_csv('global_data.csv')
    feature_list = []
    lat_lon_columns = ['Longitude', 'Latitude']
    for f in set(data.columns) - set(lat_lon_columns):
        if data[f].dtype == 'float64':
            feature_list.append(f)
    lat = list(data['Latitude'])
    lon = list(data['Longitude'])
    lat_lon = [str(lat[i]) + ',' + str(lon[i]) for i in range(len(lat))]
    data['lat_lon'] = lat_lon
    data = data.loc[~data['lat_lon'].duplicated(), :]
    data.index = range(len(data.index))
    data['index'] = data.index
    dataset = Dataset(data, 'global', feature_list)
    return dataset




def insert_map_data_iter(bouds, meandataset, grid_length, insert_to_odps = False):
    #每次按照分辨率的2倍划分网格，并计算样本点所在的box_index
    g = grid_length
    lat_min, lat_max, lon_min, lon_max = bouds
    max_grid_length = cal_max_gl(lat_min, lat_max, lon_min, lon_max)
    while g < max_grid_length:
        insertbox = InsertBox(lat_min, lat_max, lon_min, lon_max, grid_length=g, datasetName=meandataset.dataName)
        insert_box_data(insertbox, insert_to_odps = insert_to_odps)
        _ = insert_map_data(meandataset, insertbox, insert_to_odps=insert_to_odps)
        g = g * 2
        print(g)
        print(insertbox.y_num)

def prepare_mean_data(map_r, origin_r, insertbox, grid_length = 1,insert_to_odps = False, aggr_func = 'mean'):
    #####丁老师逻辑,每个网格将网格内所有点求平均作为该点的插值
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
    print('wwwwwwwwwwwwwwwwwwwwwww')
    print(len(mean_data_r))
    if insert_to_odps:
        o.write_table('MeanOriginData', mean_data_r)
    mean_data = mean_data.loc[~mean_data['box_index'].duplicated(), :]
    mean_data.columns = ['value', 'datasetName', 'name', 'index', 'Latitude', 'Longitude']
    mean_data.index = range(len(mean_data['name']))
    mean_data = Dataset(mean_data, mean_data.loc[0, 'datasetName'], feature_list=None)
    return mean_data

def pipeline(dataset, grid_length, insert_to_odps = False):
    #insertbox2 = InsertBox(-90, 90, -180, 180, grid_length=1, datasetName='global')
    insertbox = InsertBox(min(dataset.data['Latitude']), max(dataset.data['Latitude']), min(dataset.data['Longitude']), max(dataset.data['Longitude']), grid_length=grid_length, datasetName='纽芬兰')
    print(insertbox.allNum)
    insert_box_data(insertbox)
    map_r = insert_map_data(dataset, insertbox)
    ##Origin
    origin_r = insert_origin_data(dataset, insert_to_odps = insert_to_odps)
    ###MeanOriginData
    mean_data = prepare_mean_data(map_r, origin_r, insertbox, grid_length = grid_length, insert_to_odps=insert_to_odps)
    insert_map_data_iter([insertbox.lat_min, insertbox.lat_max, insertbox.lon_min, insertbox.lon_max], mean_data, grid_length=grid_length, insert_to_odps = insert_to_odps)

def save_csv_pipeline(dataset, grid_length, insert_to_odps = False):
    insertbox = InsertBox(min(dataset.data['Latitude']), max(dataset.data['Latitude']), min(dataset.data['Longitude']), max(dataset.data['Longitude']), grid_length=grid_length, datasetName='纽芬兰')
    print(insertbox.allNum)
    box_r = insert_box_data(insertbox)
    map_r = insert_map_data(dataset, insertbox)
    ##Origin
    origin_r = insert_origin_data(dataset, insert_to_odps=insert_to_odps)
    ###MeanOriginData
    mean_data = prepare_mean_data(map_r, origin_r, insertbox, grid_length = grid_length, insert_to_odps=insert_to_odps)
    
    # box_r = pd.DataFrame(box_r, columns=['datasetName', 'grid_length', 'box_x', 'box_y', 'box_index', 'box_lat', 'box_lon', 'tag'])
    # fn = f'grid_res-{grid_length}_data.csv'
    # box_r.to_csv(fn, index = False)
    fn = f'mean_res-{grid_length}_data.csv'
    mean_data = pd.DataFrame(mean_data, columns=['datasetName', 'index', 'lat', 'lon', 'name', 'value', 'grid_length', 'aggregate_func'])
    mean_data.to_csv(fn, index = False)

def stat_insert_data_num(dataset, grid_length_list):
    grid_l_set = set()
    all_num = 0
    lat_min = min(dataset.data['Latitude'])
    lat_max = max(dataset.data['Latitude'])
    lon_min = min(dataset.data['Longitude'])
    lon_max = max(dataset.data['Longitude'])
    for gl in grid_length_list:
        num = 0
        if gl not in grid_l_set:
            max_grid_length = max_grid_length = cal_max_gl(lat_min, lat_max, lon_min, lon_max)
            print(max_grid_length)
            g = gl
            num = 0
            while g < max_grid_length:
                insertbox2 = InsertBox(min(dataset.data['Latitude']), max(dataset.data['Latitude']), min(dataset.data['Longitude']), max(dataset.data['Longitude']), grid_length=g, datasetName='global')
                print('-------')
                print(g)
                print(insertbox2.x_num)
                print(insertbox2.y_num)
                num += insertbox2.allNum
                grid_l_set.add(g)
                while g in grid_l_set:
                    g =g * 2
        all_num += num
    print(all_num)
    print(grid_l_set)
if __name__ == '__main__':
    #dataset = prepare_data_global()
    dataset = prepare_data()
    #dataset.region('Newfoundland', type = 'belt')
    print(len(dataset.data.index))
    dataset.add_feature_div('K2O/Na2O', 'K2O', 'Na2O')
    dataset.add_feature_div('Rb/Sr', 'Rb', 'Sr')
    dataset.add_feature_log('log10(Rb/Sr)', 'Rb/Sr')
    #dataset.feature_list = ['K2O/Na2O']
    #print(dataset.data['K20/Na2O'])
    print(dataset.feature_list)
    # print(max(dataset.data['Latitude']))
    # pipeline(dataset, grid_length=1, insert_to_odps=False)
    #insertbox = InsertBox(min(dataset.data['Latitude']), max(dataset.data['Latitude']), min(dataset.data['Longitude']), max(dataset.data['Longitude']), grid_length=32, datasetName='纽芬兰')
    #print(insertbox.x_num)
    #print(insertbox.y_num)
    #insertbox2 = InsertBox(min(dataset.data['Latitude']), max(dataset.data['Latitude']), min(dataset.data['Longitude']), max(dataset.data['Longitude']), grid_length=2, datasetName='纽芬兰')
    #print(insertbox.x_gap)
    #insert_box_data(insertbox)
    #insert_map_data(dataset, insertbox)
    #insert_origin_data(dataset)
    #insert_map_data_iter(dataset, grid_length=5)

    #box_ind = insertbox.trans_lat_lon_to_box_index(46.74127829261257, -55.66182450189608)
    #print(box_ind)

    #stat_insert_data_num(dataset, [1])