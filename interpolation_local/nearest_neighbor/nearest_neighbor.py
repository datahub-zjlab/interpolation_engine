import pandas as pd
from scipy.interpolate import NearestNDInterpolator
import multiprocessing
from utils.odps_util import Odps

o = Odps().get_odps_instance()

n_process = multiprocessing.cpu_count()

def get_combined_data_by_feature_and_resolution_odps(feature, res, save_file=False, aggr_func='mean', dataset='纽芬兰'):
    print('load data')
    sql = f"""
with grid as (
  select box_x, box_y, box_index, box_lat, box_lon
  from insertdata
  where datasetname = '{dataset}' and grid_length = {res}),
orig as (
  select index, name, value
  from meanorigindata
  where datasetname = '{dataset}'
  and name='{feature}'
  and grid_length = {res}
)
select grid.box_index, grid.box_lat, grid.box_lon, orig.value
from grid left join orig
  on grid.box_index = orig.index;"""
    with o.execute_sql(sql).open_reader(
            tunnel=True) as reader:
        data = reader.to_pandas(n_process=n_process)

    if save_file:
        fn = f'{feature}_{res}_data.csv'
        data.to_csv(fn, index = False)
        print(fn, '\ndone')
    else:
        print('done')

    return data

if __name__ == '__main__':

    grid_length = 1
    f_list =  [
        'SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'Fe2O3T', 'FeOT', 'MnO', 'MgO', 'CaO', 'Na2O', 
        'K2O', 'P2O5', 'F', 'LOI', 'La', 'Ce', 'Y', 'Ba', 'Be', 'Cr', 
        'Cu', 'Ga', 'Li', 'Mo', 'Nb', 'Ni', 'Pb', 'Rb', 'Sr', 'Th', 
        'U', 'V', 'Zn', 'Zr', 'FeO', 'Co', 'Sn', 'W', 'Dy', 'Sc', 
        'Cd', 'H2O-', 'CO2', 'Sm', 'Eu', 'Tb', 'Lu', 'Ag', 'As', 'Au',
        'Br', 'Cs', 'Hf', 'S', 'Se', 'Ta', 'Sb', 'Ir', 'Te', 'Nd', 
        'Hg', 'Pr', 'Gd', 'Ho', 'Er', 'Tm', 'Sm(m)', 'Nd(m)', '147Sm/144Nd(m)', '143Nd/144Nd(m)', 
        'f(Sm/Nd)', 'εNd(t)', 'TDM1 (Ma)', 'Recalcul εNd(0)', 'Recalcul εNd(t)', 'Recalcul TDM1-Nd(Ma)', 'Recalcul TDM2-Nd (Ma)', 'CL', 'Pd', 'Pt', 
        'Rb(m)', 'Sr(m)', '87Rb/86Sr(m)', '87Sr/86Sr(m)', '87Sr/86Sr(i)', 'Ge', 'Mn', 'Ti', 'Tl', 'In',
        'Bi', 'δ18O', '208Pb/204Pb(m)', '207Pb/204Pb(m)', '206Pb/204Pb(m)', 'P', 'TDM2 (Ma)', 'K2O/Na2O', 'Rb/Sr', 'log10(Rb/Sr)']
    # f_list =  ['SiO2', 'Na2O', 'K2O', 'K2O/Na2O']

    all_feature_data = []
    for f in f_list:
        print(f, grid_length)

        # 提取对应特征数据
        data = get_combined_data_by_feature_and_resolution_odps(f, grid_length, save_file=False)
        print("总网格数：", len(data))

        # 获取有数值的网格数据
        data_with_values = data.dropna(subset=['value'])
        print("有数值的网格数：", len(data_with_values))

        # 获取需要插值的网格数据
        data_to_interpolate = data[data['value'].isna()]
        print("需要插值的网格数：", len(data_to_interpolate))
        
        # 最近邻插值
        points = data_with_values[['box_lon', 'box_lat']].values
        values = data_with_values['value'].values
        interpolator = NearestNDInterpolator(points, values)
        interpolated_values = interpolator(data_to_interpolate[['box_lon', 'box_lat']].values)
        data_to_interpolate.loc[:, 'value'] = interpolated_values

        # 合并结果
        df_interpolated = pd.concat([data_with_values, data_to_interpolate])
        df_interpolated = df_interpolated.sort_values(by=['box_index'])
        print(df_interpolated.shape)

        # 添加至结果总表
        df_interpolated['feature'] = f
        all_feature_data.append(df_interpolated)

    # 存储结果
    df_all = pd.concat(all_feature_data)
    print(df_all.shape)
    df_all.to_csv('nn_grid_' + str(grid_length) + '.csv', index = False)