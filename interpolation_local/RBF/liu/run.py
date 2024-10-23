from scipy.interpolate import Rbf
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os

def get_df(filename):
    path = "./source_data/"+filename +".csv"
    df = pd.read_csv(path)
    return df

def funcs(features,func_names,lens = 1):

    for func_name in func_names:
        lat = []
        lon = []
        re_value = []
        re_method = []
        re_index = []
        for j in features:
            df = get_df(j)
            df_value = df[df['value'].notnull()]
            df_nan = df[df['value'].isnull()]
            func = Rbf( df_value['box_lat'].values,df_value['box_lon'].values, df_value['value'].values, function=func_name)
            try:
                values = func(df_nan['box_lat'].values,df_nan['box_lon'].values)
            except:
                print(j,"error")
                continue

            #存储计算的插值
            lat.extend(list(df_nan['box_lat'].values))
            lon.extend(list(df_nan['box_lon'].values))
            re_index.extend(list(df_nan['box_index'].values))
            re_value.extend(values)
            re_method.extend([j]*len(values))

            #存储已存在的插值
            lat.extend(list(df_value['box_lat'].values))
            lon.extend(list(df_value['box_lon'].values))
            re_index.extend(list(df_value['box_index'].values))
            re_value.extend(list(df_value['value'].values))
            re_method.extend([j]*len(df_value))
            
            print(j,lens,"ok")

        output={"box_index":re_index,
                'method_id':re_method,
            "box_lat":lat,
            "box_lon":lon,
            "value":re_value}

        t = pd.DataFrame(output)

        store_dir = "./data/rbf/dataset4/"
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        
        t.to_csv(f"{store_dir}{func_name}_{lens}.csv",index=False)
        print(f"{store_dir}{func_name}_{lens}.csv"," ok")


all_feats = ['Au', 'Na2O', 'U', 'Zn', 'Mn', 'Mo', '87Rb/86Sr(m)', 'P', 'CaO', 'Eu', 'Hf', '87Sr/86Sr(m)',
                 'TDM2 (Ma)', 'Te', 'MnO', 'H2O-', 'Nd', 'Br', 'FeOT', 'Cd', 'Er', 'Al2O3', '207Pb/204Pb(m)',
                 'CL', 'Sr(m)', '147Sm/144Nd(m)', 'As', 'La', 'Hg', 'Gd', 'Cu', 'Pt', 'Rb', 'εNd(t)', 'FeO',
                 '208Pb/204Pb(m)', 'Bi', 'Fe2O3', 'Pd', '143Nd/144Nd(m)', 'Th', 'Pr', 'Sr', 'Zr', 'P2O5', 'MgO',
                 'Rb(m)', 'F', 'Ba', 'Ce', 'Cr', 'Recalcul TDM2-Nd (Ma)', 'Ir', 'S', 'CO2', 'Sb', 'Ho',
                 'Recalcul εNd(t)', 'f(Sm/Nd)', 'TDM1 (Ma)', 'Rb/Sr', '206Pb/204Pb(m)', 'Cs', 'Be', 'Recalcul εNd(0)',
                 'Ag', 'Ga', 'In', 'Y', 'Se', 'Ta', 'LOI', 'W', 'K2O', 'Fe2O3T', 'Pb', 'Tm', 'Nb', 'Nd(m)',
                 'Ti', 'Sm(m)', 'K2O/Na2O', 'SiO2', 'Ge', 'δ18O', 'Co', 'Tb', 'Li', 'TiO2', '87Sr/86Sr(i)',
                 'log10(Rb/Sr)', 'Tl', 'Sm', 'Recalcul TDM1-Nd(Ma)', 'Sc', 'V', 'Lu', 'Dy', 'Sn', 'Ni']

func_name = ["multiquadric","gaussian","linear"]
all_feats = ["dataset4_Nd","dataset4_TDM"]

for lens in [10]:
    
    funcs(all_feats,func_name,lens)

