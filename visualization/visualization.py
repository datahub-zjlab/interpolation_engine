import os
import re
import warnings

import contextily as ctx
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traceback2 as traceback

warnings.filterwarnings("ignore", category=FutureWarning)
from utils.odps_util import Odps

o = Odps().get_odps_instance()

# 数据加载
try:
    import database.util as uti
    _method_func_map = uti.get_method_func_map()
except ModuleNotFoundError:
    _method_func_map = {'krige':'kriging', 'rbf':'rbf', 'nn':'nearest_neighbour', 'idw':'idw'}

def get_result_data_from_method_id(method_name, method_id):
    with o.execute_sql(f'select * from Result_{_method_func_map.get(method_name)} where method_id = {method_id}').open_reader( # Result表需要对应方法
            tunnel=True) as reader:
        data = reader.to_pandas()
    return data

# 绘制柱状图
def plot_stat(df_ori, df_imp, name, num_bins=10, custom_legend=None, custom_shape=None, save_path='./'):
    # 原始数据选择需要的列
    df_ori[name] = pd.to_numeric(df_ori[name], errors='coerce')
    df_ori = df_ori[['Longitude', 'Latitude', name]]
    df_ori = df_ori.rename(columns={'Longitude': 'box_lon', 'Latitude': 'box_lat', name: 'value'})

    # 读取地图边界
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    canada = world[world['name'] == 'Canada']
    
    df_list = [df_ori, df_imp]
    for i in range(2):
        if i == 0:
            print("原始数据：")
        elif i == 1:
            print("插值数据：")
        df = df_list[i].copy()
        
        # 使用 GeoPandas 进行空间裁剪
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df['box_lon'], df['box_lat'])
        )
        gdf.set_crs(epsg=4326, inplace=True)
        if custom_shape is not None:
            gdf_in = gpd.clip(gdf, custom_shape).copy()
        else:
            gdf_in = gpd.clip(gdf, canada).copy()

        # 设置数字区间
        if custom_legend is None:
            desc = df_ori['value'].describe()
            Q1 = desc['25%']
            Q3 = desc['75%']
            IQR = Q3 - Q1
            r = 1.5
            lower_bound = Q1 - r * IQR
            upper_bound = Q3 + r * IQR
            print("利用1.5倍四分位距（IQR）对异常值进行检测，原始数据正常范围：", lower_bound, upper_bound)
            num_outliers = ((gdf_in['value'] < lower_bound) | (gdf_in['value'] > upper_bound)).sum()
            print(f"剔除{num_outliers}个异常值作图。")
            bounds = np.linspace(lower_bound, upper_bound, num_bins+1)
            bounds = np.round(bounds, 2)
        else:
            print("使用自定义数字区间作图。")
            bounds = custom_legend

        # 绘制柱状图
        plt.figure(figsize=(12, 12))
        gdf_in['bin'] = pd.cut(gdf_in['value'], bins=bounds, include_lowest=True, right=False)
        bin_counts = gdf_in['bin'].value_counts().sort_index()
        bars = plt.bar(bin_counts.index.astype(str), bin_counts.values, width=1, edgecolor='black')
        plt.xticks(rotation=45)
        for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, int(yval), ha='center', va='bottom')
        plt.xlabel('Longitude')
        plt.ylabel('Frequency')
        safe_name = re.sub(r'[\/:*?"<>|]', '_', name)
        ## 原始数据
        if i == 0:
            plt.title(name + ' - Original')
            plt.savefig(os.path.join(save_path, safe_name + '_original_hist.png'))
        ## 插值数据
        elif i == 1:
            plt.title(name + ' - Interpolation')
            plt.savefig(os.path.join(save_path, safe_name + '_interpolation_hist.png'))
            
# 绘制等值线图
def plot_contour(df, df_imp, name, alpha=0.8, show_contour=False, custom_legend=None, custom_shape=None, basemap=None, save_path='./'):

    # 读取地图边界
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    canada = world[world['name'] == 'Canada']

    # 使用 GeoPandas 进行空间裁剪
    gdf = gpd.GeoDataFrame(
        df_imp,
        geometry=gpd.points_from_xy(df_imp['box_lon'], df_imp['box_lat'])
    )
    gdf.set_crs(epsg=4326, inplace=True)
    if custom_shape is not None:
        gdf_in = gpd.clip(gdf, custom_shape).copy()
    else:
        gdf_in = gpd.clip(gdf, canada).copy()

    # 生成网格数据
    grid_lon = np.unique(gdf['box_lon'])
    grid_lat = np.unique(gdf['box_lat'])
    grid_value = pd.DataFrame(index=grid_lat, columns=grid_lon, dtype=float)
    grid_value[:] = np.nan
    for idx, row in gdf_in.iterrows():
        lat_idx = np.where(grid_lat == row['box_lat'])[0]
        lon_idx = np.where(grid_lon == row['box_lon'])[0]
        grid_value.iloc[lat_idx[0], lon_idx[0]] = row['value']
        
    # 设置图形大小
    fig, ax = plt.subplots(figsize=(14, 8))

    # 设置图例
    ## 预置颜色表
    custom_colors = ['#232172', '#30266d', '#3c3889', '#454195',
                     '#5456a0', '#847fb9', '#b7b5c3', '#e2e4c0',
                     '#f3ec9e', '#e4be77', '#b68238', '#ce8f18',
                     '#ca761b', '#bc6020', '#a0432f', '#852829'] 
    ## 默认数字区间
    df[name] = pd.to_numeric(df[name], errors='coerce')
    desc = df[name].describe()
    # print(desc)

    ## 根据custom_legend输入
    if custom_legend is None or isinstance(custom_legend, int):
        # 原始数据分布
        Q1 = desc['25%']
        Q3 = desc['75%']
        IQR = Q3 - Q1
        r = 1.5
        lower_bound = Q1 - r * IQR
        upper_bound = Q3 + r * IQR
        print("利用1.5倍四分位距（IQR）对异常值进行检测，原始数据正常范围：", lower_bound, upper_bound)
        num_outliers = ((gdf_in['value'] < lower_bound) | (gdf_in['value'] > upper_bound)).sum()
        print(f"剔除{num_outliers}个异常值作图。")
        if custom_legend is None:
            print("原始数据正常范围六等分作图。")
            bounds = np.linspace(lower_bound, upper_bound, 7)
            bounds = np.round(bounds, 4)
        else:
            print(f"原始数据正常范围{custom_legend}等分作图。")
            bounds = np.linspace(lower_bound, upper_bound, custom_legend+1)
            bounds = np.round(bounds, 4)
    elif isinstance(custom_legend, list):
        print("使用自定义数字区间作图。")
        bounds = custom_legend
    else:
        raise ValueError("custom_legend的类型无效，请提供None、整数或列表。")
    # print(bounds)
             
    ## 根据数字区间分配颜色
    n_colors = len(bounds) - 1
    indices = np.linspace(0, len(custom_colors) - 1, n_colors).astype(int) # 从颜色表中等距选择N个颜色
    selected_colors = [custom_colors[idx] for idx in indices]

    # 绘制等值线图
    cmap = mcolors.ListedColormap(selected_colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    contourf = ax.contourf(grid_lon, grid_lat, grid_value, levels=bounds, cmap=cmap, norm=norm, alpha=alpha)

    # 添加图例
    legend_labels = [f'{bounds[i]}-{bounds[i+1]}' for i in range(len(bounds)-1)]
    legend_patches = [mpatches.Patch(color=selected_colors[i], label=legend_labels[i]) for i in range(len(legend_labels))]
    ax.legend(handles=legend_patches, title=name, bbox_to_anchor=(1.05, 1), loc='upper left')

    # 绘制等值线
    if show_contour:
        ax.contour(grid_lon, grid_lat, grid_value, colors='k', linewidths=0.5)  

    # 添加底图
    if basemap is not None:
        try:
            ctx.add_basemap(ax, crs=canada.crs, source=getattr(ctx.providers.Esri, basemap))
            print(f"使用{basemap}底图。")
        except:
            print(f"不支持的底图类型{basemap}，使用默认底图。")
            canada.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
    else:
        print("使用默认底图。")
        canada.boundary.plot(ax=ax, edgecolor='black', linewidth=1)

    # 设定坐标范围
    min_lon, max_lon = df_imp['box_lon'].min(), df_imp['box_lon'].max()
    min_lat, max_lat = df_imp['box_lat'].min(), df_imp['box_lat'].max()
    plt.xlim(min_lon-1, max_lon+1)
    plt.ylim(min_lat-1, max_lat+1)

    # 添加灰色网格线
    plt.grid(color='gray', linestyle='-', linewidth=0.5)

    # 设置图形等宽比例
    plt.gca().set_aspect('equal')

    # 添加标题
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(name)

    # 存储图像
    safe_name = re.sub(r'[\/:*?"<>|]', '_', name)
    plt.savefig(os.path.join(save_path, safe_name + '_contour.png'))


def all_feature_plot(method_name, feature_dic, alpha=0.8, show_contour=False, custom_legend=None, custom_shape=None, save_path='./'):
    df = pd.read_csv('../../../../interpolationengine/visualization/data.csv')
    for f in feature_dic:
        print('--------------------')
        print(f)
        try:
            df_imp = get_result_data_from_method_id(method_name, feature_dic[f])
            # print(df_imp)
            if df_imp['value'].nunique() > 1:
                plot_contour(df, df_imp, name=f, alpha=alpha, show_contour=show_contour, custom_legend=custom_legend, custom_shape=custom_shape,
                             save_path=save_path)
            else:
                raise ValueError("原始数据只有一个值。")
        except Exception as e:
            # print(e)
            traceback.print_exc() 
            continue

if __name__ == '__main__':
    
    # 原始数据
    df = pd.read_csv('../../../../interpolationengine/visualization/data.csv')
    df['K2O/Na2O'] = df['K2O']/df['Na2O']

    # 单一特征
    df_imp = pd.read_csv('../../../../interpolationengine/interpolation_local/nearest_neighbor/nn_SiO2.csv') # 本地数据
    # df_imp = get_result_data_from_method_id('nn', 26540) # ODPS数据
    
    # plot_stat(df, df_imp, name='SiO2', num_bins=10, custom_legend=[0,10,20,30,40,50,60,70,80,90,100], custom_shape=None, save_path='./')
    
    plot_contour(df, df_imp, name='SiO2', alpha=0.8, show_contour=False, custom_legend=8, custom_shape=None, basemap=None, save_path='../../../../interpolationengine/interpolation_local/nearest_neighbor')
    
    # basemap_list = ['WorldStreetMap', 'DeLorme', 'WorldTopoMap', 'WorldImagery', 'WorldTerrain', 
    #                 'WorldShadedRelief', 'WorldPhysical', 'OceanBasemap', 'NatGeoWorldMap', 'WorldGrayCanvas', 
    #                 'ArcticImagery', 'ArcticOceanBase', 'ArcticOceanReference', 'AntarcticImagery', 'AntarcticBasemap'] # 不支持的底图DeLorme和AntarcticBasemap
    # for bm in basemap_list:
    #     plot_contour(df, df_imp, name='SiO2', alpha=0.5, show_contour=False, custom_legend=None, custom_shape=None, basemap=bm, save_path='./')

    # # 多个特征 from 本地
    # df_all = pd.read_csv('../interpolation_local/nearest_neighbor/nn_grid_2.csv')
    # f_list = ['SiO2', 'Na2O', 'K2O', 'K2O/Na2O']
    # for f in f_list:
    #     df_imp = df_all[df_all['feature'] == f]
    #     plot_contour(df, df_imp, name=f, alpha=0.8, show_contour=False, custom_legend=None, custom_shape=None, basemap=None, save_path='../interpolation_local/nearest_neighbor/plot')

    # # 多个特征 from ODPS
    # df_res = pd.read_csv('../interpolation_odps/results/nn_100-feats.csv') 
    # feature_dict = df_res.set_index('Feature')['Method_id'].to_dict()
    # print(feature_dict)
    # method_name = 'nn'
    # all_feature_plot(method_name, feature_dict, alpha=0.8, show_contour=False, custom_legend=None, custom_shape=None, basemap='WorldStreetMap', save_path='../interpolation_odps/nearest_neighbor/plot')