import os
from odps import options
import multiprocessing
import time
# from pykrige.ok import OrdinaryKriging
import pandas as pd
from utils.odps_util import Odps

o = Odps().get_odps_instance()

env_settings = {
    'odps.sql.allow.cartesian': True,
    # 'in_predicate_conversion_threshold': 2,
    'odps.sql.python.version': 'cp37',
    'odps.sql.mapper.memory': 2048,
    # 'odps.sql.mapper.cpu': 100,
    'odps.sql.mapper.split.size': 1,
    'odps.sql.reducer.memory': 2048,
    # 'odps.sql.reducer.cpu': 100,
    'odps.sql.reducer.instances': 80,
    # 'options.tunnel.limit_instance_tunnel': True,
}
options.sql.settings = env_settings
n_process = multiprocessing.cpu_count()

_thresh = 5
_radius = 20
_feature = 'Dy'
_dataset = '纽芬兰'
_method_name = 'krige'  # 'rbf' 'nn'
_method_func_map = {'krige': 'kriging', 'rbf': 'rbf', 'nn': 'nearest_neighbour', 'idw': 'idw', 'lpi': 'lpi'}
_model_name = 'gaussian'  # 'gaussian' 'linear'
_method_id = 100
_aggre_func = 'mean'
_param = None
_grid_dict = {1: 475, 2: 237, 4: 118, 8: 59, 16: 29, 32: 14, 64: 7, 128: 3}
_resolution = 2  # 默认待插值网格尺寸
_search_starting_res = 8  # 初始搜索分辨率
_grid_size = _grid_dict.get(_resolution)
_save_path = "results/"
if not os.path.exists(_save_path):
    os.makedirs(_save_path)


class Method:
    def __init__(self, m_id=None, method_name=_method_name, params={'variogram_model': _model_name}, thresh=_thresh,
                 dataset=_dataset, resolution=_resolution, feature=_feature, aggr_func=_aggre_func):
        self.m_id = m_id
        self.method_name = method_name
        self.func_name = _method_func_map.get(method_name)
        self.params = params
        self.thresh = thresh
        self.dataset = dataset
        self.resolution = resolution
        self.feature = feature
        self.aggre_func = aggr_func

    def trans_params(self, params):
        params = [(p, params[p]) for p in params.keys()]
        params.sort(key=lambda x: x[0])
        params_v = [str(i[1]) for i in params]
        parms_k = [str(i[0]) for i in params]
        params_str = ",".join(parms_k) + ':' + ','.join(params_v)
        return params_str

    def get_new_method_id_from_result(self):
        sql = f"select max(method_id) from result_{self.func_name};"
        with o.execute_sql(sql).open_reader(tunnel=True) as reader:
            m_id = reader[0][0]
        self.m_id = m_id + 1 if m_id is not None else 1
        return self.m_id

    def insert_to_insertMethod_if_not_exists(self, m_id=0):
        sql_str = 'select * from insertmethod;'
        with o.execute_sql(sql_str).open_reader(
                tunnel=True) as reader:
            pd_df = reader.to_pandas()
        max_id = pd_df['method_id'].max() if len(pd_df.index) > 0 else 0
        pd_df.set_index(
            ['insert_method', 'params', 'n_closest_points', 'datasetname', 'grid_length', 'name', 'aggregate_func'],
            inplace=True)
        insert_data = [self.method_name, self.trans_params(self.params), self.thresh, self.dataset, self.resolution,
                       self.feature, self.aggre_func]

        # if tuple(insert_data) in pd_df.index:
        #     # method_id = None          # TODO
        #     method_id = max_id + 1
        #     print(f"method ID {method_id} existed, +1")
        # else:
        method_id = max_id + 1
        print('insert:' + str([[method_id] + insert_data]))
        o.write_table('insertmethod', [[method_id] + insert_data])
        self.m_id = method_id
        return method_id


class Interpolation:
    def __init__(self, method: Method, default_res=_resolution, grid_dict=_grid_dict, origin_res=1):
        self.method = method
        # self.aggre_func = method.aggre_func
        self.default_res = default_res  # 待插值网格尺寸
        self.origin_data_res = origin_res  # 原始点所在网格尺寸
        self.grid_dict = grid_dict
        self.model_param = None
        # self.grid_size = self.grid_dict.get(self.method.resolution)
        self.grid_default_size = self.grid_dict.get(self.default_res)
        self.radius = _radius
        if method.resolution != self.default_res:
            raise ValueError("default_res must be equal to method.resolution")

    def setup(self, default_res=1, aggre_func=_aggre_func,
              grid_dict=_grid_dict, radius=_radius, param=_param):
        # self.aggre_func = aggre_func
        self.default_res = default_res
        self.radius = radius
        self.grid_dict = grid_dict
        self.model_param = param

    def get_create_result_table_sql(self):
        return f"""insert into result_{self.method.func_name} (box_index, box_lat, box_lon, method_id)
select box_index, box_lat, box_lon, {self.method.m_id}
from insertdata
where datasetname = '{self.method.dataset}' and grid_length={self.default_res};"""

    def get_update_existing_val_sql(self):
        return f"""update result_{self.method.func_name}
set value = val
from  (select mapp.box_index as bidx, {'avg' if self.method.aggre_func == 'mean' else self.method.aggre_func}(orig.value) as val
    from origindatamapinsert as mapp, meanorigindata as orig
    where  mapp.index = orig.index
    and mapp.grid_length = {self.default_res}
    and orig.name = '{self.method.feature}'
    and aggregate_func = '{self.method.aggre_func}'
    and orig.datasetname = '{self.method.dataset}'
    and mapp.datasetname = '{self.method.dataset}'
    group by bidx)
where result_{self.method.func_name}.box_index = bidx
and method_id = {self.method.m_id};"""

    def get_prefix_sql(self, resolution):
        return f"""with to_search as ( 
select tmp.box_index, tmp.box_x, tmp.box_y, tmp.box_lat, tmp.box_lon
from insertdata as tmp, result_{self.method.func_name}
where tmp.datasetname = '{self.method.dataset}'
    and tmp.grid_length = {self.default_res}
    and result_{self.method.func_name}.box_index = tmp.box_index
    and result_{self.method.func_name}.value is Null
    and result_{self.method.func_name}.method_id = {self.method.m_id}), 
grid as (
select tmp.box_x, tmp.box_y, tmp.box_index, tmp.box_lat, tmp.box_lon
from insertdata as tmp
where tmp.datasetname = '{self.method.dataset}'
    and tmp.grid_length = {resolution}),
mapp as (
select index, box_index
from origindatamapinsert
where datasetname = '{self.method.dataset}'
    and grid_length = {resolution})
"""

    def get_interpol_sql(self, resolution=_resolution, grid_size=_grid_size):
        return f"""select to_search.box_index, dointerpolate(to_search.box_lat, to_search.box_lon, nei, {self.method.thresh}, 
'{self.method.params['variogram_model']}', array({self.model_param[0]}, {self.model_param[1]})) as val
from to_search,
    (select grid.box_index as b_idx,
        findneigh(orig.lat, orig.lon, grid.box_lat, grid.box_lon, orig.value, {self.method.thresh}) as nei
    from    mapp, meanorigindata as orig, grid
    where   mapp.index = orig.index and
        mapp.box_index div {grid_size} between grid.box_x-{self.radius} and grid.box_x+{self.radius} and
        mapp.box_index % {grid_size} between grid.box_y-{self.radius} and grid.box_y+{self.radius}
        and orig.name='{self.method.feature}'
        and orig.aggregate_func='{self.method.aggre_func}'
        and orig.grid_length={self.origin_data_res}
    group by grid.box_index)
where   ((to_search.box_x div {resolution})*{grid_size} + to_search.box_y div {resolution}) = b_idx;"""

    # orig index (x,y) --> index of cur res

    def get_fast_interpol_sql(self, resolution=_resolution, grid_size=_grid_size):
        # 调用不同插值方法时候使用不同的参数配置。
        params_str = ""
        if self.method.method_name == 'lpi':
            # self.method.params = {'degree': 2}
            # params_str = ', '.join(f'{k}={v}' for k, v in self.method.params.items())
            params_str = self.method.params['degree']
        else:  # krige, rbf, idw, nn
            param_array = ','.join(str(i) for i in self.model_param)
            params_str = f'{self.method.params["variogram_model"]}, array({param_array})'

        return f"""select ts.box_index, {self.method.func_name}(ts.box_lat, ts.box_lon,
findneigh(orig.lat, orig.lon, ts.box_lat, ts.box_lon, orig.value, {self.method.thresh}),
{self.method.thresh}, {params_str}) as val
from to_search as ts, meanorigindata as orig, (
    select grid.box_index as b_idx, mapp.index as m_idx
    from    mapp, grid
    where  
        mapp.box_index div {grid_size} = grid.box_x and
        mapp.box_index % {grid_size} = grid.box_y 
    union
    select grid.box_index as b_idx, mapp.index as m_idx
    from    mapp, grid
    where  
        mapp.box_index div {grid_size} = grid.box_x-1 and
        mapp.box_index % {grid_size} = grid.box_y 
    union
    select grid.box_index as b_idx, mapp.index as m_idx
    from    mapp, grid
    where  
        mapp.box_index div {grid_size} = grid.box_x and
        mapp.box_index % {grid_size} = grid.box_y-1
    union
    select grid.box_index as b_idx, mapp.index as m_idx
    from    mapp, grid
    where  
        mapp.box_index div {grid_size} = grid.box_x+1 and
        mapp.box_index % {grid_size} = grid.box_y 
    union
    select grid.box_index as b_idx, mapp.index as m_idx
    from    mapp, grid
    where  
        mapp.box_index div {grid_size} = grid.box_x and
        mapp.box_index % {grid_size} = grid.box_y+1
    union
    select grid.box_index as b_idx, mapp.index as m_idx
    from    mapp, grid
    where  
        mapp.box_index div {grid_size} = grid.box_x-1 and
        mapp.box_index % {grid_size} = grid.box_y-1 
    union
    select grid.box_index as b_idx, mapp.index as m_idx
    from    mapp, grid
    where  
        mapp.box_index div {grid_size} = grid.box_x+1 and
        mapp.box_index % {grid_size} = grid.box_y+1 
    union
    select grid.box_index as b_idx, mapp.index as m_idx
    from    mapp, grid
    where  
        mapp.box_index div {grid_size} = grid.box_x+1 and
        mapp.box_index % {grid_size} = grid.box_y-1 
    union
    select grid.box_index as b_idx, mapp.index as m_idx
    from    mapp, grid
    where  
        mapp.box_index div {grid_size} = grid.box_x-1 and
        mapp.box_index % {grid_size} = grid.box_y+1 
    ) as nei
where ((ts.box_x div ({resolution} div {self.default_res}))*{grid_size} + ts.box_y div ({resolution} div {self.default_res})) = nei.b_idx
    and orig.index = nei.m_idx
    and orig.name='{self.method.feature}'
    and orig.aggregate_func='{self.method.aggre_func}'
    and orig.grid_length={self.origin_data_res}
group by ts.box_index, ts.box_lat, ts.box_lon;"""

    # and (orig.index div {self.grid_default_size} div {resolution} * {grid_size} + orig.index % {self.grid_default_size} div {resolution}) = nei.m_idx

    def get_insert_temp_sql(self, resolution=_resolution, grid_size=_grid_size):
        return f"""insert into temp_{self.method.func_name} (idx, value, method_id)
select box_index, val, {self.method.m_id}
from ({self.get_fast_interpol_sql(resolution=resolution, grid_size=grid_size)[:-1]})
where val is not null; """

    def get_update_sql(self):
        return f"""update result_{self.method.func_name}
set value = tmp.value
from  (select distinct idx, value from temp_{self.method.func_name} where method_id={self.method.m_id}) as tmp
where box_index=tmp.idx
    and value is null
    and method_id={self.method.m_id};"""

    def get_query_sql(self):
        return f"""select count(box_index) from result_{self.method.func_name} where method_id={self.method.m_id} and value is null;"""

    # =================================== neigh =============================
    def get_neigh_prefix_sql(self, resolution):
        return f"""with to_search as ( 
select tmp.box_index, tmp.box_x, tmp.box_y, tmp.box_lat, tmp.box_lon
from insertdata as tmp, neighbours
where tmp.datasetname = '{self.method.dataset}'
    and tmp.grid_length = {self.default_res}
    and neighbours.idx = tmp.box_index
    and neighbours.neigh is Null),
grid as (
select tmp.box_x, tmp.box_y, tmp.box_index, tmp.box_lat, tmp.box_lon
from insertdata as tmp
where tmp.datasetname = '{self.method.dataset}'
    and tmp.grid_length = {resolution}),
mapp as (
select index, box_index
from origindatamapinsert
where datasetname = '{self.method.dataset}'
    and grid_length = {resolution})
"""

    def get_neigh_sql(self, resolution=_resolution, grid_size=_grid_size):
        return f"""select to_search.box_index, findneigh(orig.lat, orig.lon, to_search.box_lat, to_search.box_lon, orig.value, {self.method.thresh}) as nei
from    meanorigindata as orig, to_search
where   orig.index div ({grid_size}) between to_search.box_x-{self.radius} and to_search.box_x+{self.radius} and
    orig.index % ({grid_size}) between to_search.box_y-{self.radius} and to_search.box_y+{self.radius} and
    orig.name='{self.method.feature}'
    and orig.aggregate_func='{self.method.aggre_func}'
    and orig.grid_length={self.origin_data_res}
group by to_search.box_index;
"""

    def get_neigh_fast_sql(self, resolution=_resolution, grid_size=_grid_size):
        return f"""select a.box_index,
    findneighonly(lat, lon, a.box_lat, a.box_lon, m_idx, {self.method.thresh}) as nei
from to_search as a, meanorigindata as orig, (
    select grid.box_index as b_idx, grid.box_lat, grid.box_lon, mapp.index as m_idx
    from    mapp, grid
    where  
        mapp.box_index div {grid_size} = grid.box_x and
        mapp.box_index % {grid_size} = grid.box_y 
    union
    select grid.box_index as b_idx, grid.box_lat, grid.box_lon, mapp.index as m_idx
    from    mapp, grid
    where  
        mapp.box_index div {grid_size} = grid.box_x-1 and
        mapp.box_index % {grid_size} = grid.box_y 
    union
    select grid.box_index as b_idx, grid.box_lat, grid.box_lon, mapp.index as m_idx
    from    mapp, grid
    where  
        mapp.box_index div {grid_size} = grid.box_x and
        mapp.box_index % {grid_size} = grid.box_y-1
    union
    select grid.box_index as b_idx, grid.box_lat, grid.box_lon, mapp.index as m_idx
    from    mapp, grid
    where  
        mapp.box_index div {grid_size} = grid.box_x+1 and
        mapp.box_index % {grid_size} = grid.box_y 
    union
    select grid.box_index as b_idx, grid.box_lat, grid.box_lon, mapp.index as m_idx
    from    mapp, grid
    where  
        mapp.box_index div {grid_size} = grid.box_x and
        mapp.box_index % {grid_size} = grid.box_y+1
    union
    select grid.box_index as b_idx, grid.box_lat, grid.box_lon, mapp.index as m_idx
    from    mapp, grid
    where  
        mapp.box_index div {grid_size} = grid.box_x-1 and
        mapp.box_index % {grid_size} = grid.box_y-1 
    union
    select grid.box_index as b_idx, grid.box_lat, grid.box_lon, mapp.index as m_idx
    from    mapp, grid
    where  
        mapp.box_index div {grid_size} = grid.box_x+1 and
        mapp.box_index % {grid_size} = grid.box_y+1 
    union
    select grid.box_index as b_idx, grid.box_lat, grid.box_lon, mapp.index as m_idx
    from    mapp, grid
    where  
        mapp.box_index div {grid_size} = grid.box_x+1 and
        mapp.box_index % {grid_size} = grid.box_y-1 
    union
    select grid.box_index as b_idx, grid.box_lat, grid.box_lon, mapp.index as m_idx
    from    mapp, grid
    where  
        mapp.box_index div {grid_size} = grid.box_x-1 and
        mapp.box_index % {grid_size} = grid.box_y+1
    ) as neis
where orig.index = neis.m_idx
    and orig.name='{self.method.feature}'
    and orig.aggregate_func='{self.method.aggre_func}'
    and orig.grid_length={self.origin_data_res}
    and ((a.box_x div {resolution})*{grid_size} + a.box_y div {resolution}) = neis.b_idx
group by a.box_index;"""

    def get_insert_neigh_sql(self, resolution=_resolution, grid_size=_grid_size):
        return f"""insert into tmpnei (idx, neigh)
select box_index, nei
from ({self.get_neigh_fast_sql(resolution=resolution, grid_size=grid_size)[:-1]})
where nei is not null;"""

    # def get_insert_result_sql(thresh=_thresh, model_name=_model_name, resolution=_resolution, grid_size=_grid_size, feature=_feature, method_id=_method_id, param=_param):
    #     return f"""insert into result (box_index, box_lat, box_lon, value, method_id)
    # select box_index, box_lat, box_lon, val, {method_id}
    # from ({get_fast_interpol_sql(thresh, model_name, resolution=resolution, grid_size=grid_size, feature=feature, param=param)[:-1]})
    # where val is not null;"""

    def calc_interpol_params(self, dryrun=True):
        sql = f"""select lat, lon, value
from meanorigindata
where name='{self.method.feature}' and datasetname='{self.method.dataset}' 
and aggregate_func='{self.method.aggre_func}' and grid_length=1;"""

        if dryrun:
            print(f"Calc {self.method.method_name} interpolation params for {self.method.feature}:\n{sql}")
            self.model_param = [0.1, 0.2]
        elif self.method.method_name == 'krige':
            with o.execute_sql(sql).open_reader(tunnel=True) as reader:
                df = reader.to_pandas(n_process=n_process)

                ok = OrdinaryKriging(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2],
                                     variogram_model=self.method.params['variogram_model'])
                self.model_param = ok.variogram_model_parameters
        print(self.model_param)
        print()
        # return self.param

    def search_neigh_and_calc(self, res=_resolution, dryrun=True):
        iters = 1
        total_time = 0
        tmp_count = 0
        done = False
        grid_size = self.grid_dict.get(res)

        while not done:
            time_st = time.time()
            print(f"============== Iter #{iters}: res={res} ===============")
            # delete tmp
            if not dryrun:
                o.execute_sql(f"delete from temp_{self.method.func_name} where method_id={self.method.m_id};")

            # 1. calc and store values into temp table
            sql = self.get_prefix_sql(resolution=res) + self.get_insert_temp_sql(grid_size=grid_size, resolution=res)
            # print('- Query: ')
            # print(sql)
            if not dryrun:
                o.execute_sql(sql)
            # print()

            # 2. store results
            to_update = self.get_update_sql()
            # print('- Running: \n', to_update)
            if not dryrun:
                o.execute_sql(to_update)

            elapsed_time = time.time() - time_st
            total_time += elapsed_time
            print('--- Time used: ', elapsed_time)

            # 3. check if all box calculated
            to_check = self.get_query_sql()
            # print('- Running: \n', to_check)
            if not dryrun:
                with o.execute_sql(to_check).open_reader(tunnel=True) as reader:
                    tmp_count = reader[0][0]
                if tmp_count == 0:
                    done = True
                else:
                    res *= 2
            else:
                res *= 2
                done = True

            if res > self.grid_default_size or res not in self.grid_dict.keys():
                done = True

            grid_size = self.grid_dict.get(res)
            print(f">>> Iter #{iters}: {tmp_count} rows remaining. \n")
            iters += 1

        # clear tmp
        if not dryrun:
            o.execute_sql(f"delete from temp_{self.method.func_name} where method_id={self.method.m_id};")
        print(f"{self.method.m_id}-- {self.method.feature} done. Total time: {total_time}")
        print("===================================================\n")
        return total_time

    def search_neigh_only(self, res=_resolution, dryrun=False):
        iters = 1
        total_time = 0
        tmp_count = 0
        done = False
        grid_size = self.grid_dict.get(res)
        self.radius = 20
        # clear old neighbour results
        if not dryrun:
            o.execute_sql("delete from tmpnei;")
            o.execute_sql(f"""insert into neighbours (idx)
select box_index
from insertdata
where datasetname = '{self.method.dataset}' and grid_length={self.default_res};""")

        while not done:
            time_st = time.time()
            print(f"============== Iter #{iters}: res={res} ===============")

            # 1. calc and store values into neighbours table
            sql = self.get_neigh_prefix_sql(resolution=res) + self.get_insert_neigh_sql(grid_size=grid_size,
                                                                                        resolution=res)
            print('- Query: \n', sql)
            if not dryrun:
                o.execute_sql(sql)
            print()

            elapsed_time = time.time() - time_st
            total_time += elapsed_time
            print('--- Time used: ', elapsed_time)

            sql = """update neighbours
set neigh = tmp.neigh
from  (select distinct idx, neigh from tmpnei) as tmp
where neighbours.idx = tmp.idx
    and neighbours.neigh is null;"""
            o.execute_sql(sql)

            # 3. check if all box calculated
            to_check = f"""select count(idx) from neighbours where neigh is null;"""
            # print('- Running: \n', to_check)
            if not dryrun:
                with o.execute_sql(to_check).open_reader(tunnel=True) as reader:
                    tmp_count = reader[0][0]
                if tmp_count == 0:
                    done = True
                else:
                    res *= 2
            else:
                res *= 2
                done = True

            if res > self.grid_default_size or res not in self.grid_dict.keys():
                done = True

            grid_size = self.grid_dict.get(res)
            print(f">>> Iter #{iters}: {tmp_count} rows remaining. \n")
            iters += 1

        print(f"{self.method.feature} done. Total time: {total_time}")
        print("===================================================\n")

        sql = self.get_neigh_prefix_sql(res) + self.get_insert_neigh_sql(res, grid_size)
        print(sql)

        with o.execute_sql(sql).open_reader(tunnel=True) as reader:
            df = reader.to_pandas(n_process=n_process)
            df.to_csv(f'neigh-{self.method.feature}-{res}.csv', encoding='utf-8', index=False)
        print(df.shape)
        print('Done')

    # ======================  main  =============================
    def run(self, dryrun=True):
        # 1. create result table and write 
        sql = self.get_create_result_table_sql()
        print(f'\n>>> Running {self.method.feature} -- {self.method.m_id}:')
        if not dryrun:
            o.execute_sql(sql)

        # 2. update boxes with existing values
        if not dryrun:
            # sql = self.get_update_existing_val_sql()
            # o.execute_sql(sql)
            to_check = self.get_query_sql()
            with o.execute_sql(to_check).open_reader(tunnel=True) as reader:
                count = reader[0][0]
            print(f"--- {count} cells to be calculated.")

        # 3. calculate global parameters for interpolation method
        if self.method.method_name == 'krige':
            self.calc_interpol_params(dryrun=dryrun)
            if self.model_param is None and not dryrun:
                raise ("ERROR: calc params failed")
        else:
            self.model_param = [0, 0]

        # 4. do it
        tot_time = self.search_neigh_and_calc(res=_search_starting_res, dryrun=dryrun)
        return tot_time


if __name__ == '__main__':
    all_feats = ['Au', 'Na2O', 'U', 'Zn', 'Mn', 'Mo', '87Rb/86Sr(m)', 'P', 'CaO', 'Eu', 'Hf', '87Sr/86Sr(m)',
                 'TDM2 (Ma)', 'Te', 'MnO', 'H2O-', 'Nd', 'Br', 'FeOT', 'Cd', 'Er', 'Al2O3', '207Pb/204Pb(m)',
                 'CL', 'Sr(m)', '147Sm/144Nd(m)', 'As', 'La', 'Hg', 'Gd', 'Cu', 'Pt', 'Rb', 'εNd(t)', 'FeO',
                 '208Pb/204Pb(m)', 'Bi', 'Fe2O3', 'Pd', '143Nd/144Nd(m)', 'Th', 'Pr', 'Sr', 'Zr', 'P2O5', 'MgO',
                 'Rb(m)', 'F', 'Ba', 'Ce', 'Cr', 'Recalcul TDM2-Nd (Ma)', 'Ir', 'S', 'CO2', 'Sb', 'Ho',
                 'Recalcul εNd(t)', 'f(Sm/Nd)', 'TDM1 (Ma)', 'Rb/Sr', '206Pb/204Pb(m)', 'Cs', 'Be', 'Recalcul εNd(0)',
                 'Ag', 'Ga', 'In', 'Y', 'Se', 'Ta', 'LOI', 'W', 'K2O', 'Fe2O3T', 'Pb', 'Tm', 'Nb', 'Nd(m)',
                 'Ti', 'Sm(m)', 'K2O/Na2O', 'SiO2', 'Ge', 'δ18O', 'Co', 'Tb', 'Li', 'TiO2', '87Sr/86Sr(i)',
                 'log10(Rb/Sr)', 'Tl', 'Sm', 'Recalcul TDM1-Nd(Ma)', 'Sc', 'V', 'Lu', 'Dy', 'Sn', 'Ni']
    all_feats = ['SiO2']

    # 配置插值方法和参数。
    method_id = 100  # method.insert_to_insertMethod_if_not_exists()中会生成一个新的method_id。
    method_name = 'lpi'
    # 三个主要超参数
    method_params = {'degree': 2} # 默认为2
    thresh = 16 # 5、16
    resolution = 4 # 1, 2, 4

    failed = []
    print('>>> All features:')
    print(all_feats)
    print(env_settings)

    time_records = pd.DataFrame(columns=['Method_id', 'Feature', 'Elapsed_time'])
    result_filename = f'{method_name}_degree={method_params["degree"]}_thresh={thresh}' + \
                      f'_res={resolution}_featnum={len(all_feats)}.csv'
    uniq_fn = 0
    while os.path.exists(os.path.join(_save_path, result_filename)):
        result_filename = f'{method_name}_degree={method_params["degree"]}_thresh={thresh}' + \
                          f'_res={resolution}_featnum={len(all_feats)}_{uniq_fn}.csv'
        uniq_fn += 1

    # for idx in run_features.keys():
    for feat in all_feats:
        method = Method(m_id=method_id, method_name=method_name, params=method_params, thresh=thresh,
                        dataset=_dataset, resolution=resolution, feature=feat, aggr_func=_aggre_func)
        if method.func_name is None:
            raise ("ERROR: Method not implemented or specified yet. Check dict: _method_func_map.")
        m_id = method.insert_to_insertMethod_if_not_exists()

        yjy = Interpolation(method, default_res=resolution)
        try:
            elapsed_time = yjy.run(dryrun=False)
            time_records.loc[len(time_records)] = [m_id, feat, elapsed_time]
            # yjy.search_neigh_only(1)
        except Exception as e:
            print(f"Error: {e}")
            failed.append(feat)

    time_records.to_csv(os.path.join(_save_path, result_filename))
    # print(time_records.head())
    print(f"All {len(all_feats)} features done. Saved to {result_filename}")
    if len(failed):
        print(f"Failed {len(failed)}/{len(all_feats)}:")
        print(failed)
