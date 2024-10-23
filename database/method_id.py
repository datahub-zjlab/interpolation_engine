from utils.odps_util import Odps

o = Odps().get_odps_instance()

class Method:
    def __init__(self, method_name='krige', params=None, thresh=5,
                 dataset='纽芬兰', resolution=1, feature='SiO2', aggregate_func = 'mean'):
        self.method_name = method_name
        self.params = params
        self.thresh = thresh
        self.dataset = dataset
        self.resolution = resolution
        self.feature = feature
        self.aggregate_func = aggregate_func

    @classmethod
    def trans_params(cls, params):
        params = [(p, params[p]) for p in params.keys()]
        params.sort(key=lambda x: x[0])
        params_v = [str(i[1]) for i in params]
        parms_k = [str(i[0]) for i in params]
        params_str = ",".join(parms_k) + ':' + ','.join(params_v)
        return params_str

def insert_to_insertMethod_if_not_exists(method, method_id = None):
    sql_str = 'select * from insertmethod;'
    with o.execute_sql(sql_str).open_reader(
            tunnel=True) as reader:
        pd_df = reader.to_pandas()
    max_id = pd_df['method_id'].max()
    pd_df.set_index(['insert_method', 'params', 'n_closest_points', 'datasetname', 'grid_length', 'name', 'aggregate_func'], inplace = True)
    insert_data = [method.method_name, Method.trans_params(method.params), method.thresh, method.dataset, method.resolution, method.feature, method.aggregate_func]
    if method_id != None:
        o.write_table('insertmethod', [[method_id] + insert_data])
    elif tuple(insert_data) in pd_df.index:
        method_id = None
    else:
        method_id = max_id + 1
        print('insert:' + str([[method_id] + insert_data]))
        o.write_table('insertmethod', [[method_id] + insert_data])
    return method_id

if __name__ == '__main__':
    m = Method(method_name='krige', params={"variogram_model": 'linear'}, resolution=2, dataset='纽芬兰', feature='Zn',
               thresh=5, aggregate_func='mean')
    insert_to_insertMethod_if_not_exists(m, method_id=905)