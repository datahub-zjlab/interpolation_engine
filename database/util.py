def trans_params(params):
    params = [(p, params[p]) for p in params.keys()]
    params.sort(key=lambda x: x[0])
    params_v = [str(i[1]) for i in params]
    parms_k = [str(i[0]) for i in params]
    params_str = ",".join(parms_k) + ':' + ','.join(params_v)
    return params_str

def get_method_func_map():
    return {
    'krige':'kriging', 
    'rbf':'rbf', 
    'nn':'nearest_neighbour', 
    'idw':'idw',
}

def remove_csv_index(fn):
    import pandas as pd
    d = pd.read_csv(fn+'.csv', index_col=0)
    print(d.head())
    print(d.shape)
    input('continue?')
    d.to_csv(fn+'-1'+'.csv', index=False)
    print('done')

if __name__ == '__main__':
    
    fn = 'all_method_params-1'
    remove_csv_index(fn)
