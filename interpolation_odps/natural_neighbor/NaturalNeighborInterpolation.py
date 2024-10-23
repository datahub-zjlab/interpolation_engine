import time

from odps import options

from interpolation_odps.yjy_interpolation import Method
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



class NaturalNeighborInterpolation:
    def __init__(self, method):
        self.method = method

    def get_interpolation_sql(self):
        sql = f"""
            with ins as (
              select box_index, box_lat, box_lon from insertdata where datasetname="{self.method.dataset}" and grid_length={self.method.resolution}
            )
            insert into result_naturalneighbor(box_index, box_lat, box_lon, value, method_id)
            select ins.box_index, ins.box_lat, ins.box_lon, naturalNeighborInterpolate(nnm.neighbors, nnm.neighbors_tri_simplices, nnm.neighbors_tri_points, nnm.neighbors_point_value, nnm.neighbors_circumcenters, ins.box_lat, ins.box_lon), {self.method.m_id}
            from naturalneighbor_map nnm
            right join ins on ins.box_index=nnm.box_index
            """

        return sql
    def run(self):
        sql = self.get_interpolation_sql()
        start = time.time()
        o.execute_sql(sql)
        end = time.time()
        print(f'cal time: {end - start}')
        return end - start

if __name__ == '__main__':

    m = Method(method_name='naturalneighbor', params = {}, thresh = 0, dataset = "纽芬兰", resolution=1, feature="SiO2", aggr_func="mean")
    m_id = m.insert_to_insertMethod_if_not_exists()
    interpolation = NaturalNeighborInterpolation(m)
    interpolation.run()

