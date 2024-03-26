import configparser
import os

array_p0 = os.environ.get('ARR_REPS_P0').split(' ')
array_p1 = os.environ.get('ARR_REPS_P1').split(' ')
array_uplinks = os.environ.get('ARR_UPLINKS').split(' ')
array_dpu_sfs = os.environ.get('ARR_DPU_SFS_ORIG').split(' ')
config = configparser.ConfigParser(allow_no_value=True)
config['HBN_UPLINKS'] = {}
for i in array_uplinks:
    config.set('HBN_UPLINKS', i, None)
config['HBN_REPS'] = {}
for i in array_p0:
    config.set('HBN_REPS', i, None)
for i in array_p1:
    config.set('HBN_REPS', i, None)
config['HBN_DPU_SFS'] = {}
for i in array_dpu_sfs:
    config.set('HBN_DPU_SFS', i, None)
with open('/etc/mellanox/hbn.conf', 'w') as configfile:
    config.write(configfile)