import configparser
import os

config = configparser.ConfigParser(allow_no_value=True)
config.read('/etc/mellanox/hbn.conf')
array_reps = config.items('HBN_REPS')
array_uplinks = config.items('HBN_UPLINKS')
array_dpu_sfs = config.items('HBN_DPU_SFS')

val=""
for i in array_uplinks:
    val += i[0] + ','
f = open("/tmp/.HBN_UPLINKS", "w")
f.write(val[:-1])
f.close()
val=""
for i in array_reps:
    val += i[0] + ','
f = open("/tmp/.HBN_REPS", "w")
f.write(val[:-1])
f.close()
val=""
for i in array_dpu_sfs:
    val += i[0] + ','
f = open("/tmp/.HBN_DPU_SFS", "w")
f.write(val[:-1])
f.close()
val=""

