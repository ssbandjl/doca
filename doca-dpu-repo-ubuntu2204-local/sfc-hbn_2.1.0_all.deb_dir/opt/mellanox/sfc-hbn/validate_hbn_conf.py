import os
import re

def parse_validate_uplink(uplinks_str):
  valid=[['p0'], ['p0', 'p1']]
  uplinks = uplinks_str.split(",")
  if  uplinks not in valid:
    print("Invalid uplinks: %s"%uplinks_str)
    exit(1)
  return uplinks

def parse_validate_dpu_interfaces(dpu_ifs_str):
  valid=[['pf0dpu0_sf','pf0dpu1_sf']]
  dpu_ifs = dpu_ifs_str.split(",")
  if  dpu_ifs not in valid:
    print("Invalid dpu interfaces: %s"%dpu_ifs_str)
    exit(1)
  return dpu_ifs

def parse_range(range_obj):
    res = []
    pre = os.path.commonprefix(range_obj)
    first = int(range_obj[0][len(pre):])
    last = int(range_obj[1][len(pre):])
    for i in range(first, last+1, 1):
       res.append(pre+str(i))
    return res

def _validate_host_interfaces(host_ifs_str):
    pattern = re.compile("^(pf[0-1](hpf|vf[0-9]+))$")
    is_valid_interface = lambda iface: pattern.match(iface)
    x = [hs.split('-') for hs in host_ifs_str.split(',')]
    for item in x:
      for i in item:
        if is_valid_interface(i):
           print("%s is a valid interface"%i)
        else:
           print("%s is a invalid interface"%i)
           exit(1)
    return x

def parse_validate_host_interfaces(host_ifs_str):
  valid=[]
  res = []
  host_ifs = _validate_host_interfaces(host_ifs_str)
  print(host_ifs)
  #  res = [item for sublist in host_ifs for item in sublist]
  for item in host_ifs:
    if len(item) == 2: # Range object
      res += parse_range(item)  # Add the entries to the list
    else:
      res.append(item[0])  # We have only a single item.
  return res

env_uplinks=os.getenv('HBN_SFC_UPLINKS', default="p0,p1")
env_host_interfaces=os.getenv('HBN_SFC_UPLINKS', default="pf0hpf,pf1hpf,pf0vf0-pf0vf13")
env_dpu_interfaces=os.getenv('HBN_SFC_DPU_SFS', default="pf0dpu0_sf,pf0dpu1_sf")

parse_validate_uplink(env_uplinks)
parse_validate_dpu_interfaces(env_dpu_interfaces)
parse_validate_host_interfaces(env_host_interfaces)
