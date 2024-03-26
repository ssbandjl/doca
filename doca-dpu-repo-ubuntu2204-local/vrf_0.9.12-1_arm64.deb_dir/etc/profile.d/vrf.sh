VRF=$(ip vrf identify)
[ -n "${VRF}" ] && VRF=":${VRF}"
export VRF

PS1='\u@\h${VRF}:\w\$ '
