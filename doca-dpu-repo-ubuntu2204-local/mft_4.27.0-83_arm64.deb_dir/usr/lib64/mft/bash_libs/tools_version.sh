MFT_VERSION_STR="mft 4.27.0-83"
TOOLS_BUILD_TIME="Jan 31 2024, 20:19:49"
TOOLS_GIT_SHA="N/A"

function print_version_string {
	tool_name=$1
	tool_version=$2

	ver=""
	if [ "$tool_vesion" == "" ]; then
		ver="$tool_name"
	else
		ver="$tool_name $tool_version, "
	fi
	ver="$ver, $MFT_VERSION_STR, built on $TOOLS_BUILD_TIME. Git SHA Hash: $TOOLS_GIT_SHA"
	echo "$ver"
}
