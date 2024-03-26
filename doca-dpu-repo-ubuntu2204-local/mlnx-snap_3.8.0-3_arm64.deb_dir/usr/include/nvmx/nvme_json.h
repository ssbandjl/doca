#ifndef _NVME_JSON_H
#define _NVME_JSON_H

#include "json.h"
#include "nvme.h"
#include "nvme_emu.h"

void nvme_json_get_quirks(nvme_config_t *config, enum nvme_quirks *quirks);
int nvme_json_get_global_id_list(nvme_config_t *config, const char *ns_path,
                                 nvme_id_ns_descriptor_t id_descs[]);

#endif
