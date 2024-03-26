/**
 * \file
 * JSON parsing and encoding
 */

#ifndef NVME_JSON_H_
#define NVME_JSON_H_

#include "utf.h"

#ifdef __cplusplus
extern "C" {
#endif

enum nvme_json_val_type {
	NVME_JSON_VAL_INVALID,
    NVME_JSON_VAL_NULL,
    NVME_JSON_VAL_TRUE,
    NVME_JSON_VAL_FALSE,
    NVME_JSON_VAL_NUMBER,
    NVME_JSON_VAL_STRING,
    NVME_JSON_VAL_ARRAY_BEGIN,
    NVME_JSON_VAL_ARRAY_END,
    NVME_JSON_VAL_OBJECT_BEGIN,
    NVME_JSON_VAL_OBJECT_END,
    NVME_JSON_VAL_NAME,
};

struct nvme_json_val {
    /**
     * Pointer to the location of the value within the parsed JSON input.
     *
     * For nvme_json_VAL_STRING and nvme_json_VAL_NAME,
     *  this points to the beginning of the decoded UTF-8 string without quotes.
     *
     * For nvme_json_VAL_NUMBER, this points to the beginning of the number as represented in
     *  the original JSON (text representation, not converted to a numeric value).
     */
    void *start;

    /**
     * Length of value.
     *
     * For nvme_json_VAL_STRING, nvme_json_VAL_NUMBER, and nvme_json_VAL_NAME,
     *  this is the length in bytes of the value starting at \ref start.
     *
     * For nvme_json_VAL_ARRAY_BEGIN and nvme_json_VAL_OBJECT_BEGIN,
     *  this is the number of values contained within the array or object (including
     *  nested objects and arrays, but not including the _END value).  The array or object _END
     *  value can be found by advancing len values from the _BEGIN value.
     */
    uint32_t len;

    /**
     * Type of value.
     */
    enum nvme_json_val_type type;
};

#define NVME_JSON_MAX_OBJ_NAME 64

/**
 * Invalid JSON syntax.
 */
#define NVME_JSON_PARSE_INVALID            -1

/**
 * JSON was valid up to the end of the current buffer, but did not represent a complete JSON value.
 */
#define NVME_JSON_PARSE_INCOMPLETE        -2

#define NVME_JSON_PARSE_MAX_DEPTH_EXCEEDED    -3

/**
 * Decode JSON strings and names in place (modify the input buffer).
 */
#define NVME_JSON_PARSE_FLAG_DECODE_IN_PLACE    0x000000001

/**
 * Allow parsing of comments.
 *
 * Comments are not allowed by the JSON RFC, so this is not enabled by default.
 */
#define NVME_JSON_PARSE_FLAG_ALLOW_COMMENTS    0x000000002

/*
 * Parse JSON data.
 *
 * \param data Raw JSON data; must be encoded in UTF-8.
 * Note that the data may be modified to perform in-place string decoding.
 *
 * \param size Size of data in bytes.
 *
 * \param end If non-NULL, this will be filled a pointer to the byte just beyond the end
 * of the valid JSON.
 *
 * \return Number of values parsed, or negative on failure:
 * NVME_JSON_PARSE_INVALID if the provided data was not valid JSON, or
 * NVME_JSON_PARSE_INCOMPLETE if the provided data was not a complete JSON value.
 */
ssize_t nvme_json_parse(void *json, size_t size,
                        struct nvme_json_val *values,
                        size_t num_values,
                        void **end, uint32_t flags);

typedef int (*nvme_json_decode_fn)(const struct nvme_json_val *val, void *out);

struct nvme_json_object_decoder {
    const char *name;
    size_t offset;
    nvme_json_decode_fn decode_func;
    bool optional;
};

typedef struct {
    void *buf;
    struct nvme_json_val *values;
    int num_values;
} nvme_config_t;

typedef nvme_config_t nvme_json_t;

/**
 * Get length of a value in number of values.
 *
 * This can be used to skip over a value while interpreting parse results.
 *
 * For nvme_json_VAL_ARRAY_BEGIN and nvme_json_VAL_OBJECT_BEGIN,
 *  this returns the number of values contained within this value, plus the _BEGIN and _END values.
 *
 * For all other values, this returns 1.
 */
size_t nvme_json_val_len(const struct nvme_json_val *val);

/**
 * Compare JSON string with null terminated C string.
 *
 * \return true if strings are equal or false if not
 */
bool nvme_json_strequal(const struct nvme_json_val *val, const char *str);

/**
 * Equivalent of strdup() for JSON string values.
 *
 * If val is not representable as a C string (contains embedded '\0' characters),
 * returns NULL.
 *
 * Caller is responsible for passing the result to free() when it is no longer needed.
 */
char *nvme_json_strdup(const struct nvme_json_val *val);

int
nvme_json_decode_object(const struct nvme_json_val *values,
                        const struct nvme_json_object_decoder *decoders,
                        size_t num_decoders, void *out);

int
nvme_json_decode_array(const struct nvme_json_val *values,
                       nvme_json_decode_fn decode_func,
                       void *out, size_t max_size,
                       size_t *out_size, size_t stride);

int
nvme_json_decode_bool(const struct nvme_json_val *val, void *out);
int
nvme_json_decode_int16(const struct nvme_json_val *val, void *out);
int
nvme_json_decode_uint16(const struct nvme_json_val *val, void *out);
int
nvme_json_decode_int32(const struct nvme_json_val *val, void *out);
int
nvme_json_decode_uint32(const struct nvme_json_val *val, void *out);
int
nvme_json_decode_int64(const struct nvme_json_val *val, void *out);
int
nvme_json_decode_uint64(const struct nvme_json_val *val, void *out);
int
nvme_json_decode_string(const struct nvme_json_val *val, void *out);

nvme_json_t *nvme_str_to_json(char *str);
void nvme_json_free(nvme_config_t *json);
void nvme_json_config_close(nvme_config_t *config);
void print_json_error(FILE *pf, int rc, const char *filename);

int nvme_json_parse_config(const char *filename,
                           nvme_config_t *config);
int nvme_read_config_file(nvme_config_t *config,
                          const char *env,
                          const char *def);

#define NVME_JSON_OBJ_INVALID_ID (-1)
typedef int nvme_json_obj_id_t;

/**
 * Extract param value according to the path,
 * the path example: "Obj1.Obj2.Arr[3].param".
 *
 * \param out - a pointer of the next possible types:
 *
 *              uint*_t/int*_t   - positive/negative values in range between INT*_MIN...UINT*_MAX,
 *              int/unsigned int   error if the actual value is not in this range
 *              bool             - 0/1,
 *              string           - could be truncated if its size less than the actual string
 *
 * \param size - size of the *out in bytes
 *
 * \return zero if succeeded, negative otherwise
 */
int nvme_json_get_value(nvme_config_t *config,
                        const char *path,
                        void *out,
                        size_t size);

/**
 * Verify param int value according to the path
 * matches the expected value given.
 *
 * \return zero if equal, negative otherwise
 */
int nvme_json_verify_value(nvme_config_t *config, const char* path, int value);

/**
 * Verify param int value according to the path
 * is lower than or equals the expected value given.
 *
 * \return zero if lte, negative otherwise,
 *  -EINVAL in case if retrieved value is greater than exp_value param
 */
int nvme_json_verify_lte_value(nvme_config_t *config, const char* path,
                               int exp_value);

/**
 * Get obj reference according to the path, the path examples:
 *
 * "Obj1.Obj2.Arr[3]" - get reference to the third elem of Arr[]
 * "Obj1.Arr[3].Obj2" - get reference to the Obj2.
 *
 * \return obbj id if succeeded, NVME_JSON_OBJ_INVALID_ID otherwise
 */
nvme_json_obj_id_t nvme_json_get_obj(nvme_config_t *config,
                                     const char *path);

/**
 * Get param value if the obj specified by the given id.
 *
 * \return zero if succeeded, negative otherwise
 */
int nvme_json_get_obj_value(nvme_config_t *config,
                            nvme_json_obj_id_t id,
                            const char *param,
                            void *out,
                            size_t size);

/**
 * Get get the number of elems of the array - specified by the path,
 * the path examples: "Obj1.Obj2.Arr" - get the number of elems of Arr[]
 *
 * \return arr size if succeeded, negative otherwise
 */
int nvme_json_get_array_size(nvme_config_t *config,
                             const char *path);

#ifdef __cplusplus
}
#endif

#endif
