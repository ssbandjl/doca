#ifndef HCOL_DTE_H
#define HCOL_DTE_H
#include <stdint.h>
#include <float.h>
#include <stddef.h>



struct dte_data_representation_t;
struct dte_basic_unit_t;
struct dte_struct_t;
struct dte_type_vector;
struct dte_type_struct;

#ifdef WORDS_BIGENDIAN /*This comes from OPAL*/
#define HCOL_DTE_WORDS_BIGENDIAN 1
#endif

#define HCOL_FLOATING_POINT_TYPE 1
#define HCOL_FIXED_SIZE_TYPE 0

typedef enum hcoll_dte_dtype_id{
    HCOL_DTE_ZERO,
    HCOL_DTE_BYTE,
    HCOL_DTE_INT16,
    HCOL_DTE_INT32,
    HCOL_DTE_INT64,
    HCOL_DTE_INT128,
    HCOL_DTE_UBYTE,
    HCOL_DTE_UINT16,
    HCOL_DTE_UINT32,
    HCOL_DTE_UINT64,
    HCOL_DTE_UINT128,
    HCOL_DTE_FLOAT32,
    HCOL_DTE_FLOAT64,
    HCOL_DTE_FLOAT96,
    HCOL_DTE_FLOAT128,
    HCOL_DTE_FLOAT32_COMPLEX,
    HCOL_DTE_FLOAT64_COMPLEX,
    HCOL_DTE_FLOAT128_COMPLEX,
    HCOL_DTE_FLOAT_INT,
    HCOL_DTE_DOUBLE_INT,
    HCOL_DTE_LONG_INT,
    HCOL_DTE_2INT,
    HCOL_DTE_SHORT_INT,
    HCOL_DTE_LONG_DOUBLE_INT,
    HCOL_DTE_2INT64,
    HCOL_DTE_2FLOAT32,
    HCOL_DTE_2FLOAT64,
    HCOL_DTE_LB,
    HCOL_DTE_UB,
    HCOL_DTE_BOOL,
    HCOL_DTE_WCHAR,
    HCOL_DTE_MPI,
    HCOL_DTE_FLOAT16,
    HCOL_DTE_MAX_PREDEFINED
}hcoll_dte_id_t;

#ifndef HCOL_DTE_WORDS_BIGENDIAN
struct dte_data_rep_t {
    union{
        struct {
            uint8_t in_line :1 ;
            uint8_t type :2;
            uint8_t contiguity :1;
            uint8_t n_types :2;
            uint8_t padding : 1;
            uint8_t is_signed : 1;
            uint8_t packed_size ;
            uint8_t mantisa_size ;
            uint16_t data_extent ;
            uint16_t offset_from_base ;
            uint8_t alignment ;
        } in_line ;
        uint64_t pointer_to_handle ;
        uint8_t raw [8];
    } data_handle ;
};
#else
struct dte_data_rep_t {
    union{
        struct {
            uint8_t alignment ;
            uint16_t offset_from_base ;
            uint16_t data_extent ;
            uint8_t mantisa_size ;
            uint8_t packed_size ;
            uint8_t is_signed : 1;
            uint8_t padding : 2;
            uint8_t n_types :2;
            uint8_t contiguity :1;
            uint8_t type :2;
            uint8_t in_line :1 ;
        } in_line ;
        uint64_t pointer_to_handle ;
        uint8_t raw [8];
    } data_handle ;
};
#endif
typedef struct dte_data_rep_t dte_data_rep_t;
struct dte_ptr;
struct dte_data_representation_t {
    union {
        struct dte_data_rep_t in_line_rep ;
        struct dte_ptr *ptr;
        struct dte_data_rep_gen_t *general_rep; /*For backwards compat - don't use it!! */
    } rep;
    short id;
};
typedef struct dte_data_representation_t dte_data_representation_t;
typedef dte_data_representation_t hcoll_datatype_t;


int hcoll_create_mpi_type(void *mpi_type, hcoll_datatype_t *hcoll_type);
/**
 * Destroys hcoll datatype
 *
 * @return HCOLL_SUCCESS, or HCOLL_ERROR on failure
 */
int hcoll_dt_destroy(hcoll_datatype_t type);

extern dte_data_representation_t byte_dte;
extern dte_data_representation_t integer16_dte;
extern dte_data_representation_t integer32_dte;
extern dte_data_representation_t integer64_dte;
extern dte_data_representation_t integer128_dte;

extern dte_data_representation_t u_byte_dte;
extern dte_data_representation_t u_integer16_dte;
extern dte_data_representation_t u_integer32_dte;
extern dte_data_representation_t u_integer64_dte;
extern dte_data_representation_t u_integer128_dte;

extern dte_data_representation_t zero_dte;

extern dte_data_representation_t float16_dte;
extern dte_data_representation_t float32_dte;
extern dte_data_representation_t float64_dte;
extern dte_data_representation_t float96_dte;
extern dte_data_representation_t float128_dte;
extern dte_data_representation_t float32_complex_dte;
extern dte_data_representation_t float64_complex_dte;
extern dte_data_representation_t float128_complex_dte;

extern dte_data_representation_t hcol_dte_float_int;
extern dte_data_representation_t hcol_dte_double_int;
extern dte_data_representation_t hcol_dte_long_int;
extern dte_data_representation_t hcol_dte_2int;
extern dte_data_representation_t hcol_dte_2int64;
extern dte_data_representation_t hcol_dte_2float32;
extern dte_data_representation_t hcol_dte_2float64;
extern dte_data_representation_t hcol_dte_short_int;
extern dte_data_representation_t hcol_dte_long_double_int;

extern dte_data_representation_t hcol_dte_lb;
extern dte_data_representation_t hcol_dte_ub;
extern dte_data_representation_t hcol_dte_bool;
extern dte_data_representation_t hcol_dte_wchar;

#define HCOL_DTE_PTR(type) (type.rep.in_line_rep.data_handle.pointer_to_handle)
#define HCOL_DTE_IS_INLINE(type) type.rep.in_line_rep.data_handle.in_line.in_line
#define HCOL_DTE_SIZE(type) (HCOL_DTE_IS_INLINE(type) ? type.rep.in_line_rep.data_handle.in_line.packed_size/8 : -1)
#define HCOL_DTE_EXTENT(type) (HCOL_DTE_IS_INLINE(type) ? type.rep.in_line_rep.data_handle.in_line.data_extent/8 : -1)


#define HCOL_DTE_IS_ZERO(type) ((HCOL_DTE_IS_INLINE(type) && type.id == HCOL_DTE_ZERO) ? 1: 0)
#define HCOL_DTE_IS_COMPLEX(type) ((!HCOL_DTE_IS_INLINE(type) && type.id == HCOL_DTE_ZERO) ? 1: 0)
#define HCOL_DTE_IS_SIGNED(type) ((HCOL_DTE_IS_INLINE(type) && type.rep.in_line_rep.data_handle.in_line.is_signed))
#define HCOL_DTE_IS_CONTIG(type) ((HCOL_DTE_IS_INLINE(type) && type.rep.in_line_rep.data_handle.in_line.contiguity))
#define HCOL_DTE_IS_FIXED_SIZE(dtype) ((HCOL_DTE_IS_INLINE(dtype) && (HCOL_FIXED_SIZE_TYPE == dtype.rep.in_line_rep.data_handle.in_line.type)))
#define HCOL_DTE_PREDEFINED_ID(dtype) (dtype.id)


/* Predefined dtypes */
#define DTE_BYTE  (byte_dte)
#define DTE_INT16 (integer16_dte)
#define DTE_INT32 (integer32_dte)
#define DTE_INT64 (integer64_dte)
#define DTE_INT128 (integer128_dte)


#define DTE_UBYTE  (u_byte_dte)
#define DTE_UINT16 (u_integer16_dte)
#define DTE_UINT32 (u_integer32_dte)
#define DTE_UINT64 (u_integer64_dte)
#define DTE_UINT128 (u_integer128_dte)


#define DTE_ZERO  (zero_dte)

#define DTE_FLOAT16 (float16_dte)
#define DTE_FLOAT32 (float32_dte)
#define DTE_FLOAT64 (float64_dte)
#define DTE_FLOAT96 (float96_dte)
#define DTE_FLOAT128 (float128_dte)

#define DTE_FLOAT32_COMPLEX (float32_complex_dte)
#define DTE_FLOAT64_COMPLEX (float64_complex_dte)
#define DTE_FLOAT128_COMPLEX (float128_complex_dte)

#define DTE_FLOAT_INT         (hcol_dte_float_int)
#define DTE_DOUBLE_INT        (hcol_dte_double_int)
#define DTE_LONG_INT          (hcol_dte_long_int)
#define DTE_2INT              (hcol_dte_2int)
#define DTE_2INT64            (hcol_dte_2int64)
#define DTE_2FLOAT32          (hcol_dte_2float32)
#define DTE_2FLOAT64          (hcol_dte_2float64)
#define DTE_SHORT_INT         (hcol_dte_short_int)
#define DTE_LONG_DOUBLE_INT   (hcol_dte_long_double_int)
#define DTE_LB                (hcol_dte_lb)
#define DTE_UB                (hcol_dte_ub)
#define DTE_BOOL              (hcol_dte_bool)
#define DTE_WCHAR             (hcol_dte_wchar)

typedef enum hcoll_dte_op_id{
    HCOL_DTE_OP_NULL,
    HCOL_DTE_OP_MAX,
    HCOL_DTE_OP_MIN,
    HCOL_DTE_OP_SUM,
    HCOL_DTE_OP_PROD,
    HCOL_DTE_OP_LAND,
    HCOL_DTE_OP_BAND,
    HCOL_DTE_OP_LOR,
    HCOL_DTE_OP_BOR,
    HCOL_DTE_OP_LXOR,
    HCOL_DTE_OP_BXOR,
    HCOL_DTE_OP_MAXLOC,
    HCOL_DTE_OP_MINLOC,
    HCOL_DTE_OP_REPLACE,
    HCOL_DTE_OP_NUM_OF_TYPES
}hcoll_dte_op_id_t;

typedef struct hcoll_dte_op{
    int id;
}hcoll_dte_op_t;

extern hcoll_dte_op_t hcoll_dte_op_null;
extern hcoll_dte_op_t hcoll_dte_op_max;
extern hcoll_dte_op_t hcoll_dte_op_min;
extern hcoll_dte_op_t hcoll_dte_op_sum;
extern hcoll_dte_op_t hcoll_dte_op_prod;
extern hcoll_dte_op_t hcoll_dte_op_land;
extern hcoll_dte_op_t hcoll_dte_op_band;
extern hcoll_dte_op_t hcoll_dte_op_lor;
extern hcoll_dte_op_t hcoll_dte_op_bor;
extern hcoll_dte_op_t hcoll_dte_op_lxor;
extern hcoll_dte_op_t hcoll_dte_op_bxor;


#define HCOL_OP_NULL &hcoll_dte_op_null
#define HCOL_OP_MAX  &hcoll_dte_op_max
#define HCOL_OP_MIN  &hcoll_dte_op_min
#define HCOL_OP_SUM  &hcoll_dte_op_sum
#define HCOL_OP_PROD &hcoll_dte_op_prod
#define HCOL_OP_LAND &hcoll_dte_op_land
#define HCOL_OP_BAND &hcoll_dte_op_band
#define HCOL_OP_LOR  &hcoll_dte_op_lor
#define HCOL_OP_BOR  &hcoll_dte_op_bor
#define HCOL_OP_LXOR &hcoll_dte_op_lxor
#define HCOL_OP_BXOR &hcoll_dte_op_bxor






static char dte_op_names[HCOL_DTE_OP_NUM_OF_TYPES][20] = {"OP_NULL",
                                                         "OP_MAX",
                                                         "OP_MIN",
                                                         "OP_SUM",
                                                         "OP_PROD",
                                                         "OP_LAND",
                                                         "OP_BAND",
                                                         "OP_LOR",
                                                         "OP_BOR",
                                                         "OP_LXOR",
                                                         "OP_BXOR",
                                                         "OP_MAXLOC",
                                                         "OP_MINLOC",
                                                         "OP_REPLACE"
                                                        };


static char dte_names[HCOL_DTE_MAX_PREDEFINED][32] = {"DTE_ZERO",
                                                      "DTE_BYTE",
                                                      "DTE_INT16",
                                                      "DTE_INT32",
                                                      "DTE_INT64",
                                                      "DTE_INT128",
                                                      "DTE_UBYTE",
                                                      "DTE_UINT16",
                                                      "DTE_UINT32",
                                                      "DTE_UINT64",
                                                      "DTE_UINT128",
                                                      "DTE_FLOAT32",
                                                      "DTE_FLOAT64",
                                                      "DTE_FLOAT128",
                                                      "DTE_FLOAT96",
                                                      "DTE_FLOAT32_COMPLEX",
                                                      "DTE_FLOAT64_COMPLEX",
                                                      "DTE_FLOAT128_COMPLEX",
                                                      "DTE_FLOAT_INT",
                                                      "DTE_DOUBLE_INT",
                                                      "DTE_LONG_INT",
                                                      "DTE_2INT",
                                                      "DTE_SHORT_INT",
                                                      "DTE_LONG_DOUBLE_INT",
                                                      "DTE_2INT64",
                                                      "DTE_2FLOAT32",
                                                      "DTE_2FLOAT64",
                                                      "DTE_LB",
                                                      "DTE_UB",
                                                      "DTE_BOOL",
                                                      "DTE_WCHAR",
                                                      "DTE_MPI"};

static inline char * hcoll_dte_dtype_name(dte_data_representation_t dtype){
    return dte_names[dtype.id];
}

static inline char * hcoll_dte_op_name(hcoll_dte_op_t *op){
    return dte_op_names[op->id];
}


/* ====================================================================================== *
   This is old unused definitions kept for backwards compatibility. Those are never used. */
enum representation_type {
    LAST
};
 
struct dte_data_rep_gen_t {
    union {
        struct dte_generalized_iovec_t * data ;
    } data_representation ;
    enum representation_type type ;
};
struct dte_generalized_iovec_t {
    uint64_t repeat_count ;
    void * base_buffer ;
    struct dte_struct_t * repeat;
};
struct dte_struct_t {
    uint32_t n_elements ;
    uint32_t stride ;
    struct dte_basic_unit_t * elements ;
};
struct dte_basic_unit_t {
    uint64_t packed_size ;
    uint64_t base_offset ;
};
/* ====================================================================================== */
#endif /* HCOL_DTE_H */
