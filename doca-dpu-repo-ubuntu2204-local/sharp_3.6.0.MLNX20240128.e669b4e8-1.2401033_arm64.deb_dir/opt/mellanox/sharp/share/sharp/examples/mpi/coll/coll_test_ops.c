/**
 * Copyright (c) 2015-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include <coll_test.h>
#include <malloc.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>

#define MAX_NB_REQS                           (512)
#define SHARP_COLL_TEST_DEFAULT_HUGEPAGE_SIZE (2 * (1ull << 20))

#define COLL_SET_ROOT(_root, _coll_root_rank, _comm_size)                                                \
    do {                                                                                                 \
        root = (_coll_root_rank >= 0 && coll_root_rank < _comm_size) ? _coll_root_rank : _comm_size - 1; \
    } while (0);

/* sharp supported datatype info*/
struct sharp_test_data_types_t sharp_test_data_types[] = {{"UINT_32_BIT", SHARP_DTYPE_UNSIGNED, MPI_UNSIGNED, 4},
                                                          {"INT_32_BIT", SHARP_DTYPE_INT, MPI_INT, 4},
                                                          {"UINT_64_BIT", SHARP_DTYPE_UNSIGNED_LONG, MPI_UNSIGNED_LONG, 8},
                                                          {"INT_64_BIT", SHARP_DTYPE_LONG, MPI_LONG, 8},
                                                          {"FLOAT_32_BIT", SHARP_DTYPE_FLOAT, MPI_FLOAT, 4},
                                                          {"FLOAT_64_BIT", SHARP_DTYPE_DOUBLE, MPI_DOUBLE, 8},
                                                          {"UINT_16_BIT", SHARP_DTYPE_UNSIGNED_SHORT, MPI_UNSIGNED_SHORT, 2},
                                                          {"INT_16_BIT", SHARP_DTYPE_SHORT, MPI_SHORT, 2},
                                                          {"FLOAT_16_BIT", SHARP_DTYPE_FLOAT_SHORT, MPI_DATATYPE_NULL, 2},
                                                          {"BFLOAT_16_BIT", SHARP_DTYPE_BFLOAT16, MPI_DATATYPE_NULL, 2},
                                                          {"UINT_8_BIT", SHARP_DTYPE_UINT8, MPI_UNSIGNED_CHAR, 1},
                                                          {"INT_8_BIT", SHARP_DTYPE_INT8, MPI_CHAR, 1},
                                                          {"NULL", SHARP_DTYPE_NULL, 0, 0}};

MPI_Datatype sharp_test_mpi_min_max_datatype[SHARP_DTYPE_NULL][SHARP_DTYPE_NULL];

/*sharp supported reduce op info */
struct sharp_op_type_t sharp_test_ops[] = {{"MAX", SHARP_OP_MAX, MPI_MAX},
                                           {"MIN", SHARP_OP_MIN, MPI_MIN},
                                           {"SUM", SHARP_OP_SUM, MPI_SUM},
                                           //	{"PROD", SHARP_OP_PROD, MPI_PROD},
                                           {"LAND", SHARP_OP_LAND, MPI_LAND},
                                           {"BAND", SHARP_OP_BAND, MPI_BAND},
                                           {"LOR", SHARP_OP_LOR, MPI_LOR},
                                           {"BOR", SHARP_OP_BOR, MPI_BOR},
                                           {"LXOR", SHARP_OP_LXOR, MPI_LXOR},
                                           {"BXOR", SHARP_OP_BXOR, MPI_BXOR},
                                           {"MAXLOC", SHARP_OP_MAXLOC, MPI_MAXLOC},
                                           {"MINLOC", SHARP_OP_MINLOC, MPI_MINLOC},
                                           {"NOOP", SHARP_OP_NULL, 0}};

static inline struct sharp_test_data_types_t* sharp_coll_test_get_data_type(int datatype)
{
    int type;

    for (type = 0; type < SHARP_DTYPE_NULL; type++) {
        if (sharp_test_data_types[type].id == datatype) {
            break;
        }
    }

    return &sharp_test_data_types[type];
}

/* determine huge page size from system info */
size_t sharp_coll_test_get_huge_page_size()
{
    static size_t huge_page_size = 0;
    char buf[256];
    int size_kb;
    FILE* f;

    /* Cache the huge page size value */
    if (huge_page_size == 0) {
        f = fopen("/proc/meminfo", "r");
        if (f != NULL) {
            while (fgets(buf, sizeof(buf), f)) {
                if (sscanf(buf, "Hugepagesize:       %d kB", &size_kb) == 1) {
                    huge_page_size = size_kb * 1024;
                    break;
                }
            }
            fclose(f);
        }

        if (huge_page_size == 0) {
            huge_page_size = SHARP_COLL_TEST_DEFAULT_HUGEPAGE_SIZE;
        }
    }

    return huge_page_size;
}

static void* allocate_huge_page_memory(size_t size, int* shmid)
{
    void* ptr;
    int align_size = sharp_coll_test_get_huge_page_size();

    /* round to hugepage size */
    size = ((size - 1) / align_size) + 1;
    size = size * align_size;
    *shmid = shmget(IPC_PRIVATE, size, SHM_HUGETLB | IPC_CREAT | 0666);
    if (*shmid < 0) {
        *shmid = 0;
        return NULL;
    }

    ptr = shmat(*shmid, NULL, 0);
    if (!ptr) {
        *shmid = 0;
        return NULL;
    }

    if (shmctl(*shmid, IPC_RMID, 0) != 0) {
        shmdt(ptr);
        *shmid = 0;
        ptr = NULL;
        fprintf(stderr, "ERROR: shmctl(IPC_RMID, shmid=%d) failed: %m\n", *shmid);
    }

    return ptr;
}

static void* allocate_memory(size_t size, enum sharp_data_memory_type mem_type, int is_src_mem)
{
    void* ptr = NULL;
    int ret;

    if (mem_type == SHARP_MEM_TYPE_HOST) {
        if (sharp_conf.host_allocator_type == USE_HUGETLB) {
            ptr = allocate_huge_page_memory(size, (is_src_mem) ? &coll_sharp_component.src_shmid : &coll_sharp_component.dst_shmid);
            if (!ptr && sharp_conf.host_allocator_type == USE_HUGETLB) {
                fprintf(stderr,
                        "ERROR:shmget(size=%zu) for HUGETLB returned unexpected error: %m. "
                        "Please check shared memory limits by 'ipcs -l'.\n",
                        size);
                return NULL;
            }
        }

        if (ptr == NULL) {
            if (posix_memalign(&ptr, (2 * 1024 * 1024), size) != 0) {
                fprintf(stderr, "Failed to allocate host memory\n");
                return NULL;
            }
        }
        if (sharp_conf.host_allocator_type == USE_HUGETHP) {
            ret = madvise(ptr, size, MADV_HUGEPAGE);
            if (ret) {
                fprintf(stderr, "ERROR: Failed to set alloation to MADV_HUGEPAGE. ptr:%p size:%ld ret:%m\n", ptr, size);
                free(ptr);
                return NULL;
            }
        }
    }
#if HAVE_CUDA
    else if (mem_type == SHARP_MEM_TYPE_CUDA)
    {
        cudaError_t cerr;

        cerr = cudaMalloc(&ptr, size);
        if (cerr != cudaSuccess) {
            fprintf(stderr, "failed to allocate cuda memory size:%ld\n", size);
            return NULL;
        }
    }
#endif

    return ptr;
}

static void free_huge_page_mem(void* ptr)
{
    int ret;

    ret = shmdt(ptr);
    if (ret) {
        fprintf(stderr, "Unable to detach shared memory segment at %p: %m\n", ptr);
    }
}

static void free_memory(void* ptr, enum sharp_data_memory_type mem_type, int is_src_mem)
{
    if (mem_type == SHARP_MEM_TYPE_HOST) {
        if (ptr) {
            if (is_src_mem) {
                if (coll_sharp_component.src_shmid) {
                    free_huge_page_mem(ptr);
                    coll_sharp_component.src_shmid = 0;
                } else {
                    free(ptr);
                }
            } else {
                if (coll_sharp_component.dst_shmid) {
                    free_huge_page_mem(ptr);
                    coll_sharp_component.dst_shmid = 0;
                } else {
                    free(ptr);
                }
            }
        }
    }
#if HAVE_CUDA
    else if (mem_type == SHARP_MEM_TYPE_CUDA)
    {
        cudaFree(ptr);
    }
#endif
}

static int is_op_min_max_loc(int op_id)
{
    return (op_id == SHARP_OP_MINLOC) || (op_id == SHARP_OP_MAXLOC);
}

static void test_data_template_init(struct data_template* template, long value)
{
    template->unsigned_int_val = (unsigned int)value;
    template->int_val = (int)value;
    template->unsigned_long_val = (unsigned long)value;
    template->long_val = (long)value;
    template->float_val = (float)value;
    template->double_val = (double)value;
    template->unsigned_short_val = (unsigned short)value;
    template->short_val = (short)value;
    template->uint8_val = (unsigned char)value;
    template->int8_val = (char)value;
    template->float_short_val = (short)value;

    /*use rank as tag for min_loc/max_loc ops */

    template->unsigned_int_loc_tag_val = (unsigned int)value - 1;
    template->int_loc_tag_val = (int)value - 1;
    template->unsigned_long_loc_tag_val = (unsigned long)value - 1;
    template->long_loc_tag_val = (long)value - 1;
    template->float_loc_tag_val = (float)value - 1;
    template->double_loc_tag_val = (double)value - 1;
    template->unsigned_short_loc_tag_val = (unsigned short)value - 1;
    template->short_loc_tag_val = (short)value - 1;
    template->float_short_loc_tag_val = (short)value - 1;
}

static void test_init_buffer(struct sharp_test_data_types_t* dtype,
                             struct sharp_test_data_types_t* tag_dtype,
                             int length,
                             struct data_template* template,
                             void* buffer,
                             enum sharp_reduce_op op)
{
    int i;
    char* ptr = buffer;

    for (i = 0; i < length; i++) {
        switch (dtype->id) {
            case SHARP_DTYPE_INT:
                *(int*)ptr = template->int_val;
                break;
            case SHARP_DTYPE_LONG:
                *(long*)ptr = template->long_val;
                break;
            case SHARP_DTYPE_UNSIGNED:
                *(unsigned int*)ptr = template->unsigned_int_val;
                break;
            case SHARP_DTYPE_UNSIGNED_LONG:
                *(unsigned long*)ptr = template->unsigned_long_val;
                break;
            case SHARP_DTYPE_FLOAT:
                *(float*)ptr = template->float_val;
                break;
            case SHARP_DTYPE_DOUBLE:
                *(double*)ptr = template->double_val;
                break;
            case SHARP_DTYPE_UNSIGNED_SHORT:
                *(unsigned short*)ptr = template->unsigned_short_val;
                break;
            case SHARP_DTYPE_SHORT:
                *(short*)ptr = template->short_val;
                break;
            case SHARP_DTYPE_UINT8:
                *(unsigned char*)ptr = template->uint8_val;
                break;
            case SHARP_DTYPE_INT8:
                *(char*)ptr = template->int8_val;
                break;
            case SHARP_DTYPE_FLOAT_SHORT:
            case SHARP_DTYPE_BFLOAT16:
                *(short*)ptr = template->float_short_val;
                break;
            default:
                break;
        }
        ptr += dtype->size;
        if (is_op_min_max_loc(op)) {
            if (coll_sharp_component.sharp_caps.sharp_pkt_version == 0 && enable_sharp_coll) {
                *(int*)ptr = template->int_loc_tag_val;
                ptr += dtype->size;
            } else {
                switch (tag_dtype->id) {
                    case SHARP_DTYPE_INT:
                        *(int*)ptr = template->int_loc_tag_val;
                        break;
                    case SHARP_DTYPE_LONG:
                        *(long*)ptr = template->long_loc_tag_val;
                        break;
                    case SHARP_DTYPE_UNSIGNED:
                        *(unsigned int*)ptr = template->unsigned_int_loc_tag_val;
                        break;
                    case SHARP_DTYPE_UNSIGNED_LONG:
                        *(unsigned long*)ptr = template->unsigned_long_loc_tag_val;
                        break;
                    case SHARP_DTYPE_FLOAT:
                        *(float*)ptr = template->float_loc_tag_val;
                        break;
                    case SHARP_DTYPE_DOUBLE:
                        *(double*)ptr = template->double_loc_tag_val;
                        break;
                    case SHARP_DTYPE_UNSIGNED_SHORT:
                        *(unsigned short*)ptr = template->unsigned_short_loc_tag_val;
                        break;
                    case SHARP_DTYPE_SHORT:
                        *(short*)ptr = template->short_loc_tag_val;
                        break;
                    case SHARP_DTYPE_FLOAT_SHORT:
                        *(short*)ptr = template->float_short_loc_tag_val;
                        break;
                    default:
                        break;
                }
                ptr += tag_dtype->size;
            }
        }
    }
}

static int is_valid_datatype(struct sharp_test_data_types_t* data_type)
{
    if (enable_sharp_coll) {
        return (__SHARP_TEST_BIT(data_type->id) & coll_sharp_component.sharp_caps.support_mask.dtypes);
    } else {
        return (data_type->mpi_id != MPI_DATATYPE_NULL);
    }
}

static int is_valid_tag_datatype(struct sharp_test_data_types_t* dtype, struct sharp_test_data_types_t* tag_dtype)
{
    if (enable_sharp_coll) {
        return (__SHARP_TEST_BIT(tag_dtype->id) & coll_sharp_component.sharp_caps.support_mask.tag_dtypes);
    } else {
        return (sharp_test_mpi_min_max_datatype[dtype->id][tag_dtype->id] != MPI_DATATYPE_NULL);
    }
}

/* handle coll request to complete */
static int is_valid_op(enum sharp_datatype type, enum sharp_reduce_op op)
{
    if (enable_sharp_coll) {
        if (!(__SHARP_TEST_BIT(op) & coll_sharp_component.sharp_caps.support_mask.reduce_ops)) {
            return 0;
        }
    }

    switch (type) {
        case SHARP_DTYPE_INT:
        case SHARP_DTYPE_LONG:
        case SHARP_DTYPE_UNSIGNED:
        case SHARP_DTYPE_UNSIGNED_LONG:
        case SHARP_DTYPE_UNSIGNED_SHORT:
        case SHARP_DTYPE_SHORT:
        case SHARP_DTYPE_UINT8:
        case SHARP_DTYPE_INT8:
            return 1;
        case SHARP_DTYPE_FLOAT:
        case SHARP_DTYPE_FLOAT_SHORT:
        case SHARP_DTYPE_BFLOAT16:
        case SHARP_DTYPE_DOUBLE:
            if (op == SHARP_OP_MIN || op == SHARP_OP_MAX || op == SHARP_OP_SUM || op == SHARP_OP_PROD || op == SHARP_OP_MINLOC ||
                op == SHARP_OP_MAXLOC)
                return 1;
        case SHARP_DTYPE_NULL:
        default:
            return 0;
    }

    return 0;
}

static double apply_double_op(enum sharp_reduce_op reduce_op, double x, double y)
{
    double value;
    switch (reduce_op) {
        case SHARP_OP_MAX:
        case SHARP_OP_MAXLOC:
            value = (x > y ? x : y);
            break;
        case SHARP_OP_MIN:
        case SHARP_OP_MINLOC:
            value = (x < y ? x : y);
            break;
        case SHARP_OP_SUM:
            value = x + y;
            break;
        case SHARP_OP_PROD:
            value = x * y;
            break;
        default:
            value = 0.0;
            break;
    }

    return value;
}

static long apply_int_op(enum sharp_reduce_op reduce_op, long x, long y)
{
    long value;
    switch (reduce_op) {
        case SHARP_OP_MAX:
        case SHARP_OP_MAXLOC:
            value = (x > y ? x : y);
            break;
        case SHARP_OP_MIN:
        case SHARP_OP_MINLOC:
            value = (x < y ? x : y);
            break;
        case SHARP_OP_SUM:
            value = x + y;
            break;
        case SHARP_OP_PROD:
            value = x * y;
            break;
        case SHARP_OP_LAND:
            value = x && y;
            break;
        case SHARP_OP_BAND:
            value = x & y;
            break;
        case SHARP_OP_LOR:
            value = x || y;
            break;
        case SHARP_OP_BOR:
            value = x | y;
            break;
        case SHARP_OP_LXOR:
            value = (x || y) && !(x && y);
            break;
        case SHARP_OP_BXOR:
            value = x ^ y;
            break;
        case SHARP_OP_NULL:
        default:
            value = 0;
            break;
    }
    return value;
}

static void test_get_reduce_result(enum sharp_reduce_op reduce_op, int num_procs, struct data_template* template)
{
    long x1 = 1, x2 = 2, iresult_64;
    int iresult_32, iresult_16;
    char iresult_8;
    double dx1 = 1.0, dx2 = 2.0, dresult = 0.0;

    if (reduce_op == SHARP_OP_MINLOC) {
        template->int_min_max_loc_result = 0;
    } else if (reduce_op == SHARP_OP_MAXLOC) {
        template->int_min_max_loc_result = (num_procs - 1);
    }

    if (num_procs == 1) {
        test_data_template_init(template, 1);
        return;
    }

    iresult_64 = apply_int_op(reduce_op, x1, x2);
    for (x1 = 3, x2 = 3; x2 <= num_procs; x1++, x2++) {
        iresult_64 = apply_int_op(reduce_op, iresult_64, x1);
    }
    iresult_32 = (int)iresult_64;
    iresult_16 = (short)iresult_64;
    iresult_8 = (char)iresult_64;
    template->int_val = iresult_32;
    template->unsigned_int_val = iresult_32;
    template->long_val = iresult_64;
    template->unsigned_long_val = iresult_64;
    template->unsigned_short_val = iresult_16;
    template->short_val = iresult_16;
    template->uint8_val = (unsigned char)iresult_8;
    template->int8_val = iresult_8;

    if (reduce_op < SHARP_OP_LAND || reduce_op == SHARP_OP_MINLOC || reduce_op == SHARP_OP_MAXLOC) {
        dresult = apply_double_op(reduce_op, dx1, dx2);
        for (dx2 = 3.0, x2 = 3; x2 <= num_procs; x2++, dx2 += 1.0) {
            dresult = apply_double_op(reduce_op, dresult, dx2);
        }
    }
    template->float_val = (float)dresult;
    template->double_val = (double)dresult;
    template->float_short_val = (short)dresult;
}

static int test_check_buffer_errors(struct sharp_test_data_types_t* dtype,
                                    struct sharp_test_data_types_t* tag_dtype,
                                    int length,
                                    struct data_template* template,
                                    void* buffer,
                                    enum sharp_reduce_op op)
{
    int i, errors = 0;
    char* ptr = buffer;

    for (i = 0; i < length; i++) {
        switch (dtype->id) {
            case SHARP_DTYPE_UINT8:
                if (*(unsigned char*)ptr != template->uint8_val) {
                    errors++;
                    fprintf(stdout,
                            "Data validation error. pos:%d, value:%d, expected:%d \n",
                            i,
                            *(unsigned char*)ptr,
                            template->uint8_val);
                }
                break;
            case SHARP_DTYPE_INT8:
                if (*(char*)ptr != template->int8_val) {
                    errors++;
                    fprintf(stdout, "Data validation error. pos:%d, value:%d, expected:%d \n", i, *(char*)ptr, template->int8_val);
                }
                break;
            case SHARP_DTYPE_INT:
                if (*(int*)ptr != template->int_val) {
                    errors++;
                    fprintf(stdout, "Data validation error. pos:%d, value:%d, expected:%d \n", i, *(int*)ptr, template->int_val);
                }
                break;
            case SHARP_DTYPE_LONG:
                if (*(long*)ptr != template->long_val) {
                    errors++;
                    fprintf(stdout, "Data validation error. pos:%d, value:%ld, expected:%ld \n", i, *(long*)ptr, template->long_val);
                }
                break;
            case SHARP_DTYPE_UNSIGNED:
                if (*(unsigned int*)ptr != template->unsigned_int_val) {
                    errors++;
                    fprintf(stdout,
                            "Data validation error. pos:%d, value:%d, expected:%d \n",
                            i,
                            *(unsigned int*)ptr,
                            template->unsigned_int_val);
                }
                break;
            case SHARP_DTYPE_UNSIGNED_LONG:
                if (*(unsigned long*)ptr != template->unsigned_long_val) {
                    errors++;
                    fprintf(stdout,
                            "Data validation error. pos:%d, value:%lu, expected:%lu \n",
                            i,
                            *(unsigned long*)ptr,
                            template->unsigned_long_val);
                }
                break;
            case SHARP_DTYPE_FLOAT:
                if (*(float*)ptr != template->float_val) {
                    errors++;
                    fprintf(stdout,
                            "Data validation error. pos:%d, value:%10.10f, expected:%10.10f \n",
                            i,
                            *(float*)ptr,
                            template->float_val);
                }
                break;
            case SHARP_DTYPE_DOUBLE:
                if (*(double*)ptr != template->double_val) {
                    errors++;
                    fprintf(stdout,
                            "Data validation error. pos:%d, value:%10.10f, expected:%10.10f\n",
                            i,
                            *(double*)ptr,
                            template->float_val);
                }
                break;
            case SHARP_DTYPE_UNSIGNED_SHORT:
                if (*(unsigned short*)ptr != template->unsigned_short_val) {
                    errors++;
                    fprintf(stdout,
                            "Data validation error. pos:%hu, value:%d, expected:%hu\n",
                            i,
                            *(unsigned short*)ptr,
                            template->unsigned_short_val);
                }
                break;
            case SHARP_DTYPE_SHORT:
                if (*(short*)ptr != template->short_val) {
                    errors++;
                    fprintf(stdout, "Data validation error. pos:%hi, value:%d, expected:%hi\n", i, *(short*)ptr, template->short_val);
                }
                break;
            case SHARP_DTYPE_FLOAT_SHORT:
            case SHARP_DTYPE_BFLOAT16:
                if (*(short*)ptr != template->float_short_val) {
                    errors++;
                    fprintf(stdout,
                            "Data validation error. pos:%d, value:%hi, expected:%hi \n",
                            i,
                            *(short*)ptr,
                            template->float_short_val);
                }
                break;
            case SHARP_DTYPE_NULL:
            default:
                break;
        }
        ptr += dtype->size;
        if (is_op_min_max_loc(op)) {
            if (coll_sharp_component.sharp_caps.sharp_pkt_version == 0 && enable_sharp_coll) {
                if (*(int*)ptr != template->int_min_max_loc_result) {
                    errors++;
                    fprintf(stdout,
                            "min/max location validation error. pos:%d, value:%d, expected:%d \n",
                            i,
                            *(int*)ptr,
                            template->int_min_max_loc_result);
                }
                ptr += dtype->size;
            } else {
                switch (tag_dtype->id) {
                    case SHARP_DTYPE_INT:
                        if (*(int*)ptr != template->int_min_max_loc_result) {
                            errors++;
                            fprintf(stdout,
                                    "min/max location validation error. pos:%d, value:%d, expected:%d \n",
                                    i,
                                    *(int*)ptr,
                                    template->int_min_max_loc_result);
                        }
                        break;
                    case SHARP_DTYPE_LONG:
                        if (*(long*)ptr != (long)template->int_min_max_loc_result) {
                            errors++;
                            fprintf(stdout,
                                    "min/max location validation error. pos:%d, value:%ld, expected:%ld \n",
                                    i,
                                    *(long*)ptr,
                                    (long)template->int_min_max_loc_result);
                        }
                        break;
                    case SHARP_DTYPE_UNSIGNED:
                        if (*(unsigned int*)ptr != (unsigned int)template->int_min_max_loc_result) {
                            errors++;
                            fprintf(stdout,
                                    "min/max location validation error. pos:%d, value:%u, expected:%u \n",
                                    i,
                                    *(unsigned int*)ptr,
                                    (unsigned int)template->int_min_max_loc_result);
                        }
                        break;
                    case SHARP_DTYPE_UNSIGNED_LONG:
                        if (*(unsigned long*)ptr != (unsigned long)template->int_min_max_loc_result) {
                            errors++;
                            fprintf(stdout,
                                    "min/max location validation error. pos:%d, value:%lu, expected:%lu \n",
                                    i,
                                    *(unsigned long*)ptr,
                                    (unsigned long)template->int_min_max_loc_result);
                        }
                        break;
                    case SHARP_DTYPE_FLOAT:
                        if (*(float*)ptr != (float)template->int_min_max_loc_result) {
                            errors++;
                            fprintf(stdout,
                                    "min/max location validation error. pos:%d, value:%f, expected:%f \n",
                                    i,
                                    *(float*)ptr,
                                    (float)template->int_min_max_loc_result);
                        }
                        break;
                    case SHARP_DTYPE_DOUBLE:
                        if (*(double*)ptr != (double)template->int_min_max_loc_result) {
                            errors++;
                            fprintf(stdout,
                                    "min/max location validation error. pos:%d, value:%f, expected:%f \n",
                                    i,
                                    *(double*)ptr,
                                    (double)template->int_min_max_loc_result);
                        }
                        break;
                    case SHARP_DTYPE_UNSIGNED_SHORT:
                        if (*(unsigned short*)ptr != (unsigned short)template->int_min_max_loc_result) {
                            errors++;
                            fprintf(stdout,
                                    "min/max location validation error. pos:%d, value:%d, expected:%d \n",
                                    i,
                                    *(unsigned short*)ptr,
                                    (unsigned short)template->int_min_max_loc_result);
                        }
                        break;
                    case SHARP_DTYPE_SHORT:
                        if (*(short*)ptr != (short)template->int_min_max_loc_result) {
                            errors++;
                            fprintf(stdout,
                                    "min/max location validation error. pos:%d, value:%d, expected:%d \n",
                                    i,
                                    *(short*)ptr,
                                    (short)template->int_min_max_loc_result);
                        }
                        break;
                    case SHARP_DTYPE_FLOAT_SHORT:
                        if (*(short*)ptr != (short)template->int_min_max_loc_result) {
                            errors++;
                            fprintf(stdout,
                                    "min/max location validation error. pos:%d, value:%d, expected:%d \n",
                                    i,
                                    *(short*)ptr,
                                    (short)template->int_min_max_loc_result);
                        }
                        break;
                    default:
                        break;
                }
                ptr += tag_dtype->size;
            }
        }
        if (errors) {
            break;
        }
    }
    return errors;
}

void coll_test_sharp_barrier_complex(coll_sharp_module_t* sharp_comm)
{
    int i, size, rank, ret;
    MPI_Comm mpi_comm;

    mpi_comm = sharp_comm->comm;
    MPI_Comm_size(mpi_comm, &size);
    MPI_Comm_rank(mpi_comm, &rank);

    if (rank == 0)
        printf("\nBarrier validation test. expecting rank print in order ");

    for (i = 0; i < size; i++) {
        if (rank == i) {
            printf("Rank %d\n", rank);
            fflush(stdout);
            usleep(100000);
        }

        if (enable_sharp_coll) {
            ret = sharp_coll_do_barrier(sharp_comm->sharp_coll_comm);
            if (ret != SHARP_COLL_SUCCESS) {
                if (rank == 0) {
                    fprintf(stderr, "Barrier failed: %s\n", sharp_coll_strerror(ret));
                }
                exit(-1);
            }
        } else {
            MPI_Barrier(mpi_comm);
        }
    }
}

void coll_test_sharp_allreduce_complex(coll_sharp_module_t* sharp_comm)
{
    int *inbuf = NULL, *outbuf = NULL;
    int count, length, data_len;
    int type, op_count, tag_type;
    int comm_size, my_rank;
    int errors, ret;
    struct data_template template;
    MPI_Comm mpi_comm = sharp_comm->comm;
    struct sharp_coll_reduce_spec reduce_spec;
    void *s_mem_mr = NULL, *r_mem_mr = NULL;

    MPI_Comm_size(mpi_comm, &comm_size);
    MPI_Comm_rank(mpi_comm, &my_rank);

    if (my_rank == 0)
        printf("\nAllreduce validation test with all datatypes and reduce ops.\n");

    inbuf = allocate_memory(sharp_conf.max_message_size, sharp_conf.s_mem_type, 1);
    if (!inbuf) {
        goto fn_fail;
    }

    outbuf = allocate_memory(sharp_conf.max_message_size, sharp_conf.r_mem_type, 0);
    if (!outbuf) {
        goto fn_fail;
    }

    if (enable_sharp_coll) {
        ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, inbuf, sharp_conf.max_message_size, &s_mem_mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "Failed to register source buffer\n");
            goto fn_fail;
        }

        ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, outbuf, sharp_conf.max_message_size, &r_mem_mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "Failed to register receive buffer\n");
            goto fn_fail;
        }
    }

    /*loop over all supported datatypes */
    for (type = 0; sharp_test_data_types[type].id != SHARP_DTYPE_NULL; type++) {
        if (!is_valid_datatype(&sharp_test_data_types[type])) {
            continue;
        }

        /*loop over all sizes */
        for (length = sharp_conf.min_message_size; length <= sharp_conf.max_message_size; length = (length == 0) ? 1 : length * 2) {
            /* loop over all ops */
            for (op_count = 0; sharp_test_ops[op_count].id != SHARP_OP_NULL; op_count++) {
                /* Not all ops are valid on all datatypes*/
                if (!is_valid_op(sharp_test_data_types[type].id, sharp_test_ops[op_count].id))
                    continue;

                for (tag_type = 0; sharp_test_data_types[tag_type].id != SHARP_DTYPE_NULL; tag_type++) {
                    if (is_op_min_max_loc(sharp_test_ops[op_count].id) &&
                        !is_valid_tag_datatype(&sharp_test_data_types[type], &sharp_test_data_types[tag_type]))
                    {
                        continue;
                    }

                    if (is_op_min_max_loc(sharp_test_ops[op_count].id)) {
                        if (coll_sharp_component.sharp_caps.sharp_pkt_version == 0 && enable_sharp_coll) {
                            count = length / (sharp_test_data_types[type].size * 2);
                            data_len = count * sharp_test_data_types[type].size * 2;
                        } else {
                            count = length / (sharp_test_data_types[type].size + sharp_test_data_types[tag_type].size);
                            data_len = count * (sharp_test_data_types[type].size + sharp_test_data_types[tag_type].size);
                        }
                        if (count <= 0) {
                            continue;
                        }

                        if (my_rank == 0) {
                            fprintf(stdout,
                                    "Allreduce Test: data type :%s tag_type:%s: Op:%s count:%d - ",
                                    sharp_test_data_types[type].name,
                                    sharp_test_data_types[tag_type].name,
                                    sharp_test_ops[op_count].name,
                                    count);
                        }
                    } else {
                        count = length / (sharp_test_data_types[type].size);
                        data_len = (count * sharp_test_data_types[type].size);
                        if (count <= 0) {
                            continue;
                        }
                        if (my_rank == 0) {
                            fprintf(stdout,
                                    "Allreduce Test: data type :%s Op:%s count:%d - ",
                                    sharp_test_data_types[type].name,
                                    sharp_test_ops[op_count].name,
                                    count);
                        }
                    }

                    /*Initialize input buffers*/
                    test_data_template_init(&template, (my_rank + 1));
                    test_init_buffer(&sharp_test_data_types[type],
                                     &sharp_test_data_types[tag_type],
                                     count,
                                     &template,
                                     inbuf,
                                     sharp_test_ops[op_count].id);
                    test_data_template_init(&template, -1);
                    test_init_buffer(&sharp_test_data_types[type],
                                     &sharp_test_data_types[tag_type],
                                     count,
                                     &template,
                                     outbuf,
                                     sharp_test_ops[op_count].id);

                    /* Do Allreduce */
                    if (enable_sharp_coll) {
                        reduce_spec.sbuf_desc.buffer.ptr = inbuf;
                        reduce_spec.sbuf_desc.buffer.length = data_len;
                        reduce_spec.sbuf_desc.buffer.mem_handle = s_mem_mr;
                        reduce_spec.sbuf_desc.type = SHARP_DATA_BUFFER;
                        reduce_spec.sbuf_desc.mem_type = SHARP_MEM_TYPE_HOST;
                        reduce_spec.rbuf_desc.buffer.ptr = outbuf;
                        reduce_spec.rbuf_desc.buffer.length = data_len;
                        reduce_spec.rbuf_desc.buffer.mem_handle = r_mem_mr;
                        reduce_spec.rbuf_desc.type = SHARP_DATA_BUFFER;
                        reduce_spec.rbuf_desc.mem_type = SHARP_MEM_TYPE_HOST;
                        reduce_spec.dtype = sharp_test_data_types[type].id;
                        reduce_spec.aggr_mode = SHARP_AGGREGATION_NONE;
                        if (coll_sharp_component.sharp_caps.sharp_pkt_version == 0) {
                            reduce_spec.tag_dtype = sharp_test_data_types[type].id;
                        } else {
                            reduce_spec.tag_dtype = sharp_test_data_types[tag_type].id;
                        }
                        reduce_spec.length = count;
                        reduce_spec.op = sharp_test_ops[op_count].id;

                        ret = sharp_coll_do_allreduce(sharp_comm->sharp_coll_comm, &reduce_spec);
                        if (ret != SHARP_COLL_SUCCESS) {
                            if (my_rank == 0) {
                                fprintf(stderr, "Allreduce failed: %s\n", sharp_coll_strerror(ret));
                            }
                            exit(-1);
                        }
                    } else {
                        MPI_Datatype dt = sharp_test_data_types[type].mpi_id;
                        if (is_op_min_max_loc(sharp_test_ops[op_count].id)) {
                            /*get mic_loc max_loc datatype */
                            dt = sharp_test_mpi_min_max_datatype[sharp_test_data_types[type].id][sharp_test_data_types[tag_type].id];
                        }

                        MPI_Allreduce(inbuf, outbuf, count, dt, sharp_test_ops[op_count].mpi_op, mpi_comm);
                    }

                    /*Validate result buffer*/
                    errors = 0;
                    test_get_reduce_result(sharp_test_ops[op_count].id, comm_size, &template);
                    errors = test_check_buffer_errors(&sharp_test_data_types[type],
                                                      &sharp_test_data_types[tag_type],
                                                      count,
                                                      &template,
                                                      outbuf,
                                                      sharp_test_ops[op_count].id);
                    if (errors) {
                        fprintf(stdout,
                                "Allreduce failed. comm:%p count:%d date type:%s reduce Op:%s\n",
                                sharp_comm->comm,
                                count,
                                sharp_test_data_types[type].name,
                                sharp_test_ops[op_count].name);
                        exit(-1);
                    }
                    if (my_rank == 0) {
                        fprintf(stdout, " PASS\n");
                    }

                    if (!is_op_min_max_loc(sharp_test_ops[op_count].id)) {
                        break;
                    }
                }
            }
        }
    }

fn_fail:
    if (enable_sharp_coll) {
        if (s_mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, s_mem_mr);
        if (r_mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, r_mem_mr);
    }
    free_memory(inbuf, sharp_conf.s_mem_type, 1);
    free_memory(outbuf, sharp_conf.r_mem_type, 0);
}
void coll_test_sharp_iallreduce_complex(coll_sharp_module_t* sharp_comm)
{
    fprintf(stdout, "iallreduce complex not implemented\n");
}

void coll_test_sharp_barrier_basic(coll_sharp_module_t* sharp_comm)
{
    int rank, ret;
    MPI_Comm mpi_comm = sharp_comm->comm;
    MPI_Comm_rank(mpi_comm, &rank);

    if (rank == 0)
        printf("Barrier Basic test - ");

    if (enable_sharp_coll) {
        ret = sharp_coll_do_barrier(sharp_comm->sharp_coll_comm);
        if (ret != SHARP_COLL_SUCCESS) {
            if (rank == 0) {
                fprintf(stderr, "Barrier failed: %s\n", sharp_coll_strerror(ret));
            }
            exit(-1);
        }
    } else {
        MPI_Barrier(mpi_comm);
    }
    if (rank == 0) {
        fprintf(stdout, " PASS\n");
    }
}

void coll_test_sharp_ibarrier_basic(coll_sharp_module_t* sharp_comm)
{
    int rank, ret;
    void* handle = NULL;
    MPI_Comm mpi_comm = sharp_comm->comm;
    MPI_Comm_rank(mpi_comm, &rank);

    if (rank == 0)
        printf("Ibarrier Basic test - ");

    if (enable_sharp_coll) {
        ret = sharp_coll_do_barrier_nb(sharp_comm->sharp_coll_comm, &handle);
        if (ret != SHARP_COLL_SUCCESS) {
            if (rank == 0) {
                fprintf(stderr, "Barrier failed: %s\n", sharp_coll_strerror(ret));
            }
            exit(-1);
        }
        sharp_coll_req_wait(handle);
    } else {
        MPI_Barrier(mpi_comm);
    }
    if (rank == 0) {
        fprintf(stdout, " PASS\n");
    }
}

static void fill_int_pattren(int* buffer, int rank, size_t count, enum sharp_data_memory_type mem_type)
{
    size_t i;

    if (mem_type == SHARP_MEM_TYPE_HOST) {
        for (i = 0; i < count; i++) {
            buffer[i] = (rank + i);
        }
    }
#if HAVE_CUDA
    else if (mem_type == SHARP_MEM_TYPE_CUDA)
    {
        int* temp;
        cudaError_t cuerr;
        temp = malloc(count * sizeof(int));
        for (i = 0; i < count; i++) {
            temp[i] = (rank + i);
        }
        cuerr = cudaMemcpy(buffer, temp, count * sizeof(int), cudaMemcpyHostToDevice);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed\n");
            free(temp);
            exit(-1);
        }
        free(temp);
    }
#endif
}

static void check_int_pattren(void* buffer, int rank, int comm_size, size_t count, enum sharp_data_memory_type mem_type, int test)
{
    int result = 0;
    int* outbuf = (int*)buffer;
    void* temp = NULL;
    size_t i;

#if HAVE_CUDA
    if (mem_type == SHARP_MEM_TYPE_CUDA) {
        cudaError_t cuerr;
        temp = malloc(count * sizeof(int));
        cuerr = cudaMemcpy(temp, buffer, count * sizeof(int), cudaMemcpyDeviceToHost);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed\n");
            exit(-1);
        }
        outbuf = (int*)temp;
    }
#endif

    for (i = 0; i < count; i++) {
        if (test == TEST_ALLREDUCE) {
            result = i * comm_size + (comm_size * (comm_size - 1)) / 2;
        } else if (test == TEST_BCAST) {
            result = (comm_size - 1 + i);
        } else if (test == TEST_REDUCE_SCATTER) {
            result = (rank * count + i) * comm_size + (comm_size * (comm_size - 1)) / 2;
        } else if (test == TEST_ALLGATHER) {
            result = (i / (count / comm_size) + i % (count / comm_size));
        }

        if (outbuf[i] != result) {
            fprintf(stdout, "[%d] data validation failed. At pos:%lu got  =%d, expected:%d \n", rank, i, outbuf[i], result);
            exit(-1);
        }
    }
    free(temp);
}

static void create_iov_buffer(struct sharp_coll_data_desc* desc, int iov_count, char* buffer, size_t length, void* memh)
{
    size_t offset;
    int i, iovsize, remainder;

    desc->type = SHARP_DATA_IOV;
    desc->iov.count = iov_count;
    desc->iov.vector = malloc(iov_count * sizeof(struct sharp_data_iov));
    assert(desc->iov.vector != NULL);
    iovsize = length / iov_count;
    remainder = length % iov_count;

    offset = 0;
    for (i = 0; i < iov_count; i++) {
        desc->iov.vector[i].ptr = buffer + offset;
        desc->iov.vector[i].mem_handle = memh;
        desc->iov.vector[i].length = iovsize;
        if (remainder > 0) {
            remainder--;
            desc->iov.vector[i].length++;
        }
        offset += desc->iov.vector[i].length;
    }
    assert(offset == length);
}

void coll_test_sharp_allreduce_basic(coll_sharp_module_t* sharp_comm, int is_rooted)
{
    int *inbuf = NULL, *outbuf = NULL;
    int comm_size, rank, root, ret;
    size_t count;
    struct sharp_coll_reduce_spec reduce_spec;
    MPI_Comm mpi_comm;
    void *s_mem_mr = NULL, *r_mem_mr = NULL;

    mpi_comm = sharp_comm->comm;
    count = sharp_conf.max_message_size / sizeof(int);

    MPI_Comm_size(mpi_comm, &comm_size);
    MPI_Comm_rank(mpi_comm, &rank);

    COLL_SET_ROOT(root, coll_root_rank, comm_size);

    if (!is_rooted && rank == 0) {
        printf("Allreduce Basic test (group_size:%d count:%lu op:%s) - ", comm_size, count, "SUM");
    } else {
        printf("Reduce Basic test (group_size:%d count:%lu op:%s root:%d) - ", comm_size, count, "SUM", root);
    }

    inbuf = allocate_memory(sharp_conf.max_message_size, sharp_conf.s_mem_type, 1);
    if (!inbuf) {
        goto fn_fail;
    }

    if (!is_rooted || (rank == root)) {
        outbuf = allocate_memory(sharp_conf.max_message_size, sharp_conf.r_mem_type, 0);
        if (!outbuf) {
            goto fn_fail;
        }
    }

    if (enable_sharp_coll) {
        ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, inbuf, sharp_conf.max_message_size, &s_mem_mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "Failed to register source buffer\n");
            goto fn_fail;
        }

        if (outbuf) {
            ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, outbuf, sharp_conf.max_message_size, &r_mem_mr);
            if (ret != SHARP_COLL_SUCCESS) {
                fprintf(stderr, "Failed to register receive  buffer\n");
                goto fn_fail;
            }
        }
    }

    fill_int_pattren(inbuf, rank, count, sharp_conf.s_mem_type);

    if (enable_sharp_coll) {
        reduce_spec.sbuf_desc.mem_type = sharp_conf.s_mem_type;
        reduce_spec.rbuf_desc.mem_type = sharp_conf.r_mem_type;

        if (sharp_conf.sdata_layout == TEST_DATA_LAYOUT_CONTIG) {
            reduce_spec.sbuf_desc.type = SHARP_DATA_BUFFER;
            reduce_spec.sbuf_desc.buffer.ptr = inbuf;
            reduce_spec.sbuf_desc.buffer.length = (count * sizeof(int));
            reduce_spec.sbuf_desc.buffer.mem_handle = s_mem_mr;
        } else if (sharp_conf.sdata_layout == TEST_DATA_LAYOUT_IOV) {
            create_iov_buffer(&reduce_spec.sbuf_desc, sharp_conf.siov_count, (char*)inbuf, (count * sizeof(int)), s_mem_mr);
        }

        if (!is_rooted || (rank == root)) {
            if (sharp_conf.rdata_layout == TEST_DATA_LAYOUT_CONTIG) {
                reduce_spec.rbuf_desc.type = SHARP_DATA_BUFFER;
                reduce_spec.rbuf_desc.buffer.ptr = outbuf;
                reduce_spec.rbuf_desc.buffer.length = (count * sizeof(int));
                reduce_spec.rbuf_desc.buffer.mem_handle = r_mem_mr;
            } else if (sharp_conf.rdata_layout == TEST_DATA_LAYOUT_IOV) {
                create_iov_buffer(&reduce_spec.rbuf_desc, sharp_conf.riov_count, (char*)outbuf, (count * sizeof(int)), r_mem_mr);
            }
        }

        reduce_spec.dtype = SHARP_DTYPE_INT;
        reduce_spec.length = count;
        reduce_spec.op = SHARP_OP_SUM;
        reduce_spec.root = root;
        reduce_spec.aggr_mode = SHARP_AGGREGATION_NONE;

        if (!is_rooted) {
            ret = sharp_coll_do_allreduce(sharp_comm->sharp_coll_comm, &reduce_spec);
        } else {
            ret = sharp_coll_do_reduce(sharp_comm->sharp_coll_comm, &reduce_spec);
        }
        if (ret != SHARP_COLL_SUCCESS) {
            if (rank == 0) {
                fprintf(stderr, "%s failed: %s\n", (is_rooted) ? "Reduce" : "Allreduce", sharp_coll_strerror(ret));
            }
            exit(-1);
        }
    } else {
        if (!is_rooted) {
            MPI_Allreduce(inbuf, outbuf, count, MPI_INT, MPI_SUM, mpi_comm);
        } else {
            MPI_Reduce(inbuf, outbuf, count, MPI_INT, MPI_SUM, root, mpi_comm);
        }
    }

    if (!is_rooted || (root == rank)) {
        check_int_pattren(outbuf, rank, comm_size, count, sharp_conf.r_mem_type, TEST_ALLREDUCE);
        if (root == rank) {
            fprintf(stdout, " PASS\n");
        }
    }

fn_fail:
    if (sharp_conf.sdata_layout == TEST_DATA_LAYOUT_IOV) {
        free(reduce_spec.sbuf_desc.iov.vector);
    }
    if (sharp_conf.rdata_layout == TEST_DATA_LAYOUT_IOV && (!is_rooted || (root == rank))) {
        free(reduce_spec.rbuf_desc.iov.vector);
    }

    if (enable_sharp_coll) {
        if (s_mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, s_mem_mr);
        if (r_mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, r_mem_mr);
    }

    free_memory(inbuf, sharp_conf.s_mem_type, 1);
    free_memory(outbuf, sharp_conf.r_mem_type, 0);
}

void coll_test_sharp_iallreduce_basic(coll_sharp_module_t* sharp_comm)
{
    int *inbuf = NULL, *outbuf = NULL;
    int count, result, i, j, comm_size, rank, ret;
    struct sharp_coll_reduce_spec reduce_spec;
    MPI_Comm mpi_comm;
    MPI_Request mpi_reqs[MAX_NB_REQS];
    MPI_Status mpi_status;
    void* sharp_reqs[MAX_NB_REQS];
    void *s_mem_mr = NULL, *r_mem_mr = NULL;
    int outstanding_non_blocking_ops = nbc_count;
    int buffer_size;

    mpi_comm = sharp_comm->comm;
    count = sharp_conf.max_message_size / sizeof(int);
    buffer_size = sharp_conf.max_message_size * outstanding_non_blocking_ops;

    MPI_Comm_size(mpi_comm, &comm_size);
    MPI_Comm_rank(mpi_comm, &rank);

    if (rank == 0)
        printf("IAllreduce Basic test(count:%d op:%s NB_count:%d) - ", count, "SUM", nbc_count);

    inbuf = allocate_memory(buffer_size, sharp_conf.s_mem_type, 1);
    if (!inbuf) {
        goto fn_fail;
    }

    outbuf = allocate_memory(buffer_size, sharp_conf.r_mem_type, 0);
    if (!outbuf) {
        goto fn_fail;
    }

    if (enable_sharp_coll) {
        ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, inbuf, buffer_size, &s_mem_mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "Failed to register source buffer\n");
            goto fn_fail;
        }

        ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, outbuf, buffer_size, &r_mem_mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "Failed to register receive buffer\n");
            goto fn_fail;
        }
    }

    for (i = 0; i < outstanding_non_blocking_ops; i++) {
        for (j = 0; j < count; j++) {
            inbuf[(i * count) + j] = (rank + j);
        }
    }

    for (i = 0; i < outstanding_non_blocking_ops; i++) {
        if (enable_sharp_coll) {
            reduce_spec.sbuf_desc.mem_type = sharp_conf.s_mem_type;
            reduce_spec.rbuf_desc.mem_type = sharp_conf.r_mem_type;

            if (sharp_conf.sdata_layout == TEST_DATA_LAYOUT_CONTIG) {
                reduce_spec.sbuf_desc.buffer.ptr = inbuf + (i * count);
                reduce_spec.sbuf_desc.buffer.length = (count * sizeof(int));
                reduce_spec.sbuf_desc.buffer.mem_handle = s_mem_mr;
                reduce_spec.sbuf_desc.type = SHARP_DATA_BUFFER;
            } else if (sharp_conf.sdata_layout == TEST_DATA_LAYOUT_IOV) {
                create_iov_buffer(&reduce_spec.sbuf_desc,
                                  sharp_conf.siov_count,
                                  (char*)(inbuf + (i * count)),
                                  (count * sizeof(int)),
                                  s_mem_mr);
            }

            if (sharp_conf.rdata_layout == TEST_DATA_LAYOUT_CONTIG) {
                reduce_spec.rbuf_desc.buffer.ptr = outbuf + (i * count);
                reduce_spec.rbuf_desc.buffer.length = (count * sizeof(int));
                reduce_spec.rbuf_desc.buffer.mem_handle = r_mem_mr;
                reduce_spec.rbuf_desc.type = SHARP_DATA_BUFFER;
            } else if (sharp_conf.rdata_layout == TEST_DATA_LAYOUT_IOV) {
                create_iov_buffer(&reduce_spec.rbuf_desc,
                                  sharp_conf.riov_count,
                                  (char*)(outbuf + (i * count)),
                                  (count * sizeof(int)),
                                  r_mem_mr);
            }

            reduce_spec.dtype = SHARP_DTYPE_INT;
            reduce_spec.length = count;
            reduce_spec.op = SHARP_OP_SUM;
            reduce_spec.aggr_mode = SHARP_AGGREGATION_NONE;

            ret = sharp_coll_do_allreduce_nb(sharp_comm->sharp_coll_comm, &reduce_spec, &sharp_reqs[i]);
            if (ret != SHARP_COLL_SUCCESS) {
                if (rank == 0)
                    fprintf(stderr, "Allreduce failed: %s\n", sharp_coll_strerror(ret));
                exit(-1);
            }
        } else {
            MPI_Iallreduce(inbuf + (i * count), outbuf + (i * count), count, MPI_INT, MPI_SUM, mpi_comm, &mpi_reqs[i]);
        }
    }

    for (j = 0; j < outstanding_non_blocking_ops; j++) {
        if (enable_sharp_coll) {
            ret = sharp_coll_req_wait(sharp_reqs[j]);
            if (ret != SHARP_COLL_SUCCESS) {
                fprintf(stderr, "sharp_coll_req_wait failed \n");
                exit(-1);
            }
        } else {
            MPI_Wait(&mpi_reqs[j], &mpi_status);
        }

        for (i = 0; i < count; i++) {
            result = i * comm_size + (comm_size * (comm_size - 1)) / 2;
            if (outbuf[(j * count) + i] != result) {
                fprintf(stdout, "data validation failed. #op:%d At pos:%d got  =%d, expected:%d\n", j, i, outbuf[(j * count) + i], result);
                exit(-1);
            }
        }
    }
    if (rank == 0) {
        fprintf(stdout, " PASS\n");
    }

fn_fail:
    if (sharp_conf.sdata_layout == TEST_DATA_LAYOUT_IOV) {
        free(reduce_spec.sbuf_desc.iov.vector);
    }
    if (sharp_conf.rdata_layout == TEST_DATA_LAYOUT_IOV) {
        free(reduce_spec.rbuf_desc.iov.vector);
    }

    if (enable_sharp_coll) {
        if (s_mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, s_mem_mr);
        if (r_mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, r_mem_mr);
    }
    free_memory(inbuf, sharp_conf.s_mem_type, 1);
    free_memory(outbuf, sharp_conf.r_mem_type, 0);
}
static void __attribute__((unused))
print_coll_iterations_perf_data(double* all_iter_time, int rank, int comm_size, int data_size, int iterations)
{
    int i, j, k, max_cutoff;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0, timer = 0.0;
    double *sum_time = NULL, max_iter_time, min_iter_time, latency;
    char *max_cutoffs, *endptr = NULL, *saveptr = NULL, *str = NULL;

    if (getenv("MAX_CUTOFF_LIST"))
        max_cutoffs = strdup(getenv("MAX_CUTOFF_LIST"));
    else
        max_cutoffs = strdup("10");

    if (!max_cutoffs)
        goto fn_fail;

    sum_time = (double*)malloc(sizeof(double) * comm_size);
    if (!sum_time)
        goto fn_fail;

    saveptr = str = strdup(max_cutoffs);
    for (endptr = strtok_r(str, ",", &saveptr); endptr; endptr = strtok_r(NULL, ",", &saveptr)) {
        max_cutoff = atoi(endptr);

        for (j = 0; j < comm_size; j++)
            sum_time[j] = 0.0;
        k = 0;
        max_time = 0.0;
        min_time = 10000.0;
        for (i = 0; i < iterations; i++) {
            max_iter_time = 0.0;
            min_iter_time = 10000.0;
            for (j = 0; j < comm_size; j++) {
                latency = (double)(all_iter_time[i + (j * iterations)] * 1e6);
                if (latency > max_iter_time)
                    max_iter_time = latency;
                if (latency < min_iter_time)
                    min_iter_time = latency;
            }
            if (max_iter_time > max_cutoff)
                continue;
            if (max_iter_time > max_time)
                max_time = max_iter_time;
            if (min_iter_time < min_time)
                min_time = min_iter_time;

            for (j = 0; j < comm_size; j++) {
                sum_time[j] += (double)(all_iter_time[i + (j * iterations)]);
            }
            k++;
        }
        if (k == 0)
            continue;

        timer = 0.0;
        for (j = 0; j < comm_size; j++)
            timer += sum_time[j];
        avg_time = (double)(timer * 1e6) / (comm_size * k);
        if (data_size < 0)
            fprintf(stdout, "%10.2f %10.2f  %10.2f %10d  %10d\n", avg_time, min_time, max_time, k, max_cutoff);
        else
            fprintf(stdout, "%15d %10.2f %10.2f  %10.2f %10d  %10d\n", data_size, avg_time, min_time, max_time, k, max_cutoff);
    }
#if 0
	fprintf(stdout, "Rank  ");
	for (j = 0; j < comm_size; j++)
		fprintf(stdout, "%8d", j);
	for (i = 0; i < iterations; i++) {
		fprintf(stdout, "\nIter:%d", i);
		for (j = 0; j < comm_size; j++) {
			fprintf(stdout, "%8.2f",
				(double)(all_iter_time[i + (j*iterations)] * 1e6));
		}
	}
#endif
    fprintf(stdout, "\n");
fn_fail:
    free(str);
    free(max_cutoffs);
    free(sum_time);
}

void coll_test_sharp_barrier_perf(coll_sharp_module_t* sharp_comm)
{
    int i, j, comm_size, rank, ret;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0, timer = 0.0;
    double *iter_time, max_iter_time, min_iter_time;
    int iterations = perf_test_iterations, skips = perf_test_skips;
    double* all_iter_time = NULL;
    MPI_Comm mpi_comm;

    mpi_comm = sharp_comm->comm;
    MPI_Comm_size(mpi_comm, &comm_size);
    MPI_Comm_rank(mpi_comm, &rank);

    if (rank == 0) {
        fprintf(stdout, "\nBarrier perf test. comm_size:%d\n", comm_size);
        fprintf(stdout, "%10s %10s %10s %10s\n", "Avg", "Min", "Max", "iters");
    }

    iter_time = (double*)malloc(sizeof(double) * iterations);
    if (!iter_time) {
        fprintf(stdout, "Failed to allocate \n");
        goto fn_fail;
    }

    if (rank == 0) {
        all_iter_time = (double*)malloc(sizeof(double) * iterations * comm_size);
        if (!all_iter_time) {
            fprintf(stdout, "Failed to allocate \n");
            goto fn_fail;
        }
    }

    timer = 0.0;
    for (i = 0, j = 0; i < iterations + skips; i++) {
        t_start = sharp_time_sec();

        if (enable_sharp_coll) {
            ret = sharp_coll_do_barrier(sharp_comm->sharp_coll_comm);
            if (ret != SHARP_COLL_SUCCESS) {
                if (rank == 0)
                    fprintf(stderr, "Barrier failed: %s\n", sharp_coll_strerror(ret));
                exit(-1);
            }
        } else {
            MPI_Barrier(mpi_comm);
        }
        if (i >= skips) {
            t_stop = sharp_time_sec();
            timer += (t_stop - t_start);
            iter_time[j++] = (t_stop - t_start);
        }
    }

    latency = (double)(timer * 1e6) / iterations;

    min_iter_time = max_iter_time = iter_time[0];
    for (i = 0; i < iterations; i++) {
        if (min_iter_time > iter_time[i])
            min_iter_time = iter_time[i];
        if (max_iter_time < iter_time[i])
            max_iter_time = iter_time[i];
    }
    min_iter_time = (double)(min_iter_time * 1e6);
    max_iter_time = (double)(max_iter_time * 1e6);

    MPI_Reduce(&min_iter_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, mpi_comm);
    MPI_Reduce(&max_iter_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_comm);
    MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

    avg_time = avg_time / comm_size;

    MPI_Gather(iter_time, iterations, MPI_DOUBLE, all_iter_time, iterations, MPI_DOUBLE, 0, mpi_comm);

    if (rank == 0) {
        fprintf(stdout, "%10.2f %10.2f %10.2f %10d\n", avg_time, min_time, max_time, iterations);
#if 0
		print_coll_iterations_perf_data(all_iter_time, rank, comm_size, -1, iterations);
#endif
        fflush(stdout);
    }
fn_fail:
    free(iter_time);
    free(all_iter_time);
}

void coll_test_sharp_allreduce_perf(coll_sharp_module_t* sharp_comm, int is_rooted)
{
    float *inbuf = NULL, *outbuf = NULL;
    size_t count, max_count, min_count;
    int i, j, comm_size, rank, ret, root;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0, timer = 0.0, avg_bw = 0.0;
    double *iter_time = NULL, max_iter_time, min_iter_time;
    double* all_iter_time = NULL;
    int iterations = perf_test_iterations, skips = perf_test_skips;
    MPI_Comm mpi_comm;
    struct sharp_coll_reduce_spec reduce_spec;
    void *s_mem_mr = NULL, *r_mem_mr = NULL;
    struct sharp_test_data_types_t* sharp_test_data_type;

    mpi_comm = sharp_comm->comm;
    sharp_test_data_type = sharp_coll_test_get_data_type(sharp_conf.datatype);

    mallopt(M_TRIM_THRESHOLD, -1);
    mallopt(M_MMAP_MAX, 0);

    max_count = sharp_conf.max_message_size / sharp_test_data_type->size;
    min_count = sharp_conf.min_message_size / sharp_test_data_type->size;

    MPI_Comm_size(mpi_comm, &comm_size);
    MPI_Comm_rank(mpi_comm, &rank);

    COLL_SET_ROOT(root, coll_root_rank, comm_size);

    if (rank == 0) {
        if (is_rooted) {
            fprintf(stdout, "\nReduce perf test. comm_size:%d reduce root:%d\n", comm_size, root);
        } else {
            fprintf(stdout, "\nAllreduce perf test. comm_size:%d\n", comm_size);
        }
        fprintf(stdout,
                "%15s %15s %15s %15s %15s %10s\n",
                "#size(bytes)",
                "Avg lat(us)",
                "Min lat(us)",
                "Max lat(us)",
                "Avg BW(Gb/s)",
                "iters");
    }

    inbuf = allocate_memory(sharp_conf.max_message_size, sharp_conf.s_mem_type, 1);
    if (!inbuf) {
        goto fn_fail;
    }

    outbuf = allocate_memory(sharp_conf.max_message_size, sharp_conf.r_mem_type, 0);
    if (!outbuf) {
        goto fn_fail;
    }

    iter_time = (double*)calloc(1, sizeof(double) * iterations);
    if (!iter_time) {
        fprintf(stdout, "Failed to allocate \n");
        goto fn_fail;
    }
    if (rank == 0) {
        all_iter_time = (double*)calloc(1, sizeof(double) * iterations * comm_size);
        if (!all_iter_time) {
            fprintf(stdout, "Failed to allocate \n");
            goto fn_fail;
        }
    }

    if (mlockall(MCL_FUTURE | MCL_CURRENT) < 0) {
        fprintf(stdout, "mlockall failed: %m\n");
        goto fn_fail;
    }

    if (enable_sharp_coll) {
        ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, inbuf, sharp_conf.max_message_size, &s_mem_mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "Failed to register source buffer\n");
            goto fn_fail;
        }

        ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, outbuf, sharp_conf.max_message_size, &r_mem_mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "Failed to register receive buffer\n");
            goto fn_fail;
        }
    }

    for (count = min_count; count <= max_count; count *= 2) {
        if (enable_sharp_coll) {
            reduce_spec.sbuf_desc.mem_type = sharp_conf.s_mem_type;
            reduce_spec.rbuf_desc.mem_type = sharp_conf.r_mem_type;

            if (sharp_conf.sdata_layout == TEST_DATA_LAYOUT_CONTIG) {
                reduce_spec.sbuf_desc.buffer.ptr = inbuf;
                reduce_spec.sbuf_desc.buffer.length = (count * sharp_test_data_type->size);
                reduce_spec.sbuf_desc.buffer.mem_handle = s_mem_mr;
                reduce_spec.sbuf_desc.type = SHARP_DATA_BUFFER;
            } else if (sharp_conf.sdata_layout == TEST_DATA_LAYOUT_IOV) {
                reduce_spec.sbuf_desc.iov.vector = NULL;
                create_iov_buffer(&reduce_spec.sbuf_desc,
                                  sharp_conf.siov_count,
                                  (char*)inbuf,
                                  (count * sharp_test_data_type->size),
                                  s_mem_mr);
            }

            if (!is_rooted || (rank == root)) {
                if (sharp_conf.rdata_layout == TEST_DATA_LAYOUT_CONTIG) {
                    reduce_spec.rbuf_desc.type = SHARP_DATA_BUFFER;
                    reduce_spec.rbuf_desc.buffer.ptr = outbuf;
                    reduce_spec.rbuf_desc.buffer.length = (count * sharp_test_data_type->size);
                    reduce_spec.rbuf_desc.buffer.mem_handle = r_mem_mr;
                } else if (sharp_conf.rdata_layout == TEST_DATA_LAYOUT_IOV) {
                    reduce_spec.rbuf_desc.iov.vector = NULL;
                    create_iov_buffer(&reduce_spec.rbuf_desc,
                                      sharp_conf.riov_count,
                                      (char*)outbuf,
                                      (count * sharp_test_data_type->size),
                                      r_mem_mr);
                }
            }

            reduce_spec.dtype = sharp_conf.datatype;
            reduce_spec.length = count;
            reduce_spec.op = SHARP_OP_SUM;
            reduce_spec.root = root;
            reduce_spec.aggr_mode = SHARP_AGGREGATION_NONE;
        }

        MPI_Barrier(mpi_comm);
        timer = 0.0;
        for (i = 0, j = 0; i < iterations + skips; i++) {
#if _DCHECK
            int pos, result;
            for (pos = 0; pos < count; pos++) {
                inbuf[pos] = i;
                outbuf[pos] = 0;
            }
#endif
            if (perf_with_barrier) {
                if (enable_sharp_coll) {
                    ret = sharp_coll_do_barrier(sharp_comm->sharp_coll_comm);
                    if (ret != SHARP_COLL_SUCCESS) {
                        if (rank == 0)
                            fprintf(stderr, "Barrier failed: %s\n", sharp_coll_strerror(ret));
                        exit(-1);
                    }
                } else {
                    MPI_Barrier(mpi_comm);
                }
            }

            t_start = sharp_time_sec();

            if (enable_sharp_coll) {
                if (!is_rooted) {
                    ret = sharp_coll_do_allreduce(sharp_comm->sharp_coll_comm, &reduce_spec);
                } else {
                    ret = sharp_coll_do_reduce(sharp_comm->sharp_coll_comm, &reduce_spec);
                }
                if (ret != SHARP_COLL_SUCCESS) {
                    if (rank == 0)
                        fprintf(stderr, "%s failed: %s\n", (is_rooted) ? "Reduce" : "Allreduce", sharp_coll_strerror(ret));
                    exit(-1);
                }
            } else {
                if (!is_rooted) {
                    MPI_Allreduce(inbuf, outbuf, count, sharp_test_data_type->mpi_id, MPI_SUM, mpi_comm);
                } else {
                    MPI_Reduce(inbuf, outbuf, count, sharp_test_data_type->mpi_id, MPI_SUM, root, mpi_comm);
                }
            }

            if (i >= skips) {
                t_stop = sharp_time_sec();
                timer += (t_stop - t_start);
                iter_time[j++] = (t_stop - t_start);
            }
#if _DCHECK
            for (pos = 0; pos < count; pos++) {
                result = i * comm_size;
                if (outbuf[pos] != result) {
                    fprintf(stdout, "data validation failed. At pos:%d got  =%d, expected:%d\n", pos, (int)outbuf[pos], result);
                    break;
                }
            }
#endif
        }

        latency = (double)(timer * 1e6) / iterations;

        min_iter_time = max_iter_time = iter_time[0];
        for (i = 0; i < iterations; i++) {
            if (min_iter_time > iter_time[i])
                min_iter_time = iter_time[i];
            if (max_iter_time < iter_time[i])
                max_iter_time = iter_time[i];
        }
        min_iter_time = (double)(min_iter_time * 1e6);
        max_iter_time = (double)(max_iter_time * 1e6);

        MPI_Reduce(&min_iter_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, mpi_comm);
        MPI_Reduce(&max_iter_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_comm);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

        avg_time = avg_time / comm_size;
        avg_bw = (count * sharp_test_data_type->size * 1.0 * 1000000) / (avg_time * 125000000);

        MPI_Gather(iter_time, iterations, MPI_DOUBLE, all_iter_time, iterations, MPI_DOUBLE, 0, mpi_comm);

        if (rank == 0) {
            fprintf(stdout,
                    "%15lu %15.2f %15.2f %15.2f %15.2f %10d\n",
                    (count * sharp_test_data_type->size),
                    avg_time,
                    min_time,
                    max_time,
                    avg_bw,
                    iterations);
#if 0
			print_coll_iterations_perf_data(all_iter_time, rank, comm_size, (int )(count * sharp_test_data_type->size), iterations);
#endif
            fflush(stdout);
        }
        if (enable_sharp_coll) {
            if (sharp_conf.sdata_layout == TEST_DATA_LAYOUT_IOV) {
                free(reduce_spec.sbuf_desc.iov.vector);
            }
            if (sharp_conf.rdata_layout == TEST_DATA_LAYOUT_IOV && (!is_rooted || (root == rank))) {
                free(reduce_spec.rbuf_desc.iov.vector);
            }
        }
    }

fn_fail:
    if (enable_sharp_coll) {
        if (s_mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, s_mem_mr);
        if (r_mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, r_mem_mr);
    }
    free_memory(inbuf, sharp_conf.s_mem_type, 1);
    free_memory(outbuf, sharp_conf.r_mem_type, 0);
    free(iter_time);
    free(all_iter_time);
}

void coll_test_sharp_iallreduce_perf(coll_sharp_module_t* sharp_comm)
{
    float *inbuf = NULL, *outbuf = NULL;
    int i, j, comm_size, rank, ret;
    size_t count, max_count, min_count;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0, timer = 0.0, avg_bw = 0.0;
    double *iter_time = NULL, max_iter_time, min_iter_time;
    double* all_iter_time = NULL;
    int iterations = perf_test_iterations, skips = perf_test_skips;
    MPI_Comm mpi_comm;
    MPI_Request mpi_reqs[MAX_NB_REQS];
    MPI_Status mpi_status;
    void* sharp_reqs[MAX_NB_REQS];
    struct sharp_coll_reduce_spec reduce_spec;
    void *s_mem_mr = NULL, *r_mem_mr = NULL;
    int outstanding_non_blocking_ops = nbc_count;
    int nbc;
    size_t buffer_size;
    struct sharp_test_data_types_t* sharp_test_data_type;

    mpi_comm = sharp_comm->comm;
    sharp_test_data_type = sharp_coll_test_get_data_type(sharp_conf.datatype);

    mallopt(M_TRIM_THRESHOLD, -1);
    mallopt(M_MMAP_MAX, 0);

    max_count = sharp_conf.max_message_size / sharp_test_data_type->size;
    min_count = sharp_conf.min_message_size / sharp_test_data_type->size;
    buffer_size = sharp_conf.max_message_size * outstanding_non_blocking_ops;

    MPI_Comm_size(mpi_comm, &comm_size);
    MPI_Comm_rank(mpi_comm, &rank);

    if (rank == 0) {
        fprintf(stdout, "\nNon-Blocking Allreduce perf test. comm_size:%d #num_nonblock_ops:%d\n", comm_size, outstanding_non_blocking_ops);
        fprintf(stdout,
                "%15s %15s %15s %15s %15s %10s\n",
                "#size(bytes)",
                "Avg lat(us)",
                "Min lat(us)",
                "Max lat(us)",
                "Avg BW(Gb/s)",
                "iters");
    }

    inbuf = allocate_memory(buffer_size, sharp_conf.s_mem_type, 1);
    if (!inbuf) {
        goto fn_fail;
    }

    outbuf = allocate_memory(buffer_size, sharp_conf.r_mem_type, 0);
    if (!outbuf) {
        goto fn_fail;
    }

    iter_time = (double*)calloc(1, sizeof(double) * iterations);
    if (!iter_time) {
        fprintf(stdout, "Failed to allocate \n");
        goto fn_fail;
    }
    if (rank == 0) {
        all_iter_time = (double*)calloc(1, sizeof(double) * iterations * comm_size);
        if (!all_iter_time) {
            fprintf(stdout, "Failed to allocate \n");
            goto fn_fail;
        }
    }

    if (mlockall(MCL_FUTURE | MCL_CURRENT) < 0) {
        fprintf(stdout, "mlockall failed: %m\n");
    }

    if (enable_sharp_coll) {
        ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, inbuf, buffer_size, &s_mem_mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "Failed to register source buffer\n");
            goto fn_fail;
        }

        ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, outbuf, buffer_size, &r_mem_mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "Failed to register receive buffer\n");
            goto fn_fail;
        }
    }

    for (count = min_count; count <= max_count; count *= 2) {
        MPI_Barrier(mpi_comm);
        timer = 0.0;
        for (i = 0, j = 0; i < iterations + skips; i++) {
#if _DCHECK
            int pos, result;
            for (pos = 0; pos < count; pos++) {
                inbuf[pos] = i;
                outbuf[pos] = 0;
            }
#endif
            if (perf_with_barrier) {
                if (enable_sharp_coll) {
                    ret = sharp_coll_do_barrier_nb(sharp_comm->sharp_coll_comm, &sharp_reqs[0]);
                    if (ret != SHARP_COLL_SUCCESS) {
                        if (rank == 0)
                            fprintf(stderr, "Barrier failed: %s\n", sharp_coll_strerror(ret));
                        exit(-1);
                    }
                    ret = sharp_coll_req_wait(sharp_reqs[0]);
                    if (ret != SHARP_COLL_SUCCESS) {
                        fprintf(stderr, "sharp_coll_req_wait failed \n");
                        exit(-1);
                    }
                } else {
                    MPI_Ibarrier(mpi_comm, &mpi_reqs[0]);
                    MPI_Wait(&mpi_reqs[0], &mpi_status);
                }
            }

            t_start = sharp_time_sec();

            for (nbc = 0; nbc < outstanding_non_blocking_ops; nbc++) {
                if (enable_sharp_coll) {
                    reduce_spec.sbuf_desc.buffer.ptr = inbuf + (nbc * count);
                    reduce_spec.sbuf_desc.buffer.length = (count * sharp_test_data_type->size);
                    reduce_spec.sbuf_desc.buffer.mem_handle = s_mem_mr;
                    reduce_spec.sbuf_desc.type = SHARP_DATA_BUFFER;
                    reduce_spec.sbuf_desc.mem_type = sharp_conf.s_mem_type;
                    reduce_spec.rbuf_desc.buffer.ptr = outbuf + (nbc * count);
                    reduce_spec.rbuf_desc.buffer.length = (count * sharp_test_data_type->size);
                    reduce_spec.rbuf_desc.buffer.mem_handle = r_mem_mr;
                    reduce_spec.rbuf_desc.mem_type = sharp_conf.r_mem_type;
                    reduce_spec.rbuf_desc.type = SHARP_DATA_BUFFER;
                    reduce_spec.dtype = sharp_conf.datatype;
                    reduce_spec.length = count;
                    reduce_spec.op = SHARP_OP_SUM;
                    reduce_spec.aggr_mode = SHARP_AGGREGATION_NONE;

                    ret = sharp_coll_do_allreduce_nb(sharp_comm->sharp_coll_comm, &reduce_spec, &sharp_reqs[nbc]);
                    if (ret != SHARP_COLL_SUCCESS) {
                        if (rank == 0)
                            fprintf(stderr, "Allreduce failed: %s\n", sharp_coll_strerror(ret));
                        exit(-1);
                    }
                } else {
                    MPI_Iallreduce(inbuf + (nbc * count),
                                   outbuf + (nbc * count),
                                   count,
                                   sharp_test_data_type->mpi_id,
                                   MPI_SUM,
                                   mpi_comm,
                                   &mpi_reqs[nbc]);
                }
            }

            for (nbc = 0; nbc < outstanding_non_blocking_ops; nbc++) {
                if (enable_sharp_coll) {
                    ret = sharp_coll_req_wait(sharp_reqs[nbc]);
                    if (ret != SHARP_COLL_SUCCESS) {
                        fprintf(stderr, "sharp_coll_req_wait failed \n");
                        exit(-1);
                    }
                } else {
                    MPI_Wait(&mpi_reqs[nbc], &mpi_status);
                }
            }

            if (i >= skips) {
                t_stop = sharp_time_sec();
                timer += (t_stop - t_start);
                iter_time[j++] = (t_stop - t_start);
            }
#if _DCHECK
            for (pos = 0; pos < count; pos++) {
                result = i * comm_size;
                if (outbuf[pos] != result) {
                    fprintf(stdout, "data validation failed. At pos:%d got  =%d, expected:%d\n", pos, (int)outbuf[pos], result);
                    break;
                }
            }
#endif
        }

        latency = (double)(timer * 1e6) / iterations;

        min_iter_time = max_iter_time = iter_time[0];
        for (i = 0; i < iterations; i++) {
            if (min_iter_time > iter_time[i])
                min_iter_time = iter_time[i];
            if (max_iter_time < iter_time[i])
                max_iter_time = iter_time[i];
        }
        min_iter_time = (double)(min_iter_time * 1e6);
        max_iter_time = (double)(max_iter_time * 1e6);

        MPI_Reduce(&min_iter_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, mpi_comm);
        MPI_Reduce(&max_iter_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_comm);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

        avg_time = avg_time / comm_size;
        avg_bw = (1.0 * outstanding_non_blocking_ops * count * sharp_test_data_type->size * 1000000) / (avg_time * 125000000);

        MPI_Gather(iter_time, iterations, MPI_DOUBLE, all_iter_time, iterations, MPI_DOUBLE, 0, mpi_comm);
        if (rank == 0) {
            fprintf(stdout,
                    "%15lu %15.2f %15.2f %15.2f %15.2f %10d\n",
                    (count * sharp_test_data_type->size),
                    avg_time,
                    min_time,
                    max_time,
                    avg_bw,
                    iterations);
#if 0
			print_coll_iterations_perf_data(all_iter_time, rank, comm_size, (int )(count * sizeof(int)), iterations);
#endif
            fflush(stdout);
        }
    }

fn_fail:
    if (enable_sharp_coll) {
        if (s_mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, s_mem_mr);
        if (r_mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, r_mem_mr);
    }
    free_memory(inbuf, sharp_conf.s_mem_type, 1);
    free_memory(outbuf, sharp_conf.r_mem_type, 0);
    free(iter_time);
    free(all_iter_time);
}

void coll_test_sharp_reduce_scatter_basic(coll_sharp_module_t* sharp_comm)
{
    int *inbuf = NULL, *outbuf = NULL;
    int comm_size, rank, ret, chunk, num_chunks = 1;
    size_t count, chunk_size, offset;
    struct sharp_coll_reduce_spec reduce_spec;
    MPI_Comm mpi_comm;
    void *s_mem_mr = NULL, *r_mem_mr = NULL;
    void** handles;

    mpi_comm = sharp_comm->comm;
    MPI_Comm_size(mpi_comm, &comm_size);
    MPI_Comm_rank(mpi_comm, &rank);

    count = sharp_conf.max_message_size / sizeof(int);

    if (getenv("NUM_REDUCE_SCATTER_CHUNKS")) {
        num_chunks = atoi(getenv("NUM_REDUCE_SCATTER_CHUNKS"));
    }

    handles = malloc(num_chunks * sizeof(void*));

    if (rank == 0) {
        printf(" Reduce Scatter Basic test (group_size:%d count:%lu op:%s) - ", comm_size, count, "SUM");
    }

    inbuf = allocate_memory(sharp_conf.max_message_size * comm_size, sharp_conf.s_mem_type, 1);
    if (!inbuf) {
        goto fn_fail;
    }

    outbuf = allocate_memory(sharp_conf.max_message_size, sharp_conf.r_mem_type, 0);
    if (!outbuf) {
        goto fn_fail;
    }

    if (enable_sharp_coll) {
        ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, inbuf, sharp_conf.max_message_size * comm_size, &s_mem_mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "Failed to register source buffer\n");
            goto fn_fail;
        }

        ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, outbuf, sharp_conf.max_message_size, &r_mem_mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "Failed to register receive  buffer\n");
            goto fn_fail;
        }
    }

    fill_int_pattren(inbuf, rank, count * comm_size, sharp_conf.s_mem_type);

    if (enable_sharp_coll) {
        for (chunk = 0; chunk < num_chunks; chunk++) {
            chunk_size = (count * sizeof(int) * comm_size) / num_chunks;
            offset = chunk * chunk_size;

            reduce_spec.sbuf_desc.mem_type = sharp_conf.s_mem_type;
            reduce_spec.rbuf_desc.mem_type = sharp_conf.r_mem_type;

            assert(sharp_conf.sdata_layout == TEST_DATA_LAYOUT_CONTIG && sharp_conf.rdata_layout == TEST_DATA_LAYOUT_CONTIG);
            reduce_spec.sbuf_desc.type = SHARP_DATA_BUFFER;
            reduce_spec.sbuf_desc.buffer.ptr = (void*)inbuf + offset;
            reduce_spec.sbuf_desc.buffer.length = chunk_size;
            reduce_spec.sbuf_desc.buffer.mem_handle = s_mem_mr;

            reduce_spec.rbuf_desc.type = SHARP_DATA_BUFFER;
            reduce_spec.rbuf_desc.buffer.ptr = (void*)outbuf + (offset % (count * sizeof(int)));
            ;
            reduce_spec.rbuf_desc.buffer.length = (count * sizeof(int));
            reduce_spec.rbuf_desc.buffer.mem_handle = r_mem_mr;

            reduce_spec.dtype = SHARP_DTYPE_INT;
            reduce_spec.length = count;
            reduce_spec.op = SHARP_OP_SUM;
            reduce_spec.aggr_mode = SHARP_AGGREGATION_NONE;
            reduce_spec.offset = offset;

            ret = sharp_coll_do_reduce_scatter_nb(sharp_comm->sharp_coll_comm, &reduce_spec, &handles[chunk]);
            if (ret != SHARP_COLL_SUCCESS) {
                if (rank == 0) {
                    fprintf(stderr, "%s failed: %s\n", "Reduce-scatter", sharp_coll_strerror(ret));
                }
                exit(-1);
            }
        }
    } else {
        int recv_count[comm_size], i;
        for (i = 0; i < comm_size; i++) {
            recv_count[i] = count;
        }

        MPI_Reduce_scatter(inbuf, outbuf, recv_count, MPI_INT, MPI_SUM, mpi_comm);
    }

    for (chunk = 0; chunk < num_chunks; chunk++) {
        sharp_coll_req_wait(handles[chunk]);
    }

    check_int_pattren(outbuf, rank, comm_size, count, sharp_conf.r_mem_type, TEST_REDUCE_SCATTER);
    if (rank == 0) {
        fprintf(stdout, " PASS\n");
    }

fn_fail:
    if (enable_sharp_coll) {
        if (s_mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, s_mem_mr);
        if (r_mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, r_mem_mr);
    }

    free(handles);

    free_memory(inbuf, sharp_conf.s_mem_type, 1);
    free_memory(outbuf, sharp_conf.r_mem_type, 0);
}

void coll_test_sharp_reduce_scatter_perf(coll_sharp_module_t* sharp_comm)
{
    float *inbuf = NULL, *outbuf = NULL;
    size_t count, max_count, min_count;
    int i, j, comm_size, rank, ret;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0, timer = 0.0, avg_bw = 0.0;
    double *iter_time = NULL, max_iter_time, min_iter_time;
    double* all_iter_time = NULL;
    int iterations = perf_test_iterations, skips = perf_test_skips;
    MPI_Comm mpi_comm;
    struct sharp_coll_reduce_spec reduce_spec;
    void *s_mem_mr = NULL, *r_mem_mr = NULL;
    struct sharp_test_data_types_t* sharp_test_data_type;

    mpi_comm = sharp_comm->comm;
    sharp_test_data_type = sharp_coll_test_get_data_type(sharp_conf.datatype);

    mallopt(M_TRIM_THRESHOLD, -1);
    mallopt(M_MMAP_MAX, 0);

    max_count = sharp_conf.max_message_size / sharp_test_data_type->size;
    min_count = sharp_conf.min_message_size / sharp_test_data_type->size;

    MPI_Comm_size(mpi_comm, &comm_size);
    MPI_Comm_rank(mpi_comm, &rank);

    if (rank == 0) {
        fprintf(stdout, "\nReduce-scatter perf test. comm_size:%d\n", comm_size);
        fprintf(stdout,
                "%15s %15s %15s %15s %15s %10s\n",
                "#size(bytes)",
                "Avg lat(us)",
                "Min lat(us)",
                "Max lat(us)",
                "Avg BW(Gb/s)",
                "iters");
    }

    inbuf = allocate_memory(sharp_conf.max_message_size * comm_size, sharp_conf.s_mem_type, 1);
    if (!inbuf) {
        goto fn_fail;
    }

    outbuf = allocate_memory(sharp_conf.max_message_size, sharp_conf.r_mem_type, 0);
    if (!outbuf) {
        goto fn_fail;
    }

    iter_time = (double*)calloc(1, sizeof(double) * iterations);
    if (!iter_time) {
        fprintf(stdout, "Failed to allocate \n");
        goto fn_fail;
    }
    if (rank == 0) {
        all_iter_time = (double*)calloc(1, sizeof(double) * iterations * comm_size);
        if (!all_iter_time) {
            fprintf(stdout, "Failed to allocate \n");
            goto fn_fail;
        }
    }

    if (mlockall(MCL_FUTURE | MCL_CURRENT) < 0) {
        fprintf(stdout, "mlockall failed: %m\n");
        goto fn_fail;
    }

    if (enable_sharp_coll) {
        ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, inbuf, sharp_conf.max_message_size * comm_size, &s_mem_mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "Failed to register source buffer\n");
            goto fn_fail;
        }

        ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, outbuf, sharp_conf.max_message_size, &r_mem_mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "Failed to register receive buffer\n");
            goto fn_fail;
        }
    }

    for (count = min_count; count <= max_count; count *= 2) {
        if (enable_sharp_coll) {
            reduce_spec.sbuf_desc.mem_type = sharp_conf.s_mem_type;
            reduce_spec.rbuf_desc.mem_type = sharp_conf.r_mem_type;

            assert(sharp_conf.sdata_layout == TEST_DATA_LAYOUT_CONTIG && sharp_conf.rdata_layout == TEST_DATA_LAYOUT_CONTIG);
            reduce_spec.sbuf_desc.buffer.ptr = inbuf;
            reduce_spec.sbuf_desc.buffer.length = (count * sharp_test_data_type->size * comm_size);
            reduce_spec.sbuf_desc.buffer.mem_handle = s_mem_mr;
            reduce_spec.sbuf_desc.type = SHARP_DATA_BUFFER;

            reduce_spec.rbuf_desc.type = SHARP_DATA_BUFFER;
            reduce_spec.rbuf_desc.buffer.ptr = outbuf;
            reduce_spec.rbuf_desc.buffer.length = (count * sharp_test_data_type->size);
            reduce_spec.rbuf_desc.buffer.mem_handle = r_mem_mr;

            reduce_spec.dtype = sharp_conf.datatype;
            reduce_spec.length = count;
            reduce_spec.op = SHARP_OP_SUM;
            reduce_spec.root = -1;
            reduce_spec.aggr_mode = SHARP_AGGREGATION_NONE;
            reduce_spec.offset = 0;
        }

        MPI_Barrier(mpi_comm);
        timer = 0.0;
        for (i = 0, j = 0; i < iterations + skips; i++) {
#if _DCHECK
            int pos, result;
            for (pos = 0; pos < count; pos++) {
                inbuf[pos] = i;
                outbuf[pos] = 0;
            }
#endif
            if (perf_with_barrier) {
                if (enable_sharp_coll) {
                    ret = sharp_coll_do_barrier(sharp_comm->sharp_coll_comm);
                    if (ret != SHARP_COLL_SUCCESS) {
                        if (rank == 0)
                            fprintf(stderr, "Barrier failed: %s\n", sharp_coll_strerror(ret));
                        exit(-1);
                    }
                } else {
                    MPI_Barrier(mpi_comm);
                }
            }

            t_start = sharp_time_sec();

            if (enable_sharp_coll) {
                ret = sharp_coll_do_reduce_scatter(sharp_comm->sharp_coll_comm, &reduce_spec);
                if (ret != SHARP_COLL_SUCCESS) {
                    if (rank == 0)
                        fprintf(stderr, "Reduce-scatter failed: %s\n", sharp_coll_strerror(ret));
                    exit(-1);
                }
            } else {
                int recv_count[comm_size], i;
                for (i = 0; i < comm_size; i++) {
                    recv_count[i] = count;
                }
                MPI_Reduce_scatter(inbuf, outbuf, recv_count, sharp_test_data_type->mpi_id, MPI_SUM, mpi_comm);
            }

            if (i >= skips) {
                t_stop = sharp_time_sec();
                timer += (t_stop - t_start);
                iter_time[j++] = (t_stop - t_start);
            }
#if _DCHECK
            for (pos = 0; pos < count; pos++) {
                result = i * comm_size;
                if (outbuf[pos] != result) {
                    fprintf(stdout, "data validation failed. At pos:%d got  =%d, expected:%d\n", pos, (int)outbuf[pos], result);
                    break;
                }
            }
#endif
        }

        latency = (double)(timer * 1e6) / iterations;

        min_iter_time = max_iter_time = iter_time[0];
        for (i = 0; i < iterations; i++) {
            if (min_iter_time > iter_time[i])
                min_iter_time = iter_time[i];
            if (max_iter_time < iter_time[i])
                max_iter_time = iter_time[i];
        }
        min_iter_time = (double)(min_iter_time * 1e6);
        max_iter_time = (double)(max_iter_time * 1e6);

        MPI_Reduce(&min_iter_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, mpi_comm);
        MPI_Reduce(&max_iter_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_comm);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

        avg_time = avg_time / comm_size;
        avg_bw = (comm_size * count * sharp_test_data_type->size * 1.0 * 1000000) / (avg_time * 125000000);

        MPI_Gather(iter_time, iterations, MPI_DOUBLE, all_iter_time, iterations, MPI_DOUBLE, 0, mpi_comm);

        if (rank == 0) {
            fprintf(stdout,
                    "%15lu %15.2f %15.2f %15.2f %15.2f %10d\n",
                    (comm_size * count * sharp_test_data_type->size),
                    avg_time,
                    min_time,
                    max_time,
                    avg_bw,
                    iterations);
#if 0
			print_coll_iterations_perf_data(all_iter_time, rank, comm_size, (int )(count * sharp_test_data_type->size), iterations);
#endif
            fflush(stdout);
        }
    }

fn_fail:
    if (enable_sharp_coll) {
        if (s_mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, s_mem_mr);
        if (r_mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, r_mem_mr);
    }
    free_memory(inbuf, sharp_conf.s_mem_type, 1);
    free_memory(outbuf, sharp_conf.r_mem_type, 0);
    free(iter_time);
    free(all_iter_time);
}

void coll_test_sharp_bcast_basic(coll_sharp_module_t* sharp_comm)
{
    int* buf = NULL;
    int comm_size, rank, root, ret;
    size_t count;
    struct sharp_coll_bcast_spec bcast_spec;
    MPI_Comm mpi_comm;
    void* mem_mr = NULL;

    mpi_comm = sharp_comm->comm;
    count = sharp_conf.max_message_size / sizeof(int);

    MPI_Comm_size(mpi_comm, &comm_size);
    MPI_Comm_rank(mpi_comm, &rank);

    COLL_SET_ROOT(root, coll_root_rank, comm_size);

    if (rank == 0) {
        printf("BCAST Basic test (group_size:%d count:%lu op:%s root:%d) \n", comm_size, count, "SUM", root);
    }

    buf = allocate_memory(sharp_conf.max_message_size, sharp_conf.s_mem_type, 1);
    if (!buf) {
        goto fn_fail;
    }

    if (enable_sharp_coll) {
        ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, buf, sharp_conf.max_message_size, &mem_mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "Failed to register source buffer\n");
            goto fn_fail;
        }
    }

    fill_int_pattren(buf, root - 1, count, sharp_conf.s_mem_type);

    if (enable_sharp_coll) {
        bcast_spec.buf_desc.mem_type = sharp_conf.s_mem_type;

        if (sharp_conf.sdata_layout == TEST_DATA_LAYOUT_CONTIG) {
            bcast_spec.buf_desc.type = SHARP_DATA_BUFFER;
            bcast_spec.buf_desc.buffer.ptr = buf;
            bcast_spec.buf_desc.buffer.length = (count * sizeof(int));
            bcast_spec.buf_desc.buffer.mem_handle = mem_mr;
        } else if (sharp_conf.sdata_layout == TEST_DATA_LAYOUT_IOV) {
            create_iov_buffer(&bcast_spec.buf_desc, sharp_conf.siov_count, (char*)buf, (count * sizeof(int)), mem_mr);
        }

        bcast_spec.size = (count * sizeof(int));
        bcast_spec.root = root;

        ret = sharp_coll_do_bcast(sharp_comm->sharp_coll_comm, &bcast_spec);
        if (ret != SHARP_COLL_SUCCESS) {
            if (rank == 0) {
                fprintf(stderr, "Bcast failed: %s\n", sharp_coll_strerror(ret));
            }
            exit(-1);
        }
    } else {
        MPI_Bcast(buf, count, MPI_INT, root, mpi_comm);
    }

    check_int_pattren(buf, rank, root, count, sharp_conf.r_mem_type, TEST_BCAST);
    if (root == rank) {
        fprintf(stdout, " PASS\n");
    }

fn_fail:
    if (sharp_conf.sdata_layout == TEST_DATA_LAYOUT_IOV) {
        free(bcast_spec.buf_desc.iov.vector);
    }

    if (enable_sharp_coll) {
        if (mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, mem_mr);
    }

    free_memory(buf, sharp_conf.s_mem_type, 1);
}

void coll_test_sharp_bcast_perf(coll_sharp_module_t* sharp_comm)
{
    int* buf = NULL;
    int i, j, comm_size, rank, ret, root;
    size_t count, max_count, min_count;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0, timer = 0.0, avg_bw = 0.0;
    double *iter_time = NULL, max_iter_time, min_iter_time;
    double* all_iter_time = NULL;
    int iterations = perf_test_iterations, skips = perf_test_skips;
    MPI_Comm mpi_comm;
    struct sharp_coll_bcast_spec bcast_spec;
    void* mem_mr = NULL;

    mpi_comm = sharp_comm->comm;

    mallopt(M_TRIM_THRESHOLD, -1);
    mallopt(M_MMAP_MAX, 0);

    max_count = sharp_conf.max_message_size / sizeof(float);
    min_count = sharp_conf.min_message_size / sizeof(float);

    MPI_Comm_size(mpi_comm, &comm_size);
    MPI_Comm_rank(mpi_comm, &rank);

    root = coll_root_rank;

    if (rank == 0) {
        fprintf(stdout, "\nBcast perf test. comm_size:%d bcast root:%d\n", comm_size, root);
        fprintf(stdout,
                "%15s %15s %15s %15s %15s %10s\n",
                "#size(bytes)",
                "Avg lat(us)",
                "Min lat(us)",
                "Max lat(us)",
                "Avg BW(Gb/s)",
                "iters");
    }

    buf = allocate_memory(sharp_conf.max_message_size, sharp_conf.s_mem_type, 1);
    if (!buf) {
        goto fn_fail;
    }

    iter_time = (double*)calloc(1, sizeof(double) * iterations);
    if (!iter_time) {
        fprintf(stdout, "Failed to allocate \n");
        goto fn_fail;
    }
    if (rank == 0) {
        all_iter_time = (double*)calloc(1, sizeof(double) * iterations * comm_size);
        if (!all_iter_time) {
            fprintf(stdout, "Failed to allocate \n");
            goto fn_fail;
        }
    }

    if (mlockall(MCL_FUTURE | MCL_CURRENT) < 0) {
        fprintf(stdout, "mlockall failed: %m\n");
        goto fn_fail;
    }

    if (enable_sharp_coll) {
        ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, buf, sharp_conf.max_message_size, &mem_mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "Failed to register source buffer\n");
            goto fn_fail;
        }
    }

    for (count = min_count; count <= max_count; count *= 2) {
        if (enable_sharp_coll) {
            bcast_spec.buf_desc.mem_type = sharp_conf.s_mem_type;

            if (sharp_conf.sdata_layout == TEST_DATA_LAYOUT_CONTIG) {
                bcast_spec.buf_desc.type = SHARP_DATA_BUFFER;
                bcast_spec.buf_desc.buffer.ptr = buf;
                bcast_spec.buf_desc.buffer.length = (count * sizeof(int));
                bcast_spec.buf_desc.buffer.mem_handle = mem_mr;
            } else if (sharp_conf.sdata_layout == TEST_DATA_LAYOUT_IOV) {
                bcast_spec.buf_desc.iov.vector = NULL;
                create_iov_buffer(&bcast_spec.buf_desc, sharp_conf.siov_count, (char*)buf, (count * sizeof(int)), mem_mr);
            }

            bcast_spec.size = (count * sizeof(int));
            bcast_spec.root = root;
        }

        MPI_Barrier(mpi_comm);
        timer = 0.0;
        for (i = 0, j = 0; i < iterations + skips; i++) {
            if (coll_root_rank == -1) {
                bcast_spec.root = (i % comm_size);
            }
#if _DCHECK
            {
                int pos;
                for (pos = 0; pos < count; pos++) {
                    buf[pos] = (rank == root) ? i : 0;
                }
            }
#endif
            if (perf_with_barrier) {
                if (enable_sharp_coll) {
                    ret = sharp_coll_do_barrier(sharp_comm->sharp_coll_comm);
                    if (ret != SHARP_COLL_SUCCESS) {
                        if (rank == 0)
                            fprintf(stderr, "Barrier failed: %s\n", sharp_coll_strerror(ret));
                        exit(-1);
                    }
                } else {
                    MPI_Barrier(mpi_comm);
                }
            }

            t_start = sharp_time_sec();

            if (enable_sharp_coll) {
                ret = sharp_coll_do_bcast(sharp_comm->sharp_coll_comm, &bcast_spec);
                if (ret != SHARP_COLL_SUCCESS) {
                    if (rank == 0)
                        fprintf(stderr, "Bcast failed: %s\n", sharp_coll_strerror(ret));
                    exit(-1);
                }
            } else {
                MPI_Bcast(buf, count, MPI_INT, root, mpi_comm);
            }

            if (i >= skips) {
                t_stop = sharp_time_sec();
                timer += (t_stop - t_start);
                iter_time[j++] = (t_stop - t_start);
            }
#if _DCHECK
            for (pos = 0; pos < count; pos++) {
                if (buf[pos] != i) {
                    fprintf(stdout, "data validation failed. At pos:%d got=%d, expected:%d\n", pos, buf[pos], i);
                    break;
                }
            }
#endif
        }

        latency = (double)(timer * 1e6) / iterations;

        min_iter_time = max_iter_time = iter_time[0];
        for (i = 0; i < iterations; i++) {
            if (min_iter_time > iter_time[i])
                min_iter_time = iter_time[i];
            if (max_iter_time < iter_time[i])
                max_iter_time = iter_time[i];
        }
        min_iter_time = (double)(min_iter_time * 1e6);
        max_iter_time = (double)(max_iter_time * 1e6);

        MPI_Reduce(&min_iter_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, mpi_comm);
        MPI_Reduce(&max_iter_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_comm);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

        avg_time = avg_time / comm_size;
        avg_bw = (count * sizeof(int) * 1.0 * 1000000) / (avg_time * 125000000);

        MPI_Gather(iter_time, iterations, MPI_DOUBLE, all_iter_time, iterations, MPI_DOUBLE, 0, mpi_comm);

        if (rank == 0) {
            fprintf(stdout,
                    "%15lu %15.2f %15.2f %15.2f %15.2f %10d\n",
                    (count * sizeof(int)),
                    avg_time,
                    min_time,
                    max_time,
                    avg_bw,
                    iterations);
#if 0
			print_coll_iterations_perf_data(all_iter_time, rank, comm_size, (int )(count * sizeof(int)), iterations);
#endif
            fflush(stdout);
        }
        if (enable_sharp_coll) {
            if (sharp_conf.sdata_layout == TEST_DATA_LAYOUT_IOV) {
                free(bcast_spec.buf_desc.iov.vector);
            }
        }
    }

fn_fail:
    if (enable_sharp_coll) {
        if (mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, mem_mr);
    }
    free_memory(buf, sharp_conf.s_mem_type, 1);
    free(iter_time);
    free(all_iter_time);
}

void coll_test_sharp_allgather_basic(coll_sharp_module_t* sharp_comm)
{
    int *inbuf = NULL, *outbuf = NULL;
    int comm_size, rank, ret, chunk, num_chunks = 1;
    size_t count, chunk_size, offset;
    struct sharp_coll_gather_spec gather_spec;
    MPI_Comm mpi_comm;
    void *s_mem_mr = NULL, *r_mem_mr = NULL;

    mpi_comm = sharp_comm->comm;
    MPI_Comm_size(mpi_comm, &comm_size);
    MPI_Comm_rank(mpi_comm, &rank);

    count = sharp_conf.max_message_size / sizeof(int);

    if (getenv("NUM_ALLGATHER_CHUNKS")) {
        num_chunks = atoi(getenv("NUM_ALLGATHER_CHUNKS"));
    }

    if (rank == 0) {
        printf("Allgather Basic test (group_size:%d count:%lu) - ", comm_size, count);
    }

    inbuf = allocate_memory(sharp_conf.max_message_size, sharp_conf.s_mem_type, 1);
    if (!inbuf) {
        goto fn_fail;
    }

    outbuf = allocate_memory(sharp_conf.max_message_size * comm_size, sharp_conf.r_mem_type, 0);
    if (!outbuf) {
        goto fn_fail;
    }

    if (enable_sharp_coll) {
        ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, inbuf, sharp_conf.max_message_size, &s_mem_mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "Failed to register source buffer\n");
            goto fn_fail;
        }

        ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, outbuf, sharp_conf.max_message_size * comm_size, &r_mem_mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "Failed to register receive  buffer\n");
            goto fn_fail;
        }
    }

    fill_int_pattren(inbuf, rank, count, sharp_conf.s_mem_type);

    if (enable_sharp_coll) {
        for (chunk = 0; chunk < num_chunks; chunk++) {
            chunk_size = (count * sizeof(int) * comm_size) / num_chunks;
            offset = chunk * chunk_size;
            gather_spec.sbuf_desc.mem_type = sharp_conf.s_mem_type;
            gather_spec.rbuf_desc.mem_type = sharp_conf.r_mem_type;

            assert(sharp_conf.sdata_layout == TEST_DATA_LAYOUT_CONTIG && sharp_conf.rdata_layout == TEST_DATA_LAYOUT_CONTIG);
            gather_spec.sbuf_desc.type = SHARP_DATA_BUFFER;
            gather_spec.sbuf_desc.buffer.ptr = (void*)inbuf + (offset % (count * sizeof(int)));
            gather_spec.sbuf_desc.buffer.length = (count * sizeof(int));
            gather_spec.sbuf_desc.buffer.mem_handle = s_mem_mr;

            gather_spec.rbuf_desc.type = SHARP_DATA_BUFFER;
            gather_spec.rbuf_desc.buffer.ptr = (void*)outbuf + offset;
            gather_spec.rbuf_desc.buffer.length = chunk_size;
            gather_spec.rbuf_desc.buffer.mem_handle = r_mem_mr;

            gather_spec.dtype = SHARP_DTYPE_INT;
            gather_spec.size = count / chunk_size;
            gather_spec.offset = offset;

            ret = sharp_coll_do_allgather(sharp_comm->sharp_coll_comm, &gather_spec);
            if (ret != SHARP_COLL_SUCCESS) {
                if (rank == 0) {
                    fprintf(stderr, "%s failed: %s\n", "Allgather", sharp_coll_strerror(ret));
                }
                exit(-1);
            }
        }
    } else {
        MPI_Allgather(inbuf, count, MPI_INT, outbuf, count, MPI_INT, mpi_comm);
    }

    check_int_pattren(outbuf, rank, comm_size, count * comm_size, sharp_conf.r_mem_type, TEST_ALLGATHER);
    if (rank == 0) {
        fprintf(stdout, " PASS\n");
    }

fn_fail:
    if (enable_sharp_coll) {
        if (s_mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, s_mem_mr);
        if (r_mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, r_mem_mr);
    }

    free_memory(inbuf, sharp_conf.s_mem_type, 1);
    free_memory(outbuf, sharp_conf.r_mem_type, 0);
}

void coll_test_sharp_allgather_perf(coll_sharp_module_t* sharp_comm)
{
    float *inbuf = NULL, *outbuf = NULL;
    size_t count, max_count, min_count;
    int i, j, comm_size, rank, ret;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0, timer = 0.0, avg_bw = 0.0;
    double *iter_time = NULL, max_iter_time, min_iter_time;
    double* all_iter_time = NULL;
    int iterations = perf_test_iterations, skips = perf_test_skips;
    MPI_Comm mpi_comm;
    struct sharp_coll_gather_spec gather_spec;
    void *s_mem_mr = NULL, *r_mem_mr = NULL;
    struct sharp_test_data_types_t* sharp_test_data_type;

    mpi_comm = sharp_comm->comm;
    sharp_test_data_type = sharp_coll_test_get_data_type(sharp_conf.datatype);

    mallopt(M_TRIM_THRESHOLD, -1);
    mallopt(M_MMAP_MAX, 0);

    MPI_Comm_size(mpi_comm, &comm_size);
    MPI_Comm_rank(mpi_comm, &rank);

    if (sharp_conf.max_message_size * comm_size > 536870912) {
        sharp_conf.max_message_size = 536870912L / comm_size;
    }

    max_count = sharp_conf.max_message_size / sharp_test_data_type->size;
    min_count = sharp_conf.min_message_size / sharp_test_data_type->size;

    if (rank == 0) {
        fprintf(stdout, "\nAllgather perf test. comm_size:%d\n", comm_size);
        fprintf(stdout,
                "%15s %15s %15s %15s %15s %10s\n",
                "#size(bytes)",
                "Avg lat(us)",
                "Min lat(us)",
                "Max lat(us)",
                "Avg BW(Gb/s)",
                "iters");
    }

    inbuf = allocate_memory(sharp_conf.max_message_size, sharp_conf.s_mem_type, 1);
    if (!inbuf) {
        goto fn_fail;
    }

    outbuf = allocate_memory(sharp_conf.max_message_size * comm_size, sharp_conf.r_mem_type, 0);
    if (!outbuf) {
        goto fn_fail;
    }

    iter_time = (double*)calloc(1, sizeof(double) * iterations);
    if (!iter_time) {
        fprintf(stdout, "Failed to allocate \n");
        goto fn_fail;
    }
    if (rank == 0) {
        all_iter_time = (double*)calloc(1, sizeof(double) * iterations * comm_size);
        if (!all_iter_time) {
            fprintf(stdout, "Failed to allocate \n");
            goto fn_fail;
        }
    }

    if (mlockall(MCL_FUTURE | MCL_CURRENT) < 0) {
        fprintf(stdout, "mlockall failed: %m\n");
        goto fn_fail;
    }

    if (enable_sharp_coll) {
        ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, inbuf, sharp_conf.max_message_size, &s_mem_mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "Failed to register source buffer\n");
            goto fn_fail;
        }

        ret = sharp_coll_reg_mr(coll_sharp_component.sharp_coll_context, outbuf, sharp_conf.max_message_size * comm_size, &r_mem_mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "Failed to register receive buffer\n");
            goto fn_fail;
        }
    }

    for (count = min_count; count <= max_count; count *= 2) {
        if (enable_sharp_coll) {
            gather_spec.sbuf_desc.mem_type = sharp_conf.s_mem_type;
            gather_spec.rbuf_desc.mem_type = sharp_conf.r_mem_type;

            assert(sharp_conf.sdata_layout == TEST_DATA_LAYOUT_CONTIG && sharp_conf.rdata_layout == TEST_DATA_LAYOUT_CONTIG);
            gather_spec.sbuf_desc.type = SHARP_DATA_BUFFER;
            gather_spec.sbuf_desc.buffer.ptr = inbuf;
            gather_spec.sbuf_desc.buffer.length = (count * sharp_test_data_type->size);
            gather_spec.sbuf_desc.buffer.mem_handle = s_mem_mr;

            gather_spec.rbuf_desc.type = SHARP_DATA_BUFFER;
            gather_spec.rbuf_desc.buffer.ptr = outbuf;
            gather_spec.rbuf_desc.buffer.length = (count * sharp_test_data_type->size * comm_size);
            gather_spec.rbuf_desc.buffer.mem_handle = r_mem_mr;

            gather_spec.dtype = sharp_conf.datatype;
            gather_spec.size = count;
            gather_spec.offset = 0;
        }

        MPI_Barrier(mpi_comm);
        timer = 0.0;
        for (i = 0, j = 0; i < iterations + skips; i++) {
#if _DCHECK
            int pos, result;
            for (pos = 0; pos < count; pos++) {
                inbuf[pos] = i;
                outbuf[pos] = 0;
            }
#endif
            if (perf_with_barrier) {
                if (enable_sharp_coll) {
                    ret = sharp_coll_do_barrier(sharp_comm->sharp_coll_comm);
                    if (ret != SHARP_COLL_SUCCESS) {
                        if (rank == 0)
                            fprintf(stderr, "Barrier failed: %s\n", sharp_coll_strerror(ret));
                        exit(-1);
                    }
                } else {
                    MPI_Barrier(mpi_comm);
                }
            }

            t_start = sharp_time_sec();

            if (enable_sharp_coll) {
                ret = sharp_coll_do_allgather(sharp_comm->sharp_coll_comm, &gather_spec);
                if (ret != SHARP_COLL_SUCCESS) {
                    if (rank == 0)
                        fprintf(stderr, "Allgatherfailed: %s\n", sharp_coll_strerror(ret));
                    exit(-1);
                }
            } else {
                MPI_Allgather(inbuf, count, sharp_test_data_type->mpi_id, outbuf, count, sharp_test_data_type->mpi_id, mpi_comm);
            }

            if (i >= skips) {
                t_stop = sharp_time_sec();
                timer += (t_stop - t_start);
                iter_time[j++] = (t_stop - t_start);
            }
#if _DCHECK
            for (pos = 0; pos < count; pos++) {
                result = i * comm_size;
                if (outbuf[pos] != result) {
                    fprintf(stdout, "data validation failed. At pos:%d got  =%d, expected:%d\n", pos, (int)outbuf[pos], result);
                    break;
                }
            }
#endif
        }

        latency = (double)(timer * 1e6) / iterations;

        min_iter_time = max_iter_time = iter_time[0];
        for (i = 0; i < iterations; i++) {
            if (min_iter_time > iter_time[i])
                min_iter_time = iter_time[i];
            if (max_iter_time < iter_time[i])
                max_iter_time = iter_time[i];
        }
        min_iter_time = (double)(min_iter_time * 1e6);
        max_iter_time = (double)(max_iter_time * 1e6);

        MPI_Reduce(&min_iter_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, mpi_comm);
        MPI_Reduce(&max_iter_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_comm);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);

        avg_time = avg_time / comm_size;
        avg_bw = (comm_size * count * sharp_test_data_type->size * 1.0 * 1000000) / (avg_time * 125000000);

        MPI_Gather(iter_time, iterations, MPI_DOUBLE, all_iter_time, iterations, MPI_DOUBLE, 0, mpi_comm);

        if (rank == 0) {
            fprintf(stdout,
                    "%15lu %15.2f %15.2f %15.2f %15.2f %10d\n",
                    (comm_size * count * sharp_test_data_type->size),
                    avg_time,
                    min_time,
                    max_time,
                    avg_bw,
                    iterations);
#if 0
			print_coll_iterations_perf_data(all_iter_time, rank, comm_size, (int )(count * sharp_test_data_type->size), iterations);
#endif
            fflush(stdout);
        }
    }

fn_fail:
    if (enable_sharp_coll) {
        if (s_mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, s_mem_mr);
        if (r_mem_mr)
            sharp_coll_dereg_mr(coll_sharp_component.sharp_coll_context, r_mem_mr);
    }
    free_memory(inbuf, sharp_conf.s_mem_type, 1);
    free_memory(outbuf, sharp_conf.r_mem_type, 0);
    free(iter_time);
    free(all_iter_time);
}
