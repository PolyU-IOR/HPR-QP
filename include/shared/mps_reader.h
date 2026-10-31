#ifndef HPRQP_MPS_READER_H
#define HPRQP_MPS_READER_H

#include <cstdio>

typedef double HPRLP_FLOAT;

/**
 * MPS File Reader for C++
 * 
 * This implementation provides functionality to read MPS (Mathematical Programming System)
 * format files and convert them to internal QP parser structures.
 * 
 * The implementation is based on and follows the design of qpsreader.jl but adapted
 * for MPS format, which is the standard format for linear programming problems.
 * 
 * Note: While the internal structures use "QPS" naming for historical reasons,
 * this reader specifically handles MPS format files.
 * 
 * References: QPSReader.jl, HiGHS
 */

/* Variable types */
enum VariableType {
    VTYPE_CONTINUOUS,
    VTYPE_BINARY,
    VTYPE_INTEGER,
    VTYPE_MARKED,  /* Internal use for integer variables */
    VTYPE_SEMICONTINUOUS,
    VTYPE_SEMIINTEGER
};

/* Row types (constraint types) */
enum RowType {
    RTYPE_OBJECTIVE,    /* 'N' type row */
    RTYPE_EQUALTO,      /* 'E' type row */
    RTYPE_LESSTHAN,     /* 'L' type row */
    RTYPE_GREATERTHAN   /* 'G' type row */
};

/* CSR (Compressed Sparse Row) matrix structure */
struct CSRMatrix {
    int nrows;          /* Number of rows */
    int ncols;          /* Number of columns */
    int nnz;            /* Number of nonzeros */
    int *row_ptr;       /* Row pointers (size: nrows+1) */
    int *col_idx;       /* Column indices (size: nnz) */
    HPRLP_FLOAT *values;     /* Values (size: nnz) */
};

/* Objective sense */
enum MPSObjSense {
    OBJSENSE_MIN,
    OBJSENSE_MAX,
    OBJSENSE_NOTSET
};

/* MPS format type */
enum MPSFormat {
    MPS_FIXED,
    MPS_FREE
};

/* Section identifiers */
enum SectionType {
    SECTION_NONE = -1,
    SECTION_ENDATA = 0,
    SECTION_NAME = 1,
    SECTION_OBJSENSE = 2,
    SECTION_ROWS = 3,
    SECTION_COLUMNS = 4,
    SECTION_RHS = 5,
    SECTION_BOUNDS = 6,
    SECTION_RANGES = 7,
    SECTION_QUADOBJ = 8,
    SECTION_QMATRIX = 8,  /* Same as QUADOBJ */
    SECTION_QCMATRIX = 9,
    SECTION_OBJECT_BOUND = 10
};

/* Name-Index mapping structure with hash table */
struct NameIndexPair {
    char *name;
    int index;
    int next;  /* For chaining in hash table */
};

/* Hash table for name-index mapping */
struct NameIndexMap {
    NameIndexPair *pairs;  /* Array of all entries */
    int *buckets;          /* Hash table buckets (indices into pairs array) */
    int size;              /* Number of entries */
    int capacity;          /* Capacity of pairs array */
    int num_buckets;       /* Number of hash buckets */
};

/* QPSData structure to hold the model data */
struct QPSData {
    /* Problem dimensions */
    int nvar;  /* Number of variables */
    int ncon;  /* Number of constraints */
    
    /* Objective function */
    MPSObjSense objsense;  /* Objective sense: minimize or maximize */
    HPRLP_FLOAT c0;          /* Constant term in objective */
    HPRLP_FLOAT *c;          /* Linear coefficients in objective (size: nvar) */
    
    /* Quadratic objective matrix Q in COO format */
    int qnnz;           /* Number of nonzeros in Q */
    int qnnz_capacity;  /* Allocated capacity */
    int *qrows;         /* Row indices */
    int *qcols;         /* Column indices */
    HPRLP_FLOAT *qvals;      /* Values */

    /* Quadratic-constraint matrices in COO form. */
    int nqc;
    int qc_capacity;
    int qcnnz;
    int qcnnz_capacity;
    int *qc_con_index;       /* Original linear-constraint row for each QC */
    int *con_to_qc;          /* Original constraint row -> QC index, or -1 */
    int *qcids;              /* QC block index for each quadratic entry */
    int *qcrows;             /* First variable index */
    int *qccols;             /* Second variable index */
    HPRLP_FLOAT *qcvals;
    
    /* Constraint matrix A in COO format */
    int annz;           /* Number of nonzeros in A */
    int annz_capacity;  /* Allocated capacity */
    int *arows;         /* Row indices */
    int *acols;         /* Column indices */
    HPRLP_FLOAT *avals;      /* Values */
    
    /* Bounds */
    HPRLP_FLOAT *lcon;  /* Constraint lower bounds (size: ncon) */
    HPRLP_FLOAT *ucon;  /* Constraint upper bounds (size: ncon) */
    HPRLP_FLOAT *lvar;  /* Variable lower bounds (size: nvar) */
    HPRLP_FLOAT *uvar;  /* Variable upper bounds (size: nvar) */
    
    /* Names */
    char *name;      /* Problem name */
    char *objname;   /* Objective function name */
    char *rhsname;   /* RHS name */
    char *bndname;   /* BOUNDS name */
    char *rngname;   /* RANGES name */
    
    char **varnames;  /* Variable names (size: nvar) */
    char **connames;  /* Constraint names (size: ncon) */
    
    /* Name-index mappings */
    NameIndexMap varindices;  /* Variable name -> index */
    NameIndexMap conindices;  /* Constraint name -> index */
    
    /* Variable and constraint types */
    VariableType *vartypes;  /* Variable types (size: nvar) */
    RowType *contypes;       /* Constraint types (size: ncon) */
    
    /* Capacities for dynamic arrays */
    int var_capacity;
    int con_capacity;
};

/* MPS Card structure for parsing */
struct MPSCard {
    int nline;          /* Line number */
    bool iscomment;     /* Is this line a comment? */
    bool isheader;      /* Is this line a section header? */
    int nfields;        /* Number of fields read */
    
    /* Fields */
    char f1[256];
    char f2[256];
    char f3[256];
    char f4[256];
    char f5[256];
    char f6[256];
};

/* Function declarations */

/* Initialize and free QPSData */
QPSData* qpsdata_create();
QPSData* qpsdata_create_with_capacity(int var_cap, int con_cap, int nnz_cap);
void qpsdata_free(QPSData *qps);

/* Main reading function */
QPSData* readqps(const char *filename, MPSFormat format);
QPSData* readqps_from_file(std::FILE *fp, MPSFormat format);

/* Name-index map functions */
void namemap_init(NameIndexMap *map);
void namemap_free(NameIndexMap *map);
int namemap_get(NameIndexMap *map, const char *name, int default_value);
void namemap_set(NameIndexMap *map, const char *name, int index);

/* Utility functions */
void trim(char *str);
char* strdup_safe(const char *str);

/* CSR matrix functions */
CSRMatrix* csr_create(int nrows, int ncols, int nnz);
void csr_free(CSRMatrix *csr);
CSRMatrix* coo_to_csr(int nrows, int ncols, int nnz, 
                      const int *row_indices, const int *col_indices, 
                      const HPRLP_FLOAT *values);
CSRMatrix* qpsdata_get_csr_matrix(const QPSData *qps);

#endif
