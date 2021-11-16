#ifndef __CNN_UTIL_H
#define __CNN_UTIL_H

#define PROD 1



#define N_1 (1)
#define M_1 (64) // should be 64, set to 2 for demo
#define R_1 (1080)
#define C_1 (1920)
#define K_1 (9)
#define S_1 (1) // never used b/c a mult by 1 is a NoOp

#define TM_1 (2)
//#define TN_1 () no need to tile a single dimension
#define TR_1 (2)
#define TC_1 (2)


#define N_2 (64) // should be 64, set to 2 for demo
#define M_2 (32) // should be 32, set to 2 for demo
#define R_2 (1080)
#define C_2 (1920)
#define K_2 (5)
#define S_2 (1) // never used b/c a mult by 1 is a NoOp

#define TM_2 (2)
#define TN_2 (2)
#define TR_2 (2)
#define TC_2 (2)


#define N_3 (32) // should be 32, set to 2 for demo
#define M_3 (1)
#define R_3 (1080)
#define C_3 (1920)
#define K_3 (5)
#define S_3 (1) // never used b/c a mult by 1 is a NoOp

//#define TM_3 (8)
#define TN_3 (2)
#define TR_3 (2)
#define TC_3 (2)

// i guess we need these too?
#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif
#ifndef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif

typedef float cnndata_t;
typedef int   index_t;

#endif
