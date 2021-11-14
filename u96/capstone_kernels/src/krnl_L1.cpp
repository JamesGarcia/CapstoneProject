
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

#include "cnn_util.h"

namespace layer_1 {

void loadBufO(cnndata_t BufO[TM_1][TR_1][TC_1]){
	for(index_t ofm_i=0;ofm_i<TM_1;ofm_i++){
		for(index_t row_i=0;row_i<TR_1;row_i++){
			for(index_t col_i=0;col_i<TC_1;col_i++){
				BufO[ofm_i][row_i][col_i] = 0;
			}
		}
	}
}

void loadBufI(cnndata_t BufI[TR_1+K_1-1][TC_1+K_1-1], const cnndata_t *input, index_t row, index_t col){
	index_t row_x = row, row_i = 0;
	for(;row_x<MIN(row+TR_1+K_1-1,R_1);row_x++,row_i++){
		index_t col_x = col, col_i = 0;
		for(;col_x<MIN(col+TC_1+K_1-1,C_1);col_x++,col_i++){
			BufI[row_i][col_i] = input[row_x*C_1+col_x];
		}
		for(;col_i<TC_1+K_1-1;col_i++){
			BufI[row_i][col_i] = 0;
		}
	}
	for(;row_i<TR_1+K_1-1;row_i++){
		for(index_t col_i=0;col_i<TC_1+K_1-1;col_i++){
			BufI[row_i][col_i] = 0;
		}
	}
}

void loadBufW(cnndata_t BufW[TM_1][K_1][K_1], const cnndata_t *weights, index_t ofm){
	index_t ofm_x = ofm, ofm_i = 0;
	for(;ofm_x<MIN(ofm+TM_1,M_1);ofm_x++,ofm_i++){
		for(index_t i=0;i<K_1;i++){
			for(index_t j=0;j<K_1;j++){
				BufW[ofm_i][i][j] = weights[ofm_x*K_1*K_1+i*K_1+j];
			}
		}
	}
	for(;ofm_i<TM_1;ofm_i++){
		for(index_t i=0;i<K_1;i++){
			for(index_t j=0;j<K_1;j++){
				BufW[ofm_i][i][j] = 0;
			}
		}
	}
}

void cnn_blocked_kernel(cnndata_t BufI[TR_1+K_1-1][TC_1+K_1-1], cnndata_t BufO[TM_1][TR_1][TC_1], cnndata_t BufW[TM_1][K_1][K_1]){
	KRow: for(index_t i=0; i<K_1; i++){
		KCol: for(index_t j=0; j<K_1; j++){
			Row: for(index_t row=0; row<TR_1; row++){
				Col: for(index_t col=0; col<TC_1; col++){
					Ofm: for(index_t ofm=0; ofm<TM_1; ofm++){
						BufO[ofm][row][col] += BufW[ofm][i][j] * BufI[row+i][col+j];
					}
				}
			}
		}
	}
}

void storeBufO(cnndata_t BufO[TM_1][TR_1][TC_1], cnndata_t *output, index_t ofm, index_t row, index_t col){
	for(index_t ofm_x=ofm,ofm_i=0; ofm_x<MIN(ofm+TM_1,M_1); ofm_x++,ofm_i++){
		for(index_t row_x=row,row_i=0; row_x<MIN(row+TR_1,R_1); row_x++,row_i++){
			for(index_t col_x=col,col_i=0; col_x<MIN(col+TC_1,C_1); col_x++,col_i++){
				output[ofm_x*R_1*C_1 + row_x*C_1 + col_x] = BufO[ofm_i][row_i][col_i];
			}
		}
	}
}

void krnl_cnn(cnndata_t *output, const cnndata_t *input, const cnndata_t *weights){
	cnndata_t BufI[TR_1+K_1-1][TC_1+K_1-1];
	cnndata_t BufO[TM_1][TR_1][TC_1];
	cnndata_t BufW[TM_1][K_1][K_1];

	for(index_t row=0;row<R_1;row+=TR_1){
		for(index_t col=0;col<C_1;col+=TC_1){
			for(index_t ofm=0;ofm<M_1;ofm+=TM_1){
				loadBufO(BufO);
				// would loop TN_1 here but N_1 = 1
				loadBufI(BufI, input, row, col);
				loadBufW(BufW, weights, ofm);
				cnn_blocked_kernel(BufI, BufO, BufW);
				storeBufO(BufO, outputs, ofm, row, col);
			}
		}
	}
}

}

