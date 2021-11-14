
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

#include "cnn_util.h"

namespace layer_3 {

void loadBufO(cnndata_t BufO[TR_3][TC_3]){
	for(index_t row_i=0;row_i<TR_3;row_i++){
		for(index_t col_i=0;col_i<TC_3;col_i++){
			BufO[row_i][col_i] = 0;
		}
	}
}

void loadBufI(cnndata_t BufI[TN_3][TR_3+K_3-1][TC_3+K_3-1], const cnndata_t *input, index_t row, index_t col, index_t ifm){
	index_t ifm_x = ifm, ifm_i = 0;
	for(;ifm_x<MIN(ifm+TN_3,N_3);ifm_x++,ifm_i++){
		index_t row_x = row, row_i = 0;
		for(;row_x<MIN(row+TR_3+K_3-1,R_3);row_x++,row_i++){
			index_t col_x = col, col_i = 0;
			for(;col_x<MIN(col+TC_3+K_3-1,C_3);col_x++,col_i++){
				BufI[ifm_i][row_i][col_i] = input[ifm_x*R_3*C_3 + row_x*C_3 + col_x];
			}
			for(;col_i<TC_3+K_3-1;col_i++){
				BufI[ifm_i][row_i][col_i] = 0;
			}
		}
		for(;row_i<TR_3+K_3-1;row_i++){
			for(index_t col_i=0;col_i<TC_3+K_3-1;col_i++){
				BufI[ifm_i][row_i][col_i] = 0;
			}
		}
	}
	for(;ifm_i<TN_3;ifm_i++){
		for(index_t row_i=0;row_i<TR_3+K_3-1;row_i++){
			for(index_t col_i=0;col_i<TC_3+K_3-1;col_i++){
				BufI[ifm_i][row_i][col_i] = 0;
			}
		}
	}
}

void loadBufW(cnndata_t BufW[TN_3][K_3][K_3], const cnndata_t *weights, index_t ifm){
	index_t ifm_x = ifm, ifm_i = 0;
	for(;ifm_x<MIN(ifm+TN_3,N_3);ifm_x++,ifm_i++){
		for(index_t i=0;i<K_3;i++){
			for(index_t j=0;j<K_3;j++){
				BufW[ifm_i][i][j] = weights[ifm_x*K_3*K_3 + i*K_3 + j];
			}
		}
	}
	for(;ifm_i<TN_3;ifm_i++){
		for(index_t i=0;i<K_3;i++){
			for(index_t j=0;j<K_3;j++){
				BufW[ifm_i][i][j] = 0;
			}
		}
	}
}

void cnn_blocked_kernel(cnndata_t BufI[TN_3][TR_3+K_3-1][TC_3+K_3-1], cnndata_t BufO[TR_3][TC_3], cnndata_t BufW[TN_3][K_3][K_3]){
	KRow: for(index_t i=0; i<K_3; i++){
		KCol: for(index_t j=0; j<K_3; j++){
			Row: for(index_t row=0; row<TR_3; row++){
				Col: for(index_t col=0; col<TC_3; col++){
					Ifm: for(index_t ifm=0; ifm<TN_3; ifm++){
						BufO[row][col] += BufW[ifm][i][j] * BufI[ifm][row+i][col+j];
					}
				}
			}
		}
	}
}

void storeBufO(cnndata_t BufO[TR_3][TC_3], cnndata_t *output, index_t row, index_t col){
	for(index_t row_x=row,row_i=0; row_x<MIN(row+TR_3,R_3); row_x++,row_i++){
		for(index_t col_x=col,col_i=0; col_x<MIN(col+TC_3,C_3); col_x++,col_i++){
			output[row_x*C_3 + col_x] = BufO[row_i][col_i];
		}
	}
}

void krnl_cnn(cnndata_t *output, const cnndata_t *input, const cnndata_t *weights){
	cnndata_t BufI[TN_3][TR_3+K_3-1][TC_3+K_3-1];
	cnndata_t BufO[TR_3][TC_3];
	cnndata_t BufW[TN_3][K_3][K_3];

	for(index_t row=0;row<R_3;row+=TR_3){
		for(index_t col=0;col<C_3;col+=TC_3){
			loadBufO(BufO);
			for(index_t ifm=0;ifm<N_3;ifm+=TN_3){
				loadBufI(BufI, input, row, col, ifm);
				loadBufW(BufW, weights, ifm);
				cnn_blocked_kernel(BufI, BufO, BufW);
			}
			storeBufO(BufO, outputs, row, col);
		}
	}
}

}
