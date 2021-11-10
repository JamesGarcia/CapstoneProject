
#define N_2 (2) // should be 64, set to 2 for demo
#define M_2 (2) // should be 32, set to 2 for demo
#define R_2 (1080)
#define C_2 (1920)
#define K_2 (5)
#define S_2 (1) // never used b/c a mult by 1 is a NoOp

#define TM_2 (2)
#define TN_2 (2)
#define TR_2 (2)
#define TC_2 (2)

#include "cnn_util.h"

namespace layer_2 {

void loadBufO(cnndata_t BufO[TM_2][TR_2][TC_2]){
	for(index_t ofm_i=0;ofm_i<TM_2;ofm_i++){
		for(index_t row_i=0;row_i<TR_2;row_i++){
			for(index_t col_i=0;col_i<TC_2;col_i++){
				BufO[ofm_i][row_i][col_i] = 0;
			}
		}
	}
}

void loadBufI(cnndata_t BufI[TN_2][TR_2+K_2-1][TC_2+K_2-1], const cnndata_t *input, index_t row, index_t col, index_t ifm){
	index_t ifm_x = ifm, ifm_i = 0;
	for(;ifm_x<MIN(ifm+TN_2,N_2);ifm_x++,ifm_i++){
		index_t row_x = row, row_i = 0;
		for(;row_x<MIN(row+TR_2+K_2-1,R_2);row_x++,row_i++){
			index_t col_x = col, col_i = 0;
			for(;col_x<MIN(col+TC_2+K_2-1,C_2);col_x++,col_i++){
				BufI[ifm_i][row_i][col_i] = input[ifm_x*R_2*C_2 + row_x*C_2 + col_x];
			}
			for(;col_i<TC_2+K_2-1;col_i++){
				BufI[ifm_i][row_i][col_i] = 0;
			}
		}
		for(;row_i<TR_2+K_2-1;row_i++){
			for(index_t col_i=0;col_i<TC_2+K_2-1;col_i++){
				BufI[ifm_i][row_i][col_i] = 0;
			}
		}
	}
	for(;ifm_i<TN_2;ifm_i++){
		for(index_t row_i=0;row_i<TR_2+K_2-1;row_i++){
			for(index_t col_i=0;col_i<TC_2+K_2-1;col_i++){
				BufI[ifm_i][row_i][col_i] = 0;
			}
		}
	}
}

void loadBufW(cnndata_t BufW[TM_2][TN_2][K_2][K_2], const cnndata_t *weights, index_t ofm, index_t ifm){
	index_t ofm_x = ofm, ofm_i = 0;
	for(;ofm_x<MIN(ofm+TM_2,M_2);ofm_x++,ofm_i++){
		index_t ifm_x = ifm, ifm_i = 0;
		for(;ifm_x<MIN(ifm+TN_2,N_2);ifm_x++,ifm_i++){
			for(index_t i=0;i<K_2;i++){
				for(index_t j=0;j<K_2;j++){
					BufW[ofm_i][ifm_i][i][j] = weights[ofm_x*N_2*K_2*K_2 + ifm_x*K_2*K_2 + i*K_2 + j];
				}
			}
		}
		for(;ifm_i<TN_2;ifm_i++){
			for(index_t i=0;i<K_2;i++){
				for(index_t j=0;j<K_2;j++){
					BufW[ofm_i][ifm_i][i][j] = 0;
				}
			}
		}


	}
	for(;ofm_i<TM_2;ofm_i++){
		for(index_t ifm_i=0;ifm_i<TN_2;ifm_i++){
			for(index_t i=0;i<K_2;i++){
				for(index_t j=0;j<K_2;j++){
					BufW[ofm_i][ifm_i][i][j] = 0;
				}
			}
		}
	}
}

void cnn_blocked_kernel(cnndata_t BufI[TN_2][TR_2+K_2-1][TC_2+K_2-1], cnndata_t BufO[TM_2][TR_2][TC_2], cnndata_t BufW[TM_2][TN_2][K_2][K_2]){
	KRow: for(index_t i=0; i<K_2; i++){
		KCol: for(index_t j=0; j<K_2; j++){
			Row: for(index_t row=0; row<TR_2; row++){
				Col: for(index_t col=0; col<TC_2; col++){
					Ofm: for(index_t ofm=0; ofm<TM_2; ofm++){
						Ifm: for(index_t ifm=0; ifm<TN_2; ifm++){
							BufO[ofm][row][col] += BufW[ofm][ifm][i][j] * BufI[ifm][row+i][col+j];
						}
					}
				}
			}
		}
	}
}

void storeBufO(cnndata_t BufO[TM_2][TR_2][TC_2], cnndata_t *output, index_t ofm, index_t row, index_t col){
	for(index_t ofm_x=ofm,ofm_i=0; ofm_x<MIN(ofm+TM_2,M_2); ofm_x++,ofm_i++){
		for(index_t row_x=row,row_i=0; row_x<MIN(row+TR_2,R_2); row_x++,row_i++){
			for(index_t col_x=col,col_i=0; col_x<MIN(col+TC_2,C_2); col_x++,col_i++){
				output[ofm_x*R_2*C_2 + row_x*C_2 + col_x] = BufO[ofm_i][row_i][col_i];
			}
		}
	}
}

void krnl_cnn(cnndata_t *output, const cnndata_t *input, const cnndata_t *weights){
	cnndata_t BufI[TN_2][TR_2+K_2-1][TC_2+K_2-1];
	cnndata_t BufO[TM_2][TR_2][TC_2];
	cnndata_t BufW[TM_2][TN_2][K_2][K_2];

	for(index_t row=0;row<R_2;row+=TR_2){
		for(index_t col=0;col<C_2;col+=TC_2){
			for(index_t ofm=0;ofm<M_2;ofm+=TM_2){
				loadBufO(BufO);
				for(index_t ifm=0;ifm<N_2;ifm+=TN_2){
					loadBufI(BufI, input, row, col, ifm);
					loadBufW(BufW, weights, ofm, ifm);
					cnn_blocked_kernel(BufI, BufO, BufW);
				}
				storeBufO(BufO, outputs, ofm, row, col);
			}
		}
	}
}

}

