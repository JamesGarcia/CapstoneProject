#include "cnn_util.h"


namespace layer_1 {

void loadBufO_L1(cnndata_t BufO[TM_1][TR_1][TC_1]){
	for(index_t ofm_i=0;ofm_i<TM_1;ofm_i++){
		for(index_t row_i=0;row_i<TR_1;row_i++){
			for(index_t col_i=0;col_i<TC_1;col_i++){
				BufO[ofm_i][row_i][col_i] = 0;
			}
		}
	}
}

#if PROD
void loadBufI_L1(cnndata_t BufI[TR_1+K_1-1][TC_1+K_1-1], const cnndata_t *input, index_t row, index_t col){
#else
void loadBufI_L1(cnndata_t BufI[TR_1+K_1-1][TC_1+K_1-1], const cnndata_t input[R_1*C_1], index_t row, index_t col){
#endif
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
#if PROD
void loadBufW_L1(cnndata_t BufW[TM_1][K_1][K_1], const cnndata_t *weights, index_t ofm){
#else
void loadBufW_L1(cnndata_t BufW[TM_1][K_1][K_1], const cnndata_t weights[M_1*K_1*K_1], index_t ofm){
#endif
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

void cnn_blocked_kernel_L1(cnndata_t BufI[TR_1+K_1-1][TC_1+K_1-1], cnndata_t BufO[TM_1][TR_1][TC_1], cnndata_t BufW[TM_1][K_1][K_1]){
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

#if PROD
void storeBufO_L1(cnndata_t BufO[TM_1][TR_1][TC_1], cnndata_t *output, index_t ofm, index_t row, index_t col){
#else
void storeBufO_L1(cnndata_t BufO[TM_1][TR_1][TC_1], cnndata_t output[M_1*R_1*C_1], index_t ofm, index_t row, index_t col){
#endif
	for(index_t ofm_x=ofm,ofm_i=0; ofm_x<MIN(ofm+TM_1,M_1); ofm_x++,ofm_i++){
		for(index_t row_x=row,row_i=0; row_x<MIN(row+TR_1,R_1); row_x++,row_i++){
			for(index_t col_x=col,col_i=0; col_x<MIN(col+TC_1,C_1); col_x++,col_i++){
				output[ofm_x*R_1*C_1 + row_x*C_1 + col_x] = BufO[ofm_i][row_i][col_i];
			}
		}
	}
}

#if PROD
void krnl_cnn_L1(cnndata_t *output, const cnndata_t *input, const cnndata_t *weights){
#else
void krnl_cnn_L1(cnndata_t output[M_1*R_1*C_1], const cnndata_t input[R_1*C_1], const cnndata_t weights[M_1*K_1*K_1]){
#endif
	cnndata_t BufI[TR_1+K_1-1][TC_1+K_1-1];
	cnndata_t BufO[TM_1][TR_1][TC_1];
	cnndata_t BufW[TM_1][K_1][K_1];

	for(index_t row=0;row<R_1;row+=TR_1){
		for(index_t col=0;col<C_1;col+=TC_1){
			for(index_t ofm=0;ofm<M_1;ofm+=TM_1){
				loadBufO_L1(BufO);
				// would loop TN_1 here but N_1 = 1
				loadBufI_L1(BufI, input, row, col);
				loadBufW_L1(BufW, weights, ofm);
				cnn_blocked_kernel_L1(BufI, BufO, BufW);
				storeBufO_L1(BufO, output, ofm, row, col);
			}
		}
	}
}

}

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

#if PROD
void loadBufI(cnndata_t BufI[TN_2][TR_2+K_2-1][TC_2+K_2-1], const cnndata_t *input, index_t row, index_t col, index_t ifm){
#else
void loadBufI(cnndata_t BufI[TN_2][TR_2+K_2-1][TC_2+K_2-1], const cnndata_t input[N_2*R_2*C_2], index_t row, index_t col, index_t ifm){
#endif
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

#if PROD
void loadBufW(cnndata_t BufW[TM_2][TN_2][K_2][K_2], const cnndata_t *weights, index_t ofm, index_t ifm){
#else
void loadBufW(cnndata_t BufW[TM_2][TN_2][K_2][K_2], const cnndata_t weights[M_2*N_2*K_2*K_2], index_t ofm, index_t ifm){
#endif
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

#if PROD
void storeBufO(cnndata_t BufO[TM_2][TR_2][TC_2], cnndata_t *output, index_t ofm, index_t row, index_t col){
#else
void storeBufO(cnndata_t BufO[TM_2][TR_2][TC_2], cnndata_t output[M_2*R_2*C_2], index_t ofm, index_t row, index_t col){
#endif
	for(index_t ofm_x=ofm,ofm_i=0; ofm_x<MIN(ofm+TM_2,M_2); ofm_x++,ofm_i++){
		for(index_t row_x=row,row_i=0; row_x<MIN(row+TR_2,R_2); row_x++,row_i++){
			for(index_t col_x=col,col_i=0; col_x<MIN(col+TC_2,C_2); col_x++,col_i++){
				output[ofm_x*R_2*C_2 + row_x*C_2 + col_x] = BufO[ofm_i][row_i][col_i];
			}
		}
	}
}

#if PROD
void krnl_cnn(cnndata_t *output, const cnndata_t *input, const cnndata_t *weights){
#else
void krnl_cnn(cnndata_t output[M_2*R_2*C_2], const cnndata_t input[N_2*R_2*C_2], const cnndata_t weights[M_2*N_2*K_2*K_2]){
#endif
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
				storeBufO(BufO, output, ofm, row, col);
			}
		}
	}
}

}

namespace layer_3 {

void loadBufO(cnndata_t BufO[TR_3][TC_3]){
	for(index_t row_i=0;row_i<TR_3;row_i++){
		for(index_t col_i=0;col_i<TC_3;col_i++){
			BufO[row_i][col_i] = 0;
		}
	}
}

#if PROD
void loadBufI(cnndata_t BufI[TN_3][TR_3+K_3-1][TC_3+K_3-1], const cnndata_t *input, index_t row, index_t col, index_t ifm){
#else
void loadBufI(cnndata_t BufI[TN_3][TR_3+K_3-1][TC_3+K_3-1], const cnndata_t input[N_3*R_3*C_3], index_t row, index_t col, index_t ifm){
#endif
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

#if PROD
void loadBufW(cnndata_t BufW[TN_3][K_3][K_3], const cnndata_t *weights, index_t ifm){
#else
void loadBufW(cnndata_t BufW[TN_3][K_3][K_3], const cnndata_t weights[N_3*K_3*K_3], index_t ifm){
#endif
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

#if PROD
void storeBufO(cnndata_t BufO[TR_3][TC_3], cnndata_t *output, index_t row, index_t col){
#else
void storeBufO(cnndata_t BufO[TR_3][TC_3], cnndata_t output[R_3*C_3], index_t row, index_t col){
#endif
	for(index_t row_x=row,row_i=0; row_x<MIN(row+TR_3,R_3); row_x++,row_i++){
		for(index_t col_x=col,col_i=0; col_x<MIN(col+TC_3,C_3); col_x++,col_i++){
			output[row_x*C_3 + col_x] = BufO[row_i][col_i];
		}
	}
}

#if PROD
void krnl_cnn(cnndata_t *output, const cnndata_t *input, const cnndata_t *weights){
#else
void krnl_cnn(cnndata_t output[R_3*C_3], const cnndata_t input[N_3*R_3*C_3], const cnndata_t weights[N_3*K_3*K_3]){
#endif
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
			storeBufO(BufO, output, row, col);
		}
	}
}

}

#if PROD
extern "C" {
void cnn_top(const cnndata_t *input, const cnndata_t *W1, const cnndata_t *W2, const cnndata_t *W3, cnndata_t *O1, cnndata_t *O2, cnndata_t *O3){
#else
void cnn_top(const cnndata_t input[R_1*C_1],
		 	 const cnndata_t W1[M_1*K_1*K_1],
			 const cnndata_t W2[M_2*N_2*K_2*K_2],
			 const cnndata_t W3[N_3*K_3*K_3],
			 cnndata_t O1[R_1*C_1*M_1],
			 cnndata_t O2[R_2*C_2*M_2],
			 cnndata_t O3[R_3*C_3]){
#endif
	layer_1::krnl_cnn_L1(O1, input, W1);
	layer_2::krnl_cnn(O2,    O1, W2);
	layer_3::krnl_cnn(O3,    O2, W3);
}
#if PROD
}
#endif
