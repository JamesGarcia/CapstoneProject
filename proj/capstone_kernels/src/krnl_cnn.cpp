#include "cnn_util.h"
#include "krnl_L1.cpp"
#include "krnl_L2.cpp"
#include "krnl_L3.cpp"

extern "C" {

void cnn_top(const cnndata_t *input, const cnndata_t *W1, const cnndata_t *W2, const cnndata_t *W3, cnndata_t *O1, cnndata_t *O2, cnndata_t *O3){
	//cnndata_t H1[M_1 * R_1 * C_1];
	//cnndata_t H2[M_2 * R_2 * C_2];

	layer_1::krnl_cnn(O1, input, W1);
	layer_2::krnl_cnn(O2,    O1, W2);
	layer_3::krnl_cnn(O3,    O2, W3);
}

}
