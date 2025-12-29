import torch
from configSettings import *
from layerUtils import *

def deltaFC(upstreamGrad, layerWeights):
	return upstreamGrad @ (layerWeights.data)

def deltaActivation(upstreamGrad, inputData):
	return upstreamGrad * (inputData > 0)

def deltaLN(upstreamGrad, layerWeights, inputData):
	tx = torch.diag(layerWeights.data) @ upstreamGrad.T
	t1 = D_FTS_MAPS * tx
	t2 = tx.sum(0)
	myNormed_data = myLN_noParams(inputData)
	t3 = myNormed_data.T * (myNormed_data.T * tx).sum(0)
	myLocVar = inputData.T.std(0, keepdim=True, unbiased=False).squeeze()
	delta_out = (t1 - t2 - t3) @ torch.diag(1 / myLocVar)
	delta_out = delta_out.T / D_FTS_MAPS
	return delta_out

def selfAttnBack(delta_i, a_iMinus1, attn_h, v_h, k_h, q_h, w_vh, w_qh, w_kh):
	delta_causal_h = ((delta_i @ v_h.T) - ((delta_i @ v_h.T) * attn_h) @ torch.ones(SEQ_LEN, SEQ_LEN)) * attn_h
	delta_raw_h = delta_causal_h / torch.sqrt(torch.tensor(HEAD_D_FTS_MAPS))
	
	delta_v = attn_h.T @ delta_i @ w_vh.T
	delta_q = delta_raw_h @ k_h @ w_qh.T
	delta_k = delta_raw_h.T @ q_h @ w_kh.T
	delta_head = delta_v + delta_q + delta_k

	my_w_vh_grad, my_b_vh_grad	= (attn_h @ a_iMinus1).T @ delta_i    / SEQ_LEN, delta_i.sum(0)		      / SEQ_LEN
	my_w_qh_grad, my_b_qh_grad	= (a_iMinus1.T @ delta_raw_h @ k_h)   / SEQ_LEN, (delta_raw_h @ k_h).sum(0)   / SEQ_LEN 
	my_w_kh_grad, my_b_kh_grad	= (delta_raw_h @ a_iMinus1).T @ q_h / SEQ_LEN, (delta_raw_h.T @ q_h).sum(0) / SEQ_LEN

	return my_w_vh_grad, my_b_vh_grad, my_w_qh_grad, my_b_qh_grad, my_w_kh_grad, my_b_kh_grad, delta_head
