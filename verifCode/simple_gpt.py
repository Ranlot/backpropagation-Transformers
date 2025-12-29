import torch
import torch.nn as nn
import torch.optim as optim
from configSettings import *
from layerUtils import *
from backPropUtils import *
from LoRA import *
# --------------------------------------------------------
class SimpleSelfAttention(nn.Module):
	def __init__(self):
		super().__init__()

		self.embedding_tok = nn.Embedding(VOCAB_SIZE, D_FTS_MAPS)
		self.embedding_pos = nn.Embedding(SEQ_LEN, D_FTS_MAPS)
		self.attention     = nn.MultiheadAttention(embed_dim=D_FTS_MAPS, num_heads=NUM_HEADS, batch_first=True)
		self.ln_1          = nn.LayerNorm(D_FTS_MAPS, eps=0)
		self.ln_2          = nn.LayerNorm(D_FTS_MAPS, eps=0)
		self.ln_final      = nn.LayerNorm(D_FTS_MAPS, eps=0)
		self.fc_logits     = LoRALayer(D_FTS_MAPS, VOCAB_SIZE, LORA_RANK) if LORA_ON else nn.Linear(D_FTS_MAPS, VOCAB_SIZE)
		self.fc_expand     = nn.Linear(D_FTS_MAPS, 4 * D_FTS_MAPS)
		self.fc_contract   = nn.Linear(4 * D_FTS_MAPS, D_FTS_MAPS)
		self.activation    = nn.ReLU()

		self.register_buffer("mask", self._generate_mask())

	def forward(self, a0):
		a1_tok = self.embedding_tok(a0) 
		a1_pos = self.embedding_pos(torch.arange(SEQ_LEN)).unsqueeze(0)
		a1 = a1_tok + a1_pos

		a2 = self.ln_1(a1)

		# no a3 since pytorch already has out projection built in the nn.MultiheadAttention

		a4, _ = self.attention(a2, a2, a2, attn_mask=self.mask[:a1.shape[1], :a1.shape[1]])
		#a4, _ = self.attention(a2, a2, a2)

		a5 = a1 + a4
		a6 = self.ln_2(a5)
		a7 = self.fc_expand(a6)
		a8 = self.activation(a7)
		a9 = self.fc_contract(a8)

		a10 = a5 + a9
		a11 = self.ln_final(a10)
		a = self.fc_logits(a11)
		return a0[0], a1[0], a2[0], a4[0], a5[0], a6[0], a7[0], a8[0], a9[0], a10[0], a11[0], a[0]

	def _generate_mask(self):
		mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN) * float('-inf'), diagonal=1)
		return mask 

	def laurent_forward(self, a0):
		my_a1_tok = self.embedding_tok(a0)
		my_a1_pos = self.embedding_pos(torch.arange(SEQ_LEN))
		my_a1 = my_a1_tok + my_a1_pos

		assert torch.abs(my_a1_tok - torch.nn.functional.one_hot(a0[0], num_classes = VOCAB_SIZE).float() @ self.embedding_tok.weight).max() < MAX_ERROR
		assert torch.abs(my_a1_pos - torch.nn.functional.one_hot(torch.arange(SEQ_LEN), num_classes = SEQ_LEN).float() @ self.embedding_pos.weight).max() < MAX_ERROR

		my_a2 = self.ln_1(my_a1)
		assert torch.abs(myLN(myLN_noParams(my_a1[0]), self.ln_1.weight, self.ln_1.bias) - my_a2[0]).max() < MAX_ERROR

		my_Qs, my_Ks, my_Attns, my_Vs, my_Cs = [], [], [], [], []
		for h in range(NUM_HEADS):
			w_qh, w_kh, w_vh = getHeadParams(self.attention.in_proj_weight, h, HEAD_D_FTS_MAPS, 'weights')
			b_qh, b_kh, b_vh = getHeadParams(self.attention.in_proj_bias,	h, HEAD_D_FTS_MAPS, 'biases')
			my_Q, my_K, my_V = (my_a2[0] @ w_qh + b_qh), (my_a2[0] @ w_kh + b_kh), (my_a2[0] @ w_vh + b_vh)
			my_Attn = (my_Q @ my_K.T) / torch.sqrt(torch.tensor(HEAD_D_FTS_MAPS))
			my_Attn = torch.tril(my_Attn) + torch.triu(torch.full_like(torch.zeros(SEQ_LEN, SEQ_LEN), -torch.inf), diagonal = 1)
			my_Attn = torch.softmax(my_Attn, 1)
			my_C = my_Attn @ my_V
			my_Qs.append(my_Q); my_Ks.append(my_K); my_Attns.append(my_Attn); my_Vs.append(my_V); my_Cs.append(my_C)

		my_a3 = torch.cat(my_Cs, dim=1)
		my_a3 = my_a3.unsqueeze(0)

		my_a4 = my_a3 @ self.attention.out_proj.weight.T + self.attention.out_proj.bias
		my_a5 = my_a1 + my_a4

		my_a6 = self.ln_2(my_a5)
		assert torch.abs(myLN(myLN_noParams(my_a5[0]), self.ln_2.weight, self.ln_2.bias) - my_a6[0]).max() < MAX_ERROR

		my_a7 = self.fc_expand(my_a6)
		assert torch.abs(my_a6 @ self.fc_expand.weight.T + self.fc_expand.bias - my_a7).max() < MAX_ERROR

		my_a8 = self.activation(my_a7)
		
		my_a9 = self.fc_contract(my_a8)
		assert torch.abs(my_a8 @ self.fc_contract.weight.T + self.fc_contract.bias - my_a9).max() < MAX_ERROR

		my_a10 = my_a5 + my_a9

		my_a11 = self.ln_final(my_a10)
		assert torch.abs(myLN(myLN_noParams(my_a10[0]), self.ln_final.weight, self.ln_final.bias) - my_a11[0]).max() < MAX_ERROR

		my_a = self.fc_logits(my_a11)

		if LORA_ON:
			assert torch.abs(LORA_RANK * (my_a11 @ self.fc_logits.D @ self.fc_logits.U) - my_a).max() < MAX_ERROR
		else:
			assert torch.abs(my_a11 @ self.fc_logits.weight.T + self.fc_logits.bias - my_a).max() < MAX_ERROR

		return a0[0], my_a1[0], my_a2[0], my_Qs, my_Ks, my_Vs, my_Attns, my_a3[0], my_a4[0], my_a5[0], my_a6[0], my_a7[0], my_a8[0], my_a9[0], my_a10[0], my_a11[0], my_a[0]
# --------------------------------------------------------
model = SimpleSelfAttention()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
# --------------------------------------------------------
for epoch in range(6):
	# --------------------------------------------------------
	assert BATCH_SIZE == 1 and NUM_HEADS > 1
	# --------------------------------------------------------
	tokens, targets = generate_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
	optimizer.zero_grad()
	a0, a1, a2, a4, a5, a6, a7, a8, a9, a10, a11, a = model(tokens)
	loss = criterion(a.view(-1, VOCAB_SIZE), targets.view(-1))  
	loss.backward()
	# --------------------------------------------------------
	y_pred = torch.softmax(a, -1)
	y_pred = y_pred.reshape(BATCH_SIZE * SEQ_LEN, VOCAB_SIZE)
	y_gt = OHE_expander(targets.reshape(BATCH_SIZE * SEQ_LEN), num_classes = VOCAB_SIZE)
	lossVerif = (- y_gt * torch.log(y_pred)).sum() / SEQ_LEN #No ignored tokens in loss: look here otherwise -> LLMsScratch/ch07/previous_chapters.py
	assert torch.abs(lossVerif - loss).max() < MAX_ERROR
	# --------------------------------------------------------
	my_a0, my_a1, my_a2, my_Qs, my_Ks, my_Vs, my_Attns, my_a3, my_a4, my_a5, my_a6, my_a7, my_a8, my_a9, my_a10, my_a11, my_a = model.laurent_forward(tokens)
	assert len(my_Qs) == len(my_Ks) == len(my_Vs) == len(my_Attns) == NUM_HEADS

	# because a3 is separated from MHA, we cannot check for it.  But we need it in backpropagation...
	lhs_chk, rhs_chk = [a0, a1, a2, a4, a5, a6, a7, a8, a9, a10, a11, a], [my_a0, my_a1, my_a2, my_a4, my_a5, my_a6, my_a7, my_a8, my_a9, my_a10, my_a11, my_a]
	assert len(lhs_chk) == len(rhs_chk) 
	for _ in zip(lhs_chk, rhs_chk):
		assert torch.abs(_[0] - _[1]).max() < MAX_ERROR
	# --------------------------------------------------------
	my_delta12 = y_pred - y_gt
	# --------------------------------------------------------
	if LORA_ON:
		my_delta11 = LORA_RANK * (my_delta12 @ (model.fc_logits.D @ model.fc_logits.U).T)
		my_fc_logits_D_grad = LORA_RANK * (my_a11.T @ my_delta12 @ model.fc_logits.U.T) / SEQ_LEN
		my_fc_logits_U_grad = LORA_RANK * ((my_a11 @ model.fc_logits.D).T @ my_delta12) / SEQ_LEN
		equalChecker(my_fc_logits_D_grad, model.fc_logits.D.grad, partialL('d_(lora)'))
		equalChecker(my_fc_logits_U_grad, model.fc_logits.U.grad, partialL('u_(lora)'))
	else:
		my_delta11 = deltaFC(my_delta12, model.fc_logits.weight)
		my_fc_logits_w_grad = (my_a11.T @ my_delta12) / SEQ_LEN
		my_fc_logits_b_grad = my_delta12.sum(0) / SEQ_LEN
		equalChecker(my_fc_logits_w_grad, model.fc_logits.weight.grad.T, partialL('w_(fc)'))
		equalChecker(my_fc_logits_b_grad, model.fc_logits.bias.grad, partialL('b_(fc)'))
	# --------------------------------------------------------
	my_ln_final_w_grad = torch.diag(myLN_noParams(my_a10).T @ my_delta11) / SEQ_LEN
	my_ln_final_b_grad = my_delta11.sum(0) / SEQ_LEN
	equalChecker(my_ln_final_w_grad, model.ln_final.weight.grad, partialL('w_(ln_fnl)'))
	equalChecker(my_ln_final_b_grad, model.ln_final.bias.grad, partialL('b_(ln_fnl)'))
	# --------------------------------------------------------
	my_delta10 = deltaLN(my_delta11, model.ln_final.weight, my_a10)
	# --------------------------------------------------------
	my_fc_contract_w_grad = (my_a8.T @ my_delta10) / SEQ_LEN
	my_fc_contract_b_grad = my_delta10.sum(0) / SEQ_LEN
	equalChecker(my_fc_contract_w_grad, model.fc_contract.weight.grad.T, partialL('w_(cntrct)'))
	equalChecker(my_fc_contract_b_grad, model.fc_contract.bias.grad, partialL('b_(cntrct)'))
	# --------------------------------------------------------
	my_delta8 = deltaFC(my_delta10, model.fc_contract.weight)
	my_delta7 = deltaActivation(my_delta8, my_a7)
	# --------------------------------------------------------
	my_fc_expand_w_grad = (my_a6.T @ my_delta7) / SEQ_LEN
	my_fc_expand_b_grad = my_delta7.sum(0) / SEQ_LEN
	equalChecker(my_fc_expand_w_grad, model.fc_expand.weight.grad.T, partialL('w_(expnd)'))
	equalChecker(my_fc_expand_b_grad, model.fc_expand.bias.grad, partialL('b_(expnd)'))
	# --------------------------------------------------------
	my_delta6 = deltaFC(my_delta7, model.fc_expand.weight)
	# --------------------------------------------------------
	my_ln_2_w_grad = torch.diag(myLN_noParams(my_a5).T @ my_delta6) / SEQ_LEN
	my_ln_2_b_grad = my_delta6.sum(0) / SEQ_LEN
	equalChecker(my_ln_2_w_grad, model.ln_2.weight.grad, partialL('w_(ln_2)'))
	equalChecker(my_ln_2_b_grad, model.ln_2.bias.grad, partialL('b_(ln_2)'))
	# --------------------------------------------------------
	my_delta5 = my_delta10 + deltaLN(my_delta6, model.ln_2.weight, my_a5)
	# --------------------------------------------------------
	my_fc_outProj_w_grad = (my_a3.T @ my_delta5) / SEQ_LEN
	my_fc_outProj_b_grad = my_delta5.sum(0) / SEQ_LEN
	equalChecker(my_fc_outProj_w_grad, model.attention.out_proj.weight.grad.T, partialL('w_(attProj)'))
	equalChecker(my_fc_outProj_b_grad, model.attention.out_proj.bias.grad, partialL('b_(attProj)'))
	# --------------------------------------------------------
	my_delta3 = deltaFC(my_delta5, model.attention.out_proj.weight)
	# --------------------------------------------------------
	# MHA
	# --------------------------------------------------------
	my_Q_1, my_Q_2 = my_Qs[0], my_Qs[1]
	my_K_1, my_K_2 = my_Ks[0], my_Ks[1]
	my_V_1, my_V_2 = my_Vs[0], my_Vs[1]
	my_Attn_1, my_Attn_2 = my_Attns[0], my_Attns[1]

	my_delta3_head_1 = my_delta3[:, 0 * HEAD_D_FTS_MAPS : 1 * HEAD_D_FTS_MAPS]
	my_delta3_head_2 = my_delta3[:, 1 * HEAD_D_FTS_MAPS : 2 * HEAD_D_FTS_MAPS]

	w_qh_grad_1, w_kh_grad_1, w_vh_grad_1 = getHeadParams(model.attention.in_proj_weight.grad, 0, HEAD_D_FTS_MAPS, 'weights')
	b_qh_grad_1, b_kh_grad_1, b_vh_grad_1 = getHeadParams(model.attention.in_proj_bias.grad,   0, HEAD_D_FTS_MAPS, 'biases')

	w_qh_grad_2, w_kh_grad_2, w_vh_grad_2 = getHeadParams(model.attention.in_proj_weight.grad, 1, HEAD_D_FTS_MAPS, 'weights')
	b_qh_grad_2, b_kh_grad_2, b_vh_grad_2 = getHeadParams(model.attention.in_proj_bias.grad,   1, HEAD_D_FTS_MAPS, 'biases')

	w_qh_1, w_kh_1, w_vh_1 = getHeadParams(model.attention.in_proj_weight, 0, HEAD_D_FTS_MAPS, 'weights')
	w_qh_2, w_kh_2, w_vh_2 = getHeadParams(model.attention.in_proj_weight, 1, HEAD_D_FTS_MAPS, 'weights')

	my_w_vh_grad_1, my_b_vh_grad_1, my_w_qh_grad_1, my_q_bh_grad_1, my_w_kh_grad_1, my_b_kh_grad_1, delta_im1_1 = selfAttnBack(my_delta3_head_1, my_a2, my_Attn_1, my_V_1, my_K_1, my_Q_1, w_vh_1, w_qh_1, w_kh_1)
	my_w_vh_grad_2, my_b_vh_grad_2, my_w_qh_grad_2, my_q_bh_grad_2, my_w_kh_grad_2, my_b_kh_grad_2, delta_im1_2 = selfAttnBack(my_delta3_head_2, my_a2, my_Attn_2, my_V_2, my_K_2, my_Q_2, w_vh_2, w_qh_2, w_kh_2)
	# --------------------------------------------------------
	print('\n')
	equalChecker(my_w_vh_grad_1, w_vh_grad_1, 'MHA ' + partialL('w_v h=1'))
	equalChecker(my_b_vh_grad_1, b_vh_grad_1, 'MHA ' + partialL('b_v h=1'))
	equalChecker(my_w_qh_grad_1, w_qh_grad_1, 'MHA ' + partialL('w_q h=1'))
	equalChecker(my_q_bh_grad_1, b_qh_grad_1, 'MHA ' + partialL('b_q h=1'))
	equalChecker(my_w_kh_grad_1, w_kh_grad_1, 'MHA ' + partialL('w_k h=1'))
	#equalChecker(my_b_kh_grad_1, b_kh_grad_1, partialL('b_k h=1'))
	# we also that the gradients for the biases of the keys are zero
	equalChecker(B_KH_GRAD_ZERO, b_kh_grad_1, 'MHA ' + partialL('b_k h=1'))

	equalChecker(my_w_vh_grad_2, w_vh_grad_2, 'MHA ' + partialL('w_v h=2'))
	equalChecker(my_b_vh_grad_2, b_vh_grad_2, 'MHA ' + partialL('b_v h=2'))
	equalChecker(my_w_qh_grad_2, w_qh_grad_2, 'MHA ' + partialL('w_q h=2'))
	equalChecker(my_q_bh_grad_2, b_qh_grad_2, 'MHA ' + partialL('b_q h=2'))
	equalChecker(my_w_kh_grad_2, w_kh_grad_2, 'MHA ' + partialL('w_k h=2'))
	#equalChecker(my_b_kh_grad_2, b_kh_grad_2, partialL('b_k h=2'))
	# we also that the gradients for the biases of the keys are zero
	equalChecker(B_KH_GRAD_ZERO, b_kh_grad_2, 'MHA ' + partialL('b_k h=2'))
	print('\n')
	# --------------------------------------------------------
	my_delta2 = delta_im1_1 + delta_im1_2
	# --------------------------------------------------------
	my_ln_1_w_grad = torch.diag(myLN_noParams(my_a1).T @ my_delta2) / SEQ_LEN
	my_ln_1_b_grad = my_delta2.sum(0) / SEQ_LEN
	equalChecker(my_ln_1_w_grad, model.ln_1.weight.grad, partialL('w_(ln_1)'))
	equalChecker(my_ln_1_b_grad, model.ln_1.bias.grad, partialL('b_(ln_1)'))
	# --------------------------------------------------------
	my_delta1 = my_delta5 + deltaLN(my_delta2, model.ln_1.weight, my_a1)
	# --------------------------------------------------------
	my_a0_OHE_tok = torch.nn.functional.one_hot(my_a0, num_classes=VOCAB_SIZE).float()
	my_a0_OHE_pos = torch.nn.functional.one_hot(torch.arange(SEQ_LEN), num_classes=SEQ_LEN).float()
	# --------------------------------------------------------
	my_emdbed_tok_grad = (my_a0_OHE_tok.T @ my_delta1) / SEQ_LEN
	my_emdbed_pos_grad = (my_a0_OHE_pos.T @ my_delta1) / SEQ_LEN
	# --------------------------------------------------------
	equalChecker(my_emdbed_tok_grad, model.embedding_tok.weight.grad, partialL('w_(tok)'))
	equalChecker(my_emdbed_pos_grad, model.embedding_pos.weight.grad, partialL('w_(pos)'))
	# --------------------------------------------------------
	optimizer.step()
	print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}\n\n")
	# --------------------------------------------------------
