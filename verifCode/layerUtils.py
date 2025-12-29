import torch
import torch.nn.functional as F

from configSettings import *

def OHE_expander(dataTensor, num_classes):
	return F.one_hot(dataTensor * (dataTensor >= 0), num_classes=num_classes) * (dataTensor.unsqueeze(-1) >= 0)

def equalChecker(a, b, nameD):
	absDiff = torch.abs(a - b)
	maxError, meanError = float(absDiff.max()), float(absDiff.mean())
	assert meanError <= maxError, f"Debug info: meanError={meanError}, maxError={maxError}"
	assert a.shape == b.shape and maxError < MAX_ERROR, f'Max error: {maxError} | Shapes: {a.shape}, {b.shape}'
	print('%s\t\tmax(Err) = %.2e\t<Err> = %.2e\t%s\tweightWatcher = %.2e' % (nameD, maxError, meanError, a.shape, weightWatcher(b)))


def generate_data(batch_size, seq_len, vocab_size):
	#x = torch.randint(1, vocab_size // 2, (batch_size, seq_len))  
	x = torch.randint(0, vocab_size, (batch_size, seq_len))
	y = x + 1
	y[y >= vocab_size] = 1	
	return x, y

def myLN_noParams(x, eps=0):
	return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True, unbiased=False) + eps)

def myLN(x, w, b):
	return x @ torch.diag(w) + b

def getHeadParams(paramTensor, headID, headDim, paramName):
	headSlicer = slice(headID * headDim, (headID + 1) * headDim)
	p_qh = paramTensor[0 * D_FTS_MAPS : 1 * D_FTS_MAPS]
	p_kh = paramTensor[1 * D_FTS_MAPS : 2 * D_FTS_MAPS]
	p_vh = paramTensor[2 * D_FTS_MAPS : 3 * D_FTS_MAPS]
	if paramName == 'weights': 
		# pytorch-specific transpose the weights before being able to slicing out the heads
		p_qh = p_qh.T[:, headSlicer]
		p_kh = p_kh.T[:, headSlicer]
		p_vh = p_vh.T[:, headSlicer]
		assert p_qh.shape == p_kh.shape == p_vh.shape == (D_FTS_MAPS, headDim)
	else:
		assert paramName == 'biases'
		p_qh = p_qh[headSlicer]
		p_kh = p_kh[headSlicer]
		p_vh = p_vh[headSlicer]
		assert p_qh.shape[0] == p_kh.shape[0] == p_vh.shape[0] == headDim
	return p_qh, p_kh, p_vh

