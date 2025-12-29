import torch
# --------------------------------------------------------
LORA_ON   = False
LORA_RANK = 16
# --------------------------------------------------------
MAX_ERROR = 1e-5
# --------------------------------------------------------
VOCAB_SIZE = 10 #50257 #10 #50
D_FTS_MAPS = 32 #768 #32
SEQ_LEN    = 4 #1024 #4
BATCH_SIZE = 1
NUM_EPOCHS = 6
NUM_HEADS  = 2
# --------------------------------------------------------
HEAD_D_FTS_MAPS = D_FTS_MAPS // NUM_HEADS
assert D_FTS_MAPS % NUM_HEADS == 0, 'D_FTS_MAPS must be divisible by NUM_HEADS'
# --------------------------------------------------------
B_KH_GRAD_ZERO = torch.zeros(HEAD_D_FTS_MAPS)
# --------------------------------------------------------
partialL      = lambda _ : '\u2202' + 'L/' + '\u2202' + _
weightWatcher = lambda _ : float(_.abs().mean())
