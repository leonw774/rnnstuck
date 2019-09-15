# fetch posts
PAGE_LENGTH_MAX = 500 # set None to be unlimited
PAGE_LENGTH_MIN = 12
LINE_LENGTH_MAX = 50
LINE_LENGTH_MIN = 2

# w2v setting
W2V_BY_VOCAB = True # if False: Create w2v model by each character
USE_START_MARK = True # if False: starting indicator would be zero vector
START_MARK = 'š'
USE_ENDING_MARK = True
ENDING_MARK = "ê"
W2V_MIN_COUNT_BY_VOCAB = 7
W2V_MIN_COUNT_BY_CHAR = 3
W2V_ITER = 5
WV_SIZE = 120

# training data configure
ZERO_OFFSET = True
USE_SEQ_LABEL = True
USE_SAVED_MODEL = False
SAVE_MODEL_NAME = "rnnstuck_model.h5"

# LSTM setting
MAX_TIMESTEP = None # set None to be unlimited
RNN_UNIT = [24, 24]
USE_ATTENTION = True or MAX_TIMESTEP != None
VOCAB_SIZE = -1
BATCH_SIZE = 64
EPOCHS = 20
VALIDATION_NUMBER = 100

OUTPUT_NUMBER = 4
OUTPUT_TIME_STEP = 128

STEP_EPOCH_RATE = 0.5
LEARNING_RATE = 0.002
LR_DECAY = 0.707
LR_DECAY_INTV = 2
LR_DECAY_POW_MAX = 8

EARLYSTOP_MIN_DELTA = 0.01
EARLYSTOP_PATIENCE = 4 if (EPOCHS < 12) else EPOCHS // 4

