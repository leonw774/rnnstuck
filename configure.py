# fetch posts
WORD_LENGTH_MAX = 400 # set None to be unlimited
WORD_LENGTH_MIN = 8
LINE_LENGTH_MAX = 40
LINE_LENGTH_MIN = 2

# w2v setting
W2V_BY_VOCAB = True # if False: Create w2v model by each character
USE_START_MARK = True # if False: starting indicator would be zero vector
START_MARK = 'š'
USE_ENDING_MARK = True
ENDING_MARK = 'ê'
W2V_MIN_COUNT_BY_VOCAB = 7
W2V_MIN_COUNT_BY_CHAR = 3
W2V_ITER = 5
WV_SIZE = 120

# training data configure
ZERO_OFFSET = True
USE_SEQ_LABEL = False
USE_SAVED_MODEL = False
SAVE_MODEL_NAME = "rnnstuck_model.h5"

# LSTM setting
MAX_TIMESTEP = 16 # set None to be unlimited
RNN_UNIT = [48]
USE_ATTENTION = True or MAX_TIMESTEP != None
VOCAB_SIZE = -1
BATCH_SIZE = 64
EPOCHS = 100
VALIDATION_NUMBER = 100

OUTPUT_NUMBER = 4
OUTPUT_TIME_STEP = 128

STEP_EPOCH_RATE = 0.9
LEARNING_RATE = 0.002
LR_DECAY = 0.707
LR_DECAY_INTV = 2
LR_DECAY_POW_MAX = 6

EARLYSTOP_MIN_DELTA = 0.01
EARLYSTOP_PATIENCE = 4

