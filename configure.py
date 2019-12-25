# fetch posts
WORD_LENGTH_MAX = 1024 # set None to be unlimited
WORD_LENGTH_MIN = 6
LINE_LENGTH_MAX = 32
LINE_LENGTH_MIN = 2

# w2v setting
W2V_BY_VOCAB = True # if False: Create w2v model by each character
USE_START_MARK = True # if False: starting indicator would be zero vector
START_MARK = 'š'
USE_ENDING_MARK = True
ENDING_MARK = 'ê'
W2V_MIN_COUNT_BY_VOCAB = 10
W2V_MIN_COUNT_BY_CHAR = 4
W2V_ITER = 4
WV_SIZE = 100

# training data configure
ZERO_OFFSET = False
USE_SAVED_MODEL = False
SAVE_MODEL_NAME = "rnnstuck_model.h5"

# Model setting
MAX_TIMESTEP = 20 # set None to be unlimited
RNN_UNIT = [160, 160]
USE_BIDIRECTION = True
USE_SEQ_RNN_OUTPUT = False
USE_ATTENTION = False
VOCAB_SIZE = -1
BATCH_SIZE = 128
EPOCHS = 32
VALIDATION_NUMBER = 100

OUTPUT_NUMBER = 4
OUTPUT_TIME_STEP = 100

STEP_EPOCH_RATE = 0.87
USE_OPTIMIZER = "rmsprop"
LEARNING_RATE = 0.01
LR_DECAY = 0.717
LR_DECAY_INTV = 1
LR_DECAY_POW_MAX = 10

EARLYSTOP_MIN_DELTA = 0.01
EARLYSTOP_PATIENCE = 4 if EPOCHS < 16 else 8

