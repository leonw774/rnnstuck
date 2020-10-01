# fetch posts
WORD_LENGTH_MAX = 400 # set None to be unlimited
WORD_LENGTH_MIN = 8
LINE_LENGTH_MAX = 100
LINE_LENGTH_MIN = 2

# w2v setting
W2V_BY_VOCAB = True # if False: Create w2v model by each character
W2V_MODEL_NAME = "./models/myword2vec_by_word.model" if W2V_BY_VOCAB else "./models/myword2vec_by_char.model"
USE_START_MARK = True # if False: starting indicator would be zero vector
START_MARK = 'š'
USE_ENDING_MARK = True
ENDING_MARK = 'ê'
W2V_MIN_COUNT = 10 if W2V_BY_VOCAB else 5
W2V_ITER = 4
WV_SIZE = 64

# training data configure
USE_SAVED_MODEL = False
SAVE_MODEL_NAME = "./models/rnnstuck_model.h5"

# Model setting
MAX_TIMESTEP = 24 # set None to be as same as WORD_LENGTH_MAX
RNN_UNIT = [64, 32]
USE_BIDIRECTION = False
USE_SEQ_RNN_OUTPUT = True
USE_ATTENTION = True
VOCAB_SIZE = -1
BATCH_SIZE = 256
EPOCHS = 32
VALIDATION_SPLIT = 12

OUTPUT_NUMBER = 4
OUTPUT_TIMESTEP = min(WORD_LENGTH_MAX, 200)
OUTPUT_SAMPLE_TEMPERATURE = 0.7

STEP_EPOCH_RATE = 1.0
OPTIMIZER_NAME = "adam"
LEARNING_RATE = 0.001
LR_DECAY = 0.71
LR_DECAY_INTV = 2
LR_DECAY_POW_MAX = 8

EARLYSTOP_MIN_DELTA = 0.01
EARLYSTOP_PATIENCE = 2 if EPOCHS < 16 else 4

