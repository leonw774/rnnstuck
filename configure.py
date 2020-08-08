# fetch posts
WORD_LENGTH_MAX = 512 # set None to be unlimited
WORD_LENGTH_MIN = 4
LINE_LENGTH_MAX = 32
LINE_LENGTH_MIN = 2

# w2v setting
W2V_BY_VOCAB = True # if False: Create w2v model by each character
W2V_MODEL_NAME = "./models/myword2vec_by_word.model" if W2V_BY_VOCAB else "./models/myword2vec_by_char.model"
USE_START_MARK = True # if False: starting indicator would be zero vector
START_MARK = 'š'
USE_ENDING_MARK = True
ENDING_MARK = 'ê'
W2V_MIN_COUNT = 8 if W2V_BY_VOCAB else 4
W2V_ITER = 4
WV_SIZE = 64

# training data configure
ZERO_OFFSET = True
USE_SAVED_MODEL = False
SAVE_MODEL_NAME = "./models/rnnstuck_model.h5"

# Model setting
MAX_TIMESTEP = 16 # set None to be as same as WORD_LENGTH_MAX
USE_CUDNN = False
RNN_UNIT = [200, 200]
USE_BIDIRECTION = False
USE_SEQ_RNN_OUTPUT = False
USE_ATTENTION = False
VOCAB_SIZE = -1
BATCH_SIZE = 256
EPOCHS = 2
VALIDATION_NUMBER = min(WORD_LENGTH_MAX, 400)

OUTPUT_NUMBER = 4
OUTPUT_TIME_STEP = 200
OUTPUT_SAMPLE_TEMPERATURE = 0.6

STEP_EPOCH_RATE = 1.0
OPTIMIZER_NAME = "rmsprop"
LEARNING_RATE = 0.002
LR_DECAY = 0.8
LR_DECAY_INTV = 2
LR_DECAY_POW_MAX = 8

EARLYSTOP_MIN_DELTA = 0.01
EARLYSTOP_PATIENCE = 2 if EPOCHS < 16 else 4

