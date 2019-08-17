# fetch posts
PAGE_LENGTH_MAX = None # set None to be unlimited
PAGE_LENGTH_MIN = 32
LINE_LENGTH_MAX = 50
LINE_LENGTH_MIN = 2

# w2v setting
W2V_BY_VOCAB = True # if False: Create w2v model by each character
START_MARK = 'š'
USE_ENDING_MARK = True
ENDING_MARK = "ê"
W2V_MIN_COUNT_BY_VOCAB = 6
W2V_MIN_COUNT_BY_CHAR = 3
W2V_ITER = 6
WV_SIZE = 200

# training data configure
MAX_TIMESTEP = None # set None to be unlimited
FIXED_TIMESTEP = False
if FIXED_TIMESTEP and MAX_TIMESTEP > PAGE_LENGTH_MIN : MAX_TIMESTEP = PAGE_LENGTH_MIN
ZERO_OFFSET = False # only valid when FIXED_TIMESTEP == False

USE_SAVED_MODEL = False
SAVE_MODEL_NAME = "rnnstuck_model.h5"

# LSTM setting
RNN_UNIT = [64] # nvidia gt730 gpu: lstm(300) is limit
USE_ATTENTION = True and MAX_TIMESTEP != None
VOCAB_SIZE = -1
BATCH_SIZE = 128
EPOCHS = 1
VALIDATION_NUMBER = 100

OUTPUT_NUMBER = 4
OUTPUT_TIME_STEP = 128

STEP_EPOCH_RATE = 0.707

