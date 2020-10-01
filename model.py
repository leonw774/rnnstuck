from generate import *
from configure import *
from tensorflow.keras import activations, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Activation, Attention, BatchNormalization, Bidirectional, Concatenate, Dense, Dropout, Embedding, Flatten, GRU, Input, Lambda, LSTM, Masking, Multiply, BatchNormalization, Permute, RepeatVector, Reshape

print("\nUSE_SAVED_MODEL:", USE_SAVED_MODEL)
print("max time step: %s\nrnn units: %s" % (MAX_TIMESTEP, RNN_UNIT))
print("\noptimizer: %s\nbatch size: %d\nepoches: %d\nlearning_rate: %f\noutput number:%d" 
    % (OPTIMIZER_NAME, BATCH_SIZE, EPOCHS, LEARNING_RATE, OUTPUT_NUMBER))

### CALLBACK FUNCTIONS ###
def sparse_categorical_perplexity(y_true, y_pred) :
    return K.square(K.sparse_categorical_crossentropy(y_true, y_pred))
              
def lrate_epoch_decay(epoch) :
    init_lr = LEARNING_RATE
    e = min(LR_DECAY_POW_MAX, (epoch + 1) // LR_DECAY_INTV) # INTV epoches per decay, with max e
    return init_lr * (LR_DECAY**e)

### DEFINE MODEL ###
def rnnstuck_model(vocab_size):
    if USE_SAVED_MODEL :
        return load_model(SAVE_MODEL_NAME)
    
    if OPTIMIZER_NAME == "sgd" :
        optier = optimizers.SGD(lr = LEARNING_RATE, momentum = 0.5, nesterov = True)
    elif OPTIMIZER_NAME == "rmsprop" :
        optier = optimizers.RMSprop(lr = LEARNING_RATE)
    elif OPTIMIZER_NAME == "adam" :
        optier = optimizers.Adam(lr = LEARNING_RATE)
    
    ## make model
    input_layer = Input([MAX_TIMESTEP, WV_SIZE])
    rnn_layer = input_layer
    
    for i, v in enumerate(RNN_UNIT) :
        is_return_seq = (i != len(RNN_UNIT) - 1) or USE_ATTENTION or USE_SEQ_RNN_OUTPUT
        if USE_BIDIRECTION :
            rnn_layer = Bidirectional(LSTM(v, return_sequences = is_return_seq))(rnn_layer)
        else :
            rnn_layer = LSTM(v, return_sequences = is_return_seq)(rnn_layer)
        rnn_layer = BatchNormalization()(rnn_layer)
        rnn_layer = Dropout(0.2)(rnn_layer)
    if USE_ATTENTION and MAX_TIMESTEP:
        rnn_layer = Attention()([rnn_layer, rnn_layer]) # self-attention
    if USE_ATTENTION or USE_SEQ_RNN_OUTPUT:
        postproc_layer = Flatten()(rnn_layer) # => (?, (MAX_TIMESTEP*last_rnn_units))
    else :
        postproc_layer = rnn_layer
    postproc_layer = BatchNormalization()(postproc_layer)
    postproc_layer = Dropout(0.2)(postproc_layer)
    guess = Dense(vocab_size, activation = "softmax")(postproc_layer)
    model = Model(input_layer, guess)
    model.compile(
        loss = "sparse_categorical_crossentropy", #sparse_categorical_perplexity,
        optimizer = optier,
        metrics = ["sparse_categorical_accuracy"])
    model.summary()
    return model
