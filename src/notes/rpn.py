'''
loss: 0.6513 - accuracy: 0.7417 - sensitivity: 0.6850 - 
val_loss: 0.4832 - val_accuracy: 0.7476 - val_sensitivity: 0.7119
One-Hot Encoding
'''
def ResBlock(inp):
    x = Conv1D(512, (3), padding="same", activation = "relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(512, (3), padding="same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Add()([x, inp])

    return x

# keras nn model
# (Res-Blocks x k) + (LSTM x 2) + Attention Layer
input_ = Input(shape = (21, max_length,))

x = Conv1D(512, (3), padding="same", activation = "relu")(input_)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Residual Blocks
x = ResBlock(x)
x = ResBlock(x)

# sequence layers
x = Bidirectional(LSTM(128, activation = 'tanh', return_sequences = True))(x)
x = Dropout(0.3)(x)
x = Bidirectional(LSTM(128, activation = 'tanh', return_sequences = True))(x)
x = Dropout(0.3)(x)
x = SeqSelfAttention(attention_activation = "sigmoid")(x)

x = Flatten()(x)

x = Dense(1024, activation = "relu")(x)
x = Dropout(0.5)(x)
out = Dense(num_classes, activation = 'softmax')(x)

model = Model(input_, out)