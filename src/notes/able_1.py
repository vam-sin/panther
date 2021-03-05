'''
ABLE1: loss: 1.3829 - accuracy: 0.6167 - sensitivity: 0.4856 - specificity: 1.0000 - val_loss: 1.7264 - 
val_accuracy: 0.5373 - val_sensitivity: 0.4305 - val_specificity: 1.0000
filename: able_53-73.h5
150 epochs
Bio Vec
'''
# LSTM model; cannot use batchnormalization on this
input_ = Input(shape = (3,100,))
x = Bidirectional(LSTM(128, activation = 'tanh', return_sequences = True))(input_)
x = Bidirectional(LSTM(128, activation = 'tanh', return_sequences = True))(x)
x = SeqSelfAttention(attention_activation='tanh')(x)
x = Flatten()(x)
out = Dense(num_classes, activation = 'softmax')(x)
model = Model(input_, out)