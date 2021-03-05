'''
ANN: loss: 0.9166 - accuracy: 0.6976 - sensitivity: 0.6195 - specificity: 1.0000 - 
val_loss: 0.9535 - val_accuracy: 0.6588 - val_sensitivity: 0.6051 - val_specificity: 1.0000
filename: ann_65-876.h5
Notes: Little Overfitting, but that's chill.
50 epochs, 256 bs
BioVec
'''
input_ = Input(shape = (300,))
x = Dense(1024, activation = "relu")(input_)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
# x = Dropout(0.3)(x) 
out = Dense(num_classes, activation = 'softmax')(x)
model = Model(input_, out)