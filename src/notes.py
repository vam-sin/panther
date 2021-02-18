# ANN1: Val Acc: 0.11162, filename: ann_11-162.h5 
input_ = Input(shape = (300,))
x = Dense(1024, activation = "relu")(input_)
x = BatchNormalization()(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
out = Dense(num_classes, activation = 'softmax')(x)
model = Model(input_, out)

'''
ANN2: loss: 2.8093 - accuracy: 0.3794 - sensitivity: 0.2100 - specificity: 1.0000 - 
val_loss: 1.7272 - val_accuracy: 0.5274 - val_sensitivity: 0.3213 - val_specificity: 1.0000, 
filename: ann_52-738.h5 
Notes: Extremely regularized, too much dropout
'''
input_ = Input(shape = (300,))
x = Dense(1024, activation = "relu")(input_)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
out = Dense(num_classes, activation = 'softmax')(x)
model = Model(input_, out)

'''
ANN3: loss: 1.3972 - accuracy: 0.6119 - sensitivity: 0.5052 - specificity: 1.0000 - 
val_loss: 1.0776 - val_accuracy: 0.6589 - val_sensitivity: 0.5692 - val_specificity: 1.0000
filename: ann_65-89.h5
Notes: More regularization than needed
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
x = Dropout(0.3)(x) 
out = Dense(num_classes, activation = 'softmax')(x)
model = Model(input_, out)

'''
ANN4: loss: 0.9166 - accuracy: 0.6976 - sensitivity: 0.6195 - specificity: 1.0000 - 
val_loss: 0.9535 - val_accuracy: 0.6588 - val_sensitivity: 0.6051 - val_specificity: 1.0000
filename: ann_65-876.h5
Notes: Little Overfitting, but that's chill.
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