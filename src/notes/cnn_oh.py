'''
One Hot Encoding:
loss: 0.5770 - accuracy: 0.7611 - sensitivity: 0.7184 - specificity: 1.0000 - 
val_loss: 0.4909 - val_accuracy: 0.7654 - val_sensitivity: 0.7349 - val_specificity: 1.0000
cnn_onehot.h5
'''
# keras nn model
input_ = Input(shape = (21, max_length,))
x = Conv1D(512, (3), padding="same", activation = "relu")(input_)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Conv1D(512, (3), padding="same", activation = "relu")(x)
x = Dropout(0.3)(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(1024, activation = "relu")(x)
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

print(model.summary())