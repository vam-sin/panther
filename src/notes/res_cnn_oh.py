'''
loss: 0.5618 - accuracy: 0.7631 - sensitivity: 0.7217 - 
val_loss: 0.4921 - val_accuracy: 0.7638 - val_sensitivity: 0.7324 - lr: 0.0010
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
input_ = Input(shape = (21, max_length,))
x = Conv1D(512, (3), padding="same", activation = "relu")(input_)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = ResBlock(x)
x = ResBlock(x)

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