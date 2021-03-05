# keras nn model
# BioVec
input_ = Input(shape = (300,))
x = Dense(1024, activation = "relu")(input_)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x) 
out = Dense(num_classes, activation = 'softmax')(x)
model = Model(input_, out)