
from keras.layers import Dense, Dropout
from keras.models import Model, Sequential
from keras.wrappers.scikit_learn import KerasRegressor


# build your own model
# create a function that returns a model, taking as parameters things you
# want to verify using cross-valdiation and model selection
def create_model(optimizer='adagrad',
                 kernel_initializer='glorot_uniform', 
                 dropout=0.2):
    model = Sequential()
    model.add(Dense(64,activation='relu',kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation='sigmoid',kernel_initializer=kernel_initializer))

    model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])

    return model

# wrap the model using the function you created
clf = KerasRegressor(build_fn=create_model,verbose=0)


##################################################################

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor


##def trn_lrn(num_classes, learning_rate):
##    
##    # Using base model for transfer learning
##    base_model = ResNet50(weights='imagenet', include_top=False,
##                          input_shape=input_shape)
##
##    # classification layer
##    x = base_model.output
##    x = Flatten()(x)
##    final_output = Dense(num_classes, activation='softmax', name='fc')(x)
##
##    # create new model using weights from our base ResNet50 model
##    model = Model(inputs=base_model.input, outputs=final_output)
##
##    model.summary()
##
##    # compile
##    model.compile(loss='categorical_crossentropy',
##                  optimizer=SGD(lr=learning_rate, momentum=0.9, metrics=['accuracy'])


#    return model

# wrap the model using the function you created
#clf = KerasRegressor(build_fn=trn_lrn(num_classes, learning_rate),verbose=0)
