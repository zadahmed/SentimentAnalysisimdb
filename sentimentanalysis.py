import tflearn
from tflearn.data_utils import to_categorical , pad_sequences
from tflearn.datasets import imdb

#imdb dataset loading
train , test  = imdb.load_data(path = 'imdb.pkl' , n_words = 10000 , valid_portion = 0.1)
#valid portion is 0.1 since we only want 10% of the data and we take 10000 as the number of words we want to train in the bag of words

trainX , trainY = train
testX , testY = test
#split the training set and testing set into two

#Data preprocessing
#sequence padding
#vectorize our inputs thats why we use sequence padding
#convert our input into matrices and pad it
trainX = pad_sequences(trainX , maxlen - 100  , value = 0.)
testX = pad_sequences(testX , maxlen - 100  , value = 0.)

#Converting labels into vectors
trainY = to_categorical(trainY , nb_classes = 2)
testY = to_categorical(testY , nb_classes = 2)

#network building
net  = tflearn.input_data([None , 100])
net  = tflearn.embedding(net , input_dim=10000 , output_dim=128) #creating word embeddings
net = tflearn.lstm(net , 128 , dropout=0.8) #remember data from sequences  long short term memory
#dropout is used to prevent overfitting by randomly turning neural networks on and off
net  = tflearn.fully_connected(net , 2 , activation = 'softmax')
#using softmax to get a prob between 0 to 1
net = tflearn.regression(net , optimizer='adam' , learning_rate=0.0001,loss='categorical_crossentropy')


#training the neural net by calling the deep neural network function
model = tflearn.DNN(net , tensorboard_verbose=0)
model.fit(trainX , trainY , validation_set=(testX , testY) , show_metric=True , batch_size=32)
