# -*- coding: utf-8 -*-
"""
This file is the final project of the CS767.

The model here attempts to predict the next word of a sentance, trained with
Magic the Gathering (MtG) cards. Additionally it attempts to generate novel texts
with the trained model.

Written by Chris Quartararo

Last updated 03/30/2021

"""



#%% Import modules and set up workspace

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras  # ML library
import nltk  # Natural Language Toolkit
import re
import plotly.express as px
from itertools import chain  # Iteration tools for preprocessing

# My specific hardware needs this little section. Too bad I don't have a real beefy
# GPU with a huge vRAM number. Guess I'll just have to spend a million dollars to
# get a brand new one.
# Import tensorflow and set memory to growth to allocated memory as needed
import tensorflow
physical_devices = tensorflow.config.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)


# Set working directory for ease of use
working_dir = 'G:\\Google Drive\\.School\\2020 MS Comp Sci\\[04] Machine Learning - CS767\\Project\\'


#%% Import data set  ## CAN SKIP IF USING SAVED DATA ##

#### Can skip to the next chunk if already run and saved once ####

# Get data set and read it from a json
mtg_json = pd.read_json(working_dir + 'AtomicCards.json')['data']
mtg_json = mtg_json.drop(['date', 'version'])

# Create a list to hold cards in
cards = pd.DataFrame()

# Loop through sets
for c in mtg_json:
    # Turn list into a pandas df
    temp_c = pd.DataFrame(c)
        
    # Append cards to cards list 
    cards = cards.append(temp_c, ignore_index=True)

# Remove cards that have no text
oracle = cards[cards.text.isnull() == False]

# Change text to lower case
oracle.text = oracle.text.str.lower()

# Write data set to a csv so we don't have to do this for loop again
oracle.to_csv(working_dir + "oracle.csv", index=False)
    

#%% Imports a saved dataset to save time

oracle = pd.read_csv(working_dir + "oracle.csv")


#%% Card text investigation

# Find strings with newline characters
oracle[oracle.text.str.contains('\n', na=False)]

# Cat all sentences together
full_text = ""
for c in oracle['text']:
    full_text += " "
    full_text += c

# Get frequency of ccharacters
char_freqs = pd.Series([c for c in full_text]).value_counts()
char_freqs = pd.DataFrame({'char':char_freqs.index, 'freq':char_freqs})
plt.figure(figsize=(12,9))
plot = plt.bar(x=char_freqs['char'], height=char_freqs['freq'])
plt.title("Frequency of Characters in the full text")


# Prints card names with small number of characters
for char in ['☐', '®', 'á', 'â', 'é', 'í', 'ö', 'ú', 'û', 'π', '_']:
    for i, c in enumerate(oracle['text']):
        if char in c:
            print(f"{char} in {oracle['name'][i]}")
            

#%% Preprocess text

# Find strings with newline characters
oracle[oracle.text.str.contains('\n', na=False)]

# Cat all sentences together
full_text = ""
for c in oracle['text']:
    full_text += " "
    full_text += c
 

# Replace scarce characters
full_text = full_text.replace('☐', " ")
full_text = full_text.replace('®', " ")
full_text = full_text.replace('á', "a")
full_text = full_text.replace('â', "a")
full_text = full_text.replace('é', "e")
full_text = full_text.replace('í', "i")
full_text = full_text.replace('ö', "o")
full_text = full_text.replace('ú', "u")
full_text = full_text.replace('û', "u")
full_text = full_text.replace('π', "pi")
full_text = full_text.replace('½', "half")
full_text = full_text.replace('∞', "infinity")
full_text = full_text.replace('_', "")

    
## Character based model
chars = sorted(list(set(full_text)))
chars_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
vocab_size = len(chars)

"""
## Word based model    
# Remove commas
full_text = full_text.replace(",", "")
# Remove apostrophes
full_text = full_text.replace("'", "")
# Remove new lines
full_text = full_text.replace("\n", " ")
"""

# Tokenize text with NLTK
words_all = nltk.word_tokenize(full_text)
#oracle['tokens'] = oracle.text.apply(nltk.word_tokenize)

# Create a single list of words in the oracle.text column
#words_all = list(chain.from_iterable(oracle.tokens))

## Remove stop words
#stopwords = set(nltk.corpus.stopwords.words('english'))
#words_filtered = [word for word in words_all if not word in stopwords]

## Stem words
#stemmer = nltk.stem.PorterStemmer()
#words_filtered = [stemmer.stem(word) for word in words_filtered]




#%% Create sentence slices

## Parameters
SEQ_LEN = 100  # How long each "sentence" will be
STEP = 10  # How far to move to create the next "sentence"

sentances = []  # n-gram words or input values

## Create sequences of length SEQ_LEN + 1 to also include the next word prediction
# We will be splitting these "sentences" into the inputs and outputs with a Keras tokenizer in the next step
for i in range(0, len(full_text) - SEQ_LEN, STEP):
    sentances.append([chars_to_int[c] for c in full_text[i: i + SEQ_LEN + 1]])  # sequences

# Turn these into np arrays to work with keras
sentances = np.array(sentances)

# Create x and y sequences for character based model
x, y = sentances[:, :-1], sentances[:, -1]

# We'll try to use sparse CCE for performance reasons
#y = keras.utils.to_categorical(y, dtype="int8")




#%% Token statistics  ## WORD BASED ##

freqs_df = pd.Series(words_all).value_counts()
#n, bins, patches = plt.hist(x=words_all)


#%% Encode the words with Keras Tokenizer  ## WORD BASED ##

#### Only needs to be run with a word based model #####

# Store some important values to be used later
vocab_size = len(freqs_df)-sum(freqs_df <=1)  # word_index starts at key value 1 so we need to increase the length by 1

## Tokenize the words, ie assign each word a unique index
# Create tokenizer
# Vocab length is the size of all vocab words that appear more than once
tokenizer = keras.preprocessing.text.Tokenizer(num_words=len(freqs_df)-sum(freqs_df <=1), oov_token='oov')
tokenizer.fit_on_texts(sentances.tolist())  # Can't feed tokenizer np array so convert to python list
token_words = tokenizer.texts_to_sequences(sentances.tolist())  # Output the tokenized words 

# Make the token_words into an np array
token_words = np.array(token_words) 
# Array is too big, reduce the size
token_words = token_words.astype("int16")

# Create input and output sequesnces
x, y = token_words[:, :-1], token_words[:,-1]

# Change y values to categorical 
# The array is too big as what keras outputs (float32) to wrangle so I had to make it 8-bit ints.
# Not a problem as this is a one-hot vector so it's ranged [0-1]. Using a character based model
# would have probably sidestepped this issue.
#
# What a great excuse to buy new computer parts though ;)
y = keras.utils.to_categorical(y, dtype="int8")  # One hot encode the output variables in a matrix of size vocab_size 


#%% Create training and testing sets

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.2)


#%% Character freqency in training set

char_freqs_train = pd.Series([c for c in y_train]).value_counts()
char_freqs_train = pd.DataFrame({'char':[int_to_char[c] for c in char_freqs_train.index], "freq":char_freqs_train})
plt.figure(figsize=(12,9))
plot_train = plt.bar(x=char_freqs_train['char'], height=char_freqs_train['freq'])
plt.title("Frequency of characters as labels in the training set")

#%% Build the model

"""
To Do:
    - Play with regularization
    - Play with number of LSTM layers
    - Play with number of LSTM modules
    - Add 1D Conv layers
"""

# We're going to use sequential API to make it easy unless we need some crazy architechture.
# We're also going to make this a function so that we can wrap it into some sklearn stuff for searching

def build_model(num_nodes=200, activation='adam', layers=[True, False], 
                dropout_rate=.2, 
                num_before_layers = 1, 
                num_after_layers = 0,
                embedding_size=50):
    model = keras.Sequential()
    
    ## Add layers
    # Start with embeddings. We take the vocab size and reduce it to a size of 200
    # Hopefully this will allow me to actually run the thing.
    model.add(keras.layers.Embedding(vocab_size, embedding_size))
    model.add(keras.layers.Dropout(dropout_rate))
    

    
    # Let's try a 1d Convolution to see if we can avoid repettitive patterns
#    model.add(keras.layers.Conv1D(50, 5, padding='same', activation='relu'))
#    model.add(keras.layers.BatchNormalization())
    
        # Dense layers
    for i in range(0,num_before_layers):
        model.add(keras.layers.Dense(100, activation='relu'))
        model.add(keras.layers.BatchNormalization())
    
    # Then we add the LSTM layers
    # Currently we have one LSTM layer that does not return its input as we need to give that output to the next layer
    for i in layers:
        model.add(keras.layers.LSTM(num_nodes, return_sequences=i))
        model.add(keras.layers.Dropout(dropout_rate))  # Needs dropout because we can't do batch normalization
    
    # Dense layers
    for i in range(0,num_after_layers):
        model.add(keras.layers.Dense(100, activation='relu'))
        model.add(keras.layers.BatchNormalization())
    
    # Then we add dense output layer
    model.add(keras.layers.Dense(units=vocab_size, activation='softmax'))
    
    
    ## Compiling
    # We are going to start with RMSProp with no momentum for optimization and see how we do.
    # The loss will be categorical crossentropy since I believe that will allow a better representation
    # of the error. Outputs have been one hot encoded already to accomodate this loss function.
    model.compile(optimizer=activation, 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])
    
    # Return the model to be used
    return model

mtgLSTM = build_model()


#%% Fit the model!

# Create an early stopping callback with patience of 2
early_stopping = keras.callbacks.EarlyStopping(patience=2,
                                               restore_best_weights=True)

# Fit the model
# We are using a validation set of .2 so that we can do early stopping
with tensorflow.device('/cpu:0'):
    history = mtgLSTM.fit(x_train, y_train, 
                          validation_split=0.2,
                          epochs=5, 
                          batch_size=128,
                          callbacks=[early_stopping])

#%% Save the model

# Save model to working directory
mtgLSTM.save(working_dir + "mtgLSTM_classic_18epochs_Stopped")

#%% Load the model

## Loads model weights
# Embedding layer size 50
# Dense layer size 100
# 2 LSTM layers size 200
# Output dense layer
mtgLSTM = keras.models.load_model(working_dir + "mtgLSTM_classic_18epochs_Stopped")

#%% Evaluate model

# Handy little helper method
def evaluate(mode, inputs, outputs):
    print("Evaluation of the model")
    
    results = mtgLSTM.evaluate(inputs, outputs, batch_size=128)
                               
    print(f"Loss: {results[0]}")
    print(f"Accuracy: {results[1]}")

# Run the evaluation and print the results
evaluate(mtgLSTM, x_test, y_test)

#%% Confusion matrix

# Testing predictions
predictions = mtgLSTM.predict(x_test)
predictions_test = np.apply_along_axis(np.argmax, axis=1, arr=predictions)

# Function to create a nice confusion matrix with the testing sets
def confusion_matrix(inputs=x_test, labels=y_test):
    matrix = tensorflow.math.confusion_matrix(labels, predictions_test).numpy()
    matrix_norm = np.around(matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis], decimals=2)
    classes = int_to_char.values()
    matrix_df = pd.DataFrame(matrix_norm,
                     index = classes, 
                     columns = classes)
    # Annotated Confusion Matrix
    plt.figure(figsize=(30,16))
    sns.heatmap(matrix_df, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    # Non annotated Confustion Matrix
    plt.figure(figsize=(30,16))
    sns.heatmap(matrix_df, annot=False,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

confusion_matrix()

"""
Evaluation performance
----------------------
All models tested have a Dense output layer activated with softmax with the shape of vocabulary to predict the output word.


1. Simple LSTM model - Accuracy: 64%
This model has 3 layers, an embedding layer, a single LSTM layer with 200 nodes, and a dense output layer.
Takes an input sequence of length 10

2. 2 Layer LSTM - Accuracy: 65.7%
This model has 4 layers, an embedding layer, two LSTM layers with 200 nodes, and a dense output layer.
Takees an input sequence of length 10

3. 2 layer LSTM - Accuracy: 65.3%
This model has 2 layers, an embedding layer, X LSTM layers with 200 nodes, and a dense output layer.
Takes an input of 20 words

4. 4 Layer LSTM - Accuracy: 65.4%
This model has 5 layers, an embedding layer, four LSTM layers with 200 nodes, and a dense output layer.
Takes an input sequence of length 10

5. 4 Layer LSTM - Accuracy: 
This model has 5 layers, an embedding layer, four LSTM layers with 50 nodes, and a dense output layer.
Takes an input sequence of length 10

6. 2 Layer LSTM - Accuracy: 79.6%
This model was moved to a character based model and uses sparse CCE. Note that this model was not trained on
the entirety of the data set for the sake of time, the training set contains only 300,000 instances rather than the 
potential ~3 million sets that will be used to train the final model.

"""

#%% Attempt to do searching on the hyperparameter space

# Create an early stopping callback with patience of 2
early_stopping = keras.callbacks.EarlyStopping(patience=2,
                                               restore_best_weights=True)

# Create wrapper for sklearn
model_wrapper = keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_model)

# Create parameter search spaces
param_grid = {'epochs':[20],
              'batch_size':[512],
              'layers':[[True, False]],
              'num_nodes':[200],
              'num_before_layers':[0,1],
              'num_after_layers':[0,1],
              'embedding_size':[5,10,50]
              }

# Set up the grid search
gs = sklearn.model_selection.GridSearchCV(estimator=model_wrapper, 
                                          param_grid=param_grid, 
                                          cv=5,
                                          verbose=3)

# Perform the grid search
gs.fit(x_train, y_train,
       validation_split=0.2,
       callbacks=[early_stopping],
       verbose=2)

# Print the best parameters
print(gs.best_params_)


#%% Methods for generating outputs

# function to easily return a letter
def predict_letter(input, model=mtgLSTM):
    return np.argmax(model.predict(input.reshape(1,input.shape[0])))

# function to return a random letter from the probability distribution of predictions
def predict_random_letter(input, model=mtgLSTM):
    preds = model.predict(input.reshape(1,input.shape[0]))
    if np.random.choice(range(2), p = [.32,.68]):
        preds = preds.reshape(preds.shape[1])
        pred = np.random.choice(range(0,len(chars)), p=preds)
    else:
        pred = np.argmax(preds)
    return pred

# Creates a prediction of text based on the model from a seed
def generate(seed, length):
    
    # Holds predictions
    preds = ""

    # Input to the predictor
    input = seed

    # Sets up a loop for length "length"
    for i in range(length):
        # Get the prediction from the input
        prediction = predict_random_letter(input)
        char_prediction = int_to_char[prediction]

        # Rework the input to predict the next letter
        input = np.append(input, prediction)
        input = input[1:]

        # Append prediction to the preds
        preds += char_prediction

    # Return the prediction text
    return preds


# Creates a certain number of sentances ending in periods
def generate_sentences(seed, num_sentences):
        
    # Holds predictions
    preds = ""

    # Input to the predictor
    input = seed
    
    # Number of sentences generated
    count = 0
    
    # Flag to start saving text after a period
    save = 0

    # Sets up a loop for length "length"
    while count <= num_sentences:
        # Get the prediction from the input
        prediction = predict_random_letter(input)
        char_prediction = int_to_char[prediction]

        # Rework the input to predict the next letter
        input = np.append(input, prediction)
        if len(input > 100):
            input = input[1:]

        # Append prediction to the preds
        if save == 1:
            preds += char_prediction
        
        # Increments the count if we have found a period
        if prediction == 11 or 0:
            save = 1
            count += 1

    # Return the prediction text
    return preds


def create_texts(seed, length):

    # Get seed text
    seed_text = ""
    for char in seed:
        seed_text += int_to_char[char]
    
    # Get generated text
    generated_text = generate_sentences(seed, num_sentences=length)

    # Show both
    print(f"Seed:\n{seed_text}\n")
    print(f"Generated {length} sentences:\n{generated_text}")

#%% Generate novel texts

# Create custom seed
seed_text = "merfenby of the fire"
seed_text = "sing the enchanted creature"
seed_coded = []
for c in seed_text:
    seed_coded.append(chars_to_int[c])
seed_coded = np.array(seed_coded)

# Get seed text from the training set
seed = x[np.random.choice(range(1,300000))]
# Use custom seed
#seed = seed_coded

# Specify number of sentences
num_sentences = np.random.choice(range(1,5))

# Generate the sentence
create_texts(seed, num_sentences)

