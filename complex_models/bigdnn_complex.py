import os
import glob
import pandas as pd
import numpy as np
import pickle
from lib.custom_tokeniser import custom_tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

from scipy.sparse import load_npz, save_npz
from lib.pass_fun import pass_fun
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import datetime


# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(preprocessor=pass_fun, tokenizer=pass_fun, sublinear_tf=True, max_features=2**13)

# Directory containing your Parquet files
#parquet_directory = 'shuffled_deduped.parquet/'
#parquet_directory = 'testrun.parquet/'
parquet_test_dir = 'data/test.parquet/'
parquet_val_dir = 'data/val.parquet/'
parquet_train_dir = 'data/train.parquet/'
parquet_small_train = 'small_train.parquet'


#parquet_test_dir = 'test/'+parquet_test_dir # For debugging
#parquet_val_dir = 'test/'+parquet_val_dir
#parquet_train_dir = 'test/'+parquet_train_dir


tfidf_pickle = 'data/tfidf-4096.pkl'

numpy_directory = 'data/intermediate/'
modelfile = "data/biggus_chungus"


# Fit TfidfVectorizer to a third of data
print('Trying to read pickle...')
try:
    tfidf_vectorizer = pickle.load(open(tfidf_pickle, "rb"))
    print('Tf-idf pickle read!')
except:
    print('pickle not found. Refitting vectoriser')
    df = pd.read_parquet(parquet_small_train, columns=['tokens'], engine='fastparquet')
    tfidf_vectorizer.fit(df['tokens'])
    pickle.dump(tfidf_vectorizer, open(tfidf_pickle, 'wb') )

def transform(path):
# Iterate through the Parquet files, transforming the content using the TfidfVectorizer, and saving the vectorized versions
    for parquet_file in sorted(glob.glob(os.path.join(path, '*.parquet'))):
        print("second pass: parsing", str(parquet_file))
        df = pd.read_parquet(parquet_file, engine='fastparquet')
    
        # Transform the content for the training and testing sets
        X = tfidf_vectorizer.transform(df['tokens'])
    
        # Create labels array for the training and testing sets
        y = np.array(df['class'])
    
        # Extract the original filename without the path and extension
        filename = os.path.splitext(os.path.basename(parquet_file))[0]

        # Save the training data as .npz and .npy files with the original filename
        save_npz(os.path.join(numpy_directory, f'X_{filename}.npz'), X)
        np.save(os.path.join(numpy_directory, f'y_{filename}.npy'), y)

#transform(parquet_train_dir)
#transform(parquet_val_dir)
#transform(parquet_test_dir)

##################### TRAIN ###############################
# Load the training data
train_files = sorted(glob.glob(numpy_directory+'X_train_*.npz'))
train_label_files = sorted(glob.glob(numpy_directory+'y_train_*.npy'))
try:
    model = load_model(modelfile)
    print('Loaded tf model')
except:
    print('tf model not found. Training....')

    #Determine the input dimension from the first training file
    input_dim = load_npz(train_files[0]).shape[1]

# Create a neural network model
    model = Sequential()
    model.add(Dense(1024, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

# Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])

# Train the model incrementally using the saved training set files
    batch_size = 1536
    epochs = 2

    for x_file, y_file in zip(train_files, train_label_files):
        print("training on", x_file, y_file)
        X_train_chunk = load_npz(x_file)
        y_train_chunk = np.load(y_file, allow_pickle=True).astype(int)

        # Train the model in smaller batches
        num_samples = X_train_chunk.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)

                X_batch = X_train_chunk[start_idx:end_idx].todense()
                y_batch = y_train_chunk[start_idx:end_idx]

                loss, acc = model.train_on_batch(X_batch, y_batch)
                print(f" - Batch {batch_idx + 1}/{num_batches}: loss={loss:.4f}, accuracy={acc:.4f}")
                
    model.save(modelfile)
       
        
##################### PREDICT  ###################################
batch_size = 8096*2
# Load the test set files

step='test' # test or val
test_files = sorted(glob.glob(numpy_directory+'X_'+step+'_*.npz'))
test_label_files = sorted(glob.glob(numpy_directory+'y_'+step+'_*.npy'))

y_pred = []
y_true = []
y_pred_binary = []

# Make predictions on the test data
for x_file, y_file in zip(test_files, test_label_files):
    print("predicting on", x_file, y_file)
    X_test_chunk = load_npz(x_file)
    y_test_chunk = np.load(y_file, allow_pickle=True)

    # Process the test data in smaller batches
    num_samples = X_test_chunk.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)

        X_batch = X_test_chunk[start_idx:end_idx].todense()
        y_batch = y_test_chunk[start_idx:end_idx]

        # Get the predictions for this batch
        #y_pred_chunk = model.predict(X_batch)
        
        # Since the output activation is sigmoid, we need to threshold the predictions
        #y_pred_chunk = (y_pred_chunk > 0.5).astype(int).flatten()
        y_pred_chunk = model.predict(X_batch).flatten()

        y_pred_binary_chunk = (y_pred_chunk > 0.5).astype(int)
        y_pred_binary.extend(y_pred_binary_chunk)
        
        y_pred.extend(y_pred_chunk)
        y_true.extend(y_batch)



# Calculate the accuracy
print(f'classification report: biggus chungus, on {step} data')
print('finished:', datetime.datetime.now())
print(classification_report(y_true, y_pred_binary))


# Save results for later use
y_pred = np.array(y_pred)  # Your predictions
y_true = np.array(y_true)  # True labels
np.save("data/predictions/big_dnn_y_preds.npy", y_pred)
np.save("data/predictions/big_dnn_y_true.npy", y_true)
"""
# Calculate the ROC curve and AUC score
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC (Bigdnn)')
plt.legend(loc="lower right")
#plt.show()

plt.savefig('figures/ROC_curve_bigdnn_complex.png')
"""
