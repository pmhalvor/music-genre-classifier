# Music Genre Classification 
## Resources Sheet


# Learning resources
## Introduction to audio processing w/ Python
These are good for learning the basic libraries used for audio analysis, and for getting a feel for the data.
- [Audio Processing in Python - C. MacLeod](https://realpython.com/python-scipy-fft/)
- [Understanding the Mal Spectrum - L. Roberts](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)
- [Getting to know the Mel Spectrogram - D. Gartzman](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0)
- [The Dummy's guide to MFCC - P. Nair (Medium)](https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd)


## Deep-dives and Courses

- [Audio Course - HuggingFace](https://huggingface.co/learn/audio-course/chapter0/introduction)
- [Deep Learning for Audio Classification - S. Adams](https://www.youtube.com/watch?v=Z7YM-HAz-IY&list=PLhA3b2k8R3t2Ng1WW_7MiXeh1pfQJQi_P)


# Project template
I asked ChatGPT how one should build a project around exploring different music classifiers. Here's what it said:

### Step 1: Data Preparation
1. **Data Collection:**
   - Gather a diverse dataset of audio samples representing different music genres. Websites like GTZAN, Free Music Archive, or Kaggle provide such datasets.

2. **Feature Extraction:**
   - Extract MFCC features from the audio samples. Libraries like Librosa (Python) or MIRtoolbox (MATLAB) can help with this.
Store stage output locally.

### Step 2: Data Exploration and Preprocessing
1. **Exploratory Data Analysis (EDA):**
   - Visualize the data to understand the distribution of MFCC features across genres.

2. **Data Preprocessing:**
   - Normalize the MFCC features to ensure all features contribute equally.
   - Split the data into training and testing sets.

### Step 3: Feature Selection (Optional)
1. **Dimensionality Reduction:**
   - Apply techniques like Principal Component Analysis (PCA) to reduce the dimensionality of MFCCs if the feature space is large.

### Step 4: Model Selection
1. **Traditional Machine Learning Models:**
   - **Random Forest:**
     - Random Forest classifiers work well for this task due to their ability to handle high-dimensional data.
   - **Support Vector Machine (SVM):**
     - SVM classifiers are effective for classification tasks and can handle complex relationships in the data.
   - **k-Nearest Neighbors (kNN):**
     - kNN classifiers are simple yet powerful for this scenario, especially for small to medium-sized datasets.

2. **Deep Learning Models (Optional):**
   - **Multilayer Perceptron (MLP):**
     - MLPs are suitable for this task since they can handle high-dimensional data and learn complex relationships.
   - **Convolutional Neural Networks (CNNs):**
     - CNNs are suitable for analyzing spectrogram-like data. You can convert MFCCs into images and use CNN architectures.
   - **Recurrent Neural Networks (RNNs):**
     - RNNs can capture sequential patterns in music. You might consider transforming MFCCs into sequences and use RNN architectures.

### Step 5: Model Training and Evaluation
1. **Training:**
   - Train each selected model on the training data.

2. **Evaluation:**
   - Evaluate models using metrics like accuracy, precision, recall, and F1-score.
   - Perform cross-validation to assess models' generalizability.

### Step 6: Post-Processing and Analysis
1. **Ensemble Methods:**
   - Experiment with ensemble methods like Voting Classifier or Stacking Classifier to combine predictions from multiple models.

2. **Pretrained models:**
   - Use pretrained models like VGGish or OpenL3 to extract features from audio samples and train classifiers on top of them.

3. **Error Analysis:**
   - Analyze misclassified samples to understand patterns or common mistakes made by the models.

### Step 7: Reporting and Documentation
1. **Documentation:**
   - Document the entire process, including data preprocessing, model selection, hyperparameters, and evaluation results.

2. **Report:**
   - Prepare a report summarizing findings, challenges faced, and insights gained during the analysis.


## Specific steps
Some notes on specific steps to take:

### 2.1 EDA 
This EDA step provides insights into the duration variations across genres, which can be essential for selecting appropriate model architectures and sequence lengths during the model building process.

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the GTZAN dataset (change this to your dataset path)
dataset_path = "/path/to/your/gtzan_dataset"

# Function to extract MFCCs from audio file
def extract_mfcc(file_path, n_mfcc=13, hop_length=512, n_fft=2048):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    return mfccs

# List to store extracted MFCCs and corresponding labels
mfccs_list = []
labels = []

# Loop through each genre folder in the dataset
for genre in os.listdir(dataset_path):
    genre_folder = os.path.join(dataset_path, genre)
    for file in os.listdir(genre_folder):
        file_path = os.path.join(genre_folder, file)
        mfccs = extract_mfcc(file_path)
        # Store MFCCs and genre label
        mfccs_list.append(mfccs)
        labels.append(genre)

# Convert the list of MFCCs and labels to numpy arrays
mfccs_array = np.array(mfccs_list)
labels_array = np.array(labels)

# Visualize MFCCs using a box plot for each genre
plt.figure(figsize=(12, 6))
sns.boxplot(x=labels_array, y=[mfcc.shape[1] for mfcc in mfccs_array])
plt.xlabel('Genre')
plt.ylabel('Number of Frames (Time Steps)')
plt.title('Distribution of Number of Frames for Each Genre')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

———————————————— 
Checking the distributions of MFCCs between genres is a valuable step in Exploratory Data Analysis (EDA). While individual audio samples within genres may exhibit considerable variation, aggregating the MFCCs over multiple samples within each genre can reveal meaningful patterns and differences.
To compare the distributions of MFCCs between genres, you can create violin plots or box plots for each MFCC coefficient across different genres. These visualizations allow you to observe the central tendency, spread, and shape of the MFCC distributions within each genre.


import pandas as pd

# Prepare data for visualization
mfcc_data = []
for i, mfcc in enumerate(mfccs_array):
    genre = labels_array[i]
    for j in range(mfcc.shape[0]):  # Number of MFCC coefficients
        mfcc_data.append([genre, j, np.mean(mfcc[j]), np.std(mfcc[j])])

# Create a DataFrame for visualization
df = pd.DataFrame(mfcc_data, columns=['Genre', 'MFCC_Coefficient', 'Mean', 'Standard_Deviation'])

# Create violin plots to compare MFCC distributions between genres
plt.figure(figsize=(12, 6))
sns.violinplot(x='MFCC_Coefficient', y='Mean', hue='Genre', data=df, split=True, inner='quart')
plt.xlabel('MFCC Coefficient')
plt.ylabel('Mean Value')
plt.title('MFCC Distributions Across Genres')
plt.tight_layout()
plt.show()

