# Deep Learning Lab Assignments

Complete implementations of all 10 Deep Learning lab assignments.

## Setup

```bash
pip install -r requirements.txt
```

## Labs

| # | Topic | File |
|---|-------|------|
| 1 | PCA – Dimensionality Reduction | `Lab1_PCA/lab1_pca.py` |
| 2 | Simple & Multiple Linear Regression | `Lab2_LinearRegression/lab2_linear_regression.py` |
| 3 | K-Nearest Neighbours (Iris) | `Lab3_KNN/lab3_knn.py` |
| 4 | Naive Bayes Classifier | `Lab4_NaiveBayes/lab4_naive_bayes.py` |
| 5 | Decision Tree + Random Forest + AdaBoost | `Lab5_DecisionTree_RF_AdaBoost/lab5_tree_rf_adaboost.py` |
| 6 | CNN – MNIST Digit Classification | `Lab6_CNN_MNIST/lab6_cnn_mnist.py` |
| 7 | CNN – Plant Disease Detection | `Lab7_CNN_PlantDisease/lab7_plant_disease.py` |
| 8 | CNN – Fashion MNIST Classification | `Lab8_CNN_FashionMNIST/lab8_fashion_mnist.py` |
| 9 | RNN – Stock Price Prediction | `Lab9_RNN_StockPrice/lab9_rnn_stock.py` |
| 10 | LSTM – Weather Prediction | `Lab10_LSTM_Weather/lab10_lstm_weather.py` |

## How to Run

```bash
# Example
cd Lab1_PCA
python lab1_pca.py
```

Labs 1–6, 8–10 use built-in datasets and run without any external data files.  
**Lab 7** requires a plant disease dataset folder (e.g., PlantVillage from Kaggle).

## Notes

- Labs 9 & 10 include synthetic data generators so they run immediately even without a real CSV. Swap in your actual CSV when available (instructions in each file).
- All plots are saved as `.png` in the same folder as the script.


lab 1
Viva questions (click to reveal answer)

What is PCA?
›
Principal Component Analysis. It reduces dimensions by finding new axes (principal components) that capture maximum variance in data.
Why standardize before PCA?
›
PCA is variance-based. If features have different scales (e.g., height in cm vs weight in kg), larger-scale features dominate. Standardization fixes this.
What is a scree plot?
›
A line graph of explained variance ratio vs. number of components. The 'elbow' point tells us how many components to keep.
What are principal components?
›
New orthogonal axes created by PCA. PC1 has max variance, PC2 is orthogonal to PC1 with next max variance, etc.
What is explained_variance_ratio_?
›
Fraction of total variance captured by each principal component. Sum = 1.0 for all components.
Difference between fit() and fit_transform()?
›
fit() learns parameters (mean, std). transform() applies them. fit_transform() does both in one step.


lab 2
Viva questions (click to reveal answer)

What is Simple vs Multiple Linear Regression?
›
SLR: one input feature predicts one output. MLR: multiple input features predict one output.
What does R² score mean?
›
Coefficient of determination. R²=0.85 means model explains 85% of variance in target. Closer to 1 is better.
What is the difference between MSE and RMSE?
›
MSE is in squared units. RMSE is sqrt(MSE), in same unit as target — easier to interpret as average error.
Why do we split into train and test?
›
To evaluate how well model generalizes to unseen data. Training on all data gives overly optimistic metrics.
What assumption does Linear Regression make?
›
Linear relationship between features and target, no multicollinearity, homoscedasticity (constant variance of errors), errors are normally distributed.
What is overfitting?
›
Model learns training data too well including noise, performs poorly on test data. MLR with many features risks this.


lab 3

Viva questions (click to reveal answer)

How does KNN work?
›
KNN stores training data. For a new point, it finds K nearest neighbors using distance (Euclidean by default) and assigns the majority class among those K neighbors.
What is the effect of K value?
›
Small K (e.g., 1) = overfits, sensitive to noise. Large K = smoother boundary, might underfit. Optimal K found using cross-validation.
What distance metric does KNN use?
›
Default is Euclidean distance: sqrt(sum of (xi-yi)²). Can also use Manhattan, Minkowski.
Is KNN a lazy learner?
›
Yes. It does no training computation — just stores data. All computation happens at prediction time.
Why scale features for KNN?
›
KNN uses distance. Features with large values dominate. StandardScaler ensures equal contribution from all features.
What is Iris dataset?
›
Famous dataset with 150 samples, 3 classes (Setosa, Versicolor, Virginica), 4 features (sepal length/width, petal length/width).

lab 4

What is Bayes theorem?
›
P(A|B) = P(B|A) × P(A) / P(B). In classification: P(class|features) ∝ P(features|class) × P(class).
What is the 'Naive' assumption?
›
Features are conditionally independent given the class. In reality this is rarely true, but the classifier still works well in practice.
When to use Gaussian NB vs Multinomial NB?
›
Gaussian NB: continuous features (e.g., height, weight). Multinomial NB: discrete counts (e.g., word frequencies in text).
What is precision vs recall?
›
Precision = TP/(TP+FP) — how many predicted positives are actually positive. Recall = TP/(TP+FN) — how many actual positives were found.
What is F1-score?
›
Harmonic mean of precision and recall: 2×(P×R)/(P+R). Good when class imbalance exists.
Advantage of Naive Bayes?
›
Fast, works well with small data, handles high dimensions well, good baseline for text classification.


lab 5

How does a Decision Tree split?
›
Finds the feature and threshold that maximizes information gain (or minimizes Gini impurity) at each node.
What is overfitting in decision trees?
›
A deep tree memorizes training data including noise, gives high train accuracy but low test accuracy.
What is cost complexity pruning?
›
After full tree is built, branches are removed if they don't significantly improve accuracy. ccp_alpha controls aggressiveness.
How does Random Forest reduce overfitting?
›
Each tree is trained on a random bootstrap sample of data with random feature subset. Averaging many uncorrelated trees reduces variance.
What is bagging vs boosting?
›
Bagging (Random Forest): trains trees in parallel on random subsets, averages results. Boosting (AdaBoost): trains sequentially, each model corrects errors of previous.
What is a decision stump?
›
A decision tree with depth=1. Makes one split. Used as weak learner in AdaBoost.
What is Gini impurity?
›
Measure of how often a randomly chosen element would be misclassified. 0 = pure node. Used to decide splits.


lab 6

mnist.load_data()
Downloads MNIST: 60k train + 10k test images of handwritten digits 0–9.
X_tr.reshape(-1, 28, 28, 1)
Add channel dimension. CNN expects (batch, height, width, channels). Grayscale = 1 channel.
/ 255.0
Normalize pixel values from 0-255 to 0-1. Helps neural network train faster.
Conv2D(32, (3,3), activation='relu')
32 filters of 3×3 size. Each filter detects a different pattern (edge, curve, etc.).
MaxPooling2D((2,2))
Reduces spatial size by half by taking max in 2×2 window. Reduces params, adds translation invariance.
Flatten()
Converts 2D feature maps to 1D vector to feed into Dense layers.
Dense(64, activation='relu')
Fully connected layer with 64 neurons. Learns combinations of features.
Dense(10, activation='softmax')
10 neurons for 10 digit classes. Softmax gives probability distribution summing to 1.
sparse_categorical_crossentropy
Loss for multi-class when labels are integers (not one-hot encoded).
np.argmax(m.predict(X_te), axis=1)
predict() gives probabilities. argmax picks the class with highest probability.
confusion_matrix + sns.heatmap
Shows which classes are confused with each other. Diagonal = correct predictions.

What is a convolutional layer?
›
Applies learnable filters to detect spatial patterns (edges, textures). Each filter produces a feature map.
What is MaxPooling?
›
Downsamples feature maps by taking maximum value in a window. Reduces computation and adds translation invariance.
What is ReLU activation?
›
f(x) = max(0,x). Removes negative values. Solves vanishing gradient. Most common hidden layer activation.
What is Softmax?
›
Converts raw scores to probabilities that sum to 1. Used in output layer for multi-class classification.
What is Adam optimizer?
›
Adaptive learning rate optimizer. Combines momentum and RMSProp. Default choice for most deep learning.
What does a confusion matrix show?
›
Row = actual class, Column = predicted. Diagonal = correct. Off-diagonal = misclassifications. Shows which classes get confused.
What is sparse_categorical_crossentropy vs categorical_crossentropy?
›
sparse: labels are integers (0,1,2...). categorical: labels are one-hot encoded ([0,1,0,...]).


lab 7

ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
Data augmentation: rescales pixels to 0-1, randomly rotates ±20°, flips horizontally. Creates more training variety.
flow_from_directory('plant_data', target_size=(64,64))
Reads images from folder structure (one subfolder per class). Resizes to 64×64.
class_mode='categorical'
One-hot encodes labels. For multi-class with categorical_crossentropy.
subset='training' / subset='validation'
Splits data using validation_split=0.2. 80% train, 20% validation.
n_cls = len(train_data.class_indices)
Automatically gets number of plant disease classes from folder names.
Dropout(0.5)
Randomly disables 50% of neurons during training. Prevents overfitting.
Dense(128, activation='relu')
Larger dense layer to learn complex disease patterns.
Dense(n_cls, activation='softmax')
Output layer with one neuron per disease class.
h = m.fit(...)
h (history) stores accuracy/loss per epoch. Used for plotting.
h.history['accuracy'] vs h.history['val_accuracy']
Compare train vs validation accuracy per epoch. Large gap = overfitting.


What is data augmentation?
›
Artificially expanding training data by applying transformations (rotation, flip, zoom). Reduces overfitting, especially with small datasets.
What is Dropout?
›
Regularization technique: randomly sets neuron outputs to 0 during training. Forces network to learn redundant representations.
What is overfitting and how to detect it?
›
Train accuracy >> validation accuracy. Model memorizes training data instead of generalizing.
Why use color images (3 channels) for plant disease?
›
Disease symptoms (yellowing, spots, discoloration) are color-dependent. 3 channels (RGB) preserve this info.
What datasets are used for plant disease?
›
PlantVillage dataset is most common: 50k+ images of 38 plant disease classes.
What is transfer learning? (bonus)
›
Using a pre-trained model (like VGG, ResNet) as base and fine-tuning for new task. Much better for small datasets.


lab 8

names = [...]
10 clothing category names corresponding to labels 0–9.
X_tr[..., np.newaxis]
Adds channel dimension. ... means 'all existing dims'. Same as reshape(-1,28,28,1).
/ 255.0
Normalizes pixels to 0–1 range.
BatchNormalization()
Normalizes layer outputs during training. Speeds up training, acts as regularizer.
Dropout(0.4)
40% dropout. Fashion MNIST is harder than MNIST, more regularization needed.
batch_size=64
Process 64 images at once per gradient update. Balances speed and stability.
m.evaluate(X_te, y_te)
Returns final test loss and accuracy.
m.predict(X_te[:5])
Get probability predictions for first 5 test images.
names[np.argmax(pred[i])]
Convert probability array to class name using argmax.


What is Fashion MNIST?
›
Drop-in replacement for MNIST. 70k grayscale 28×28 images of 10 clothing categories. Harder than digit MNIST.
What is Batch Normalization?
›
Normalizes the input of each layer (zero mean, unit variance) during training. Reduces internal covariate shift, allows higher learning rates.
What is batch size?
›
Number of training samples processed before updating model weights. Larger batch = faster but needs more memory and may converge to worse solution.
Why is Fashion MNIST harder than MNIST?
›
Digits have distinct shapes; clothing items share more visual similarity (e.g., shirt vs. coat vs. pullover).
What is sparse_categorical_crossentropy?
›
Cross-entropy loss for multi-class when labels are integers. If labels were one-hot vectors, use categorical_crossentropy.
What is an epoch?
›
One complete pass through the entire training dataset. Multiple epochs allow the model to refine weights.



lab 9

df['Close'].values.reshape(-1,1)
Get closing prices as column vector. reshape needed for MinMaxScaler.
MinMaxScaler()
Scale values to 0-1 range. RNNs train poorly on large raw stock prices.
def mk_seq(data, steps=60)
Create sliding window sequences. Each X is 60 past days, y is next day price.
X.append(data[i-steps:i, 0])
Take 60 consecutive values as input sequence.
SimpleRNN(64, return_sequences=True)
64 RNN units. return_sequences=True passes output at each step to next RNN layer.
SimpleRNN(32)
Second RNN layer, returns only final output (return_sequences=False by default).
Dense(1)
Output: single predicted price (regression, not classification).
loss='mse'
Mean Squared Error for regression. No activation on output layer.
sc.inverse_transform(pred)
Convert normalized predictions back to actual dollar values.

What is an RNN?
›
Recurrent Neural Network. Has internal state (memory) that persists across time steps. Processes sequences by passing hidden state from one step to the next.
Why use RNN for stock prediction?
›
Stock prices are time series — future prices depend on past prices. RNNs capture temporal dependencies.
What is the vanishing gradient problem?
›
In RNNs, gradients shrink exponentially through many time steps, making it hard to learn long-term dependencies. Solved by LSTM/GRU.
Why MinMaxScaler for RNN (not StandardScaler)?
›
RNNs are sensitive to input scale. MinMaxScaler bounds values to [0,1] which suits sigmoid/tanh activations.
What is return_sequences=True?
›
When stacking RNN layers, intermediate layers must pass output at every time step (not just last). return_sequences=True enables this.
Why 60 time steps?
›
Common heuristic for stock data (~3 trading months). Model sees 60 previous days to predict day 61.


lab 10

df['Temperature'].values.reshape(-1,1)
Extract temperature column as 2D array for scaler.
MinMaxScaler() + fit_transform
Normalize temperatures to [0,1]. Helps LSTM converge faster.
mk_seq(d, steps=30)
30-day sliding windows. Input = 30 past temperatures, output = next day.
LSTM(64, return_sequences=True)
64 LSTM units. return_sequences=True for stacking another LSTM layer below.
LSTM(32)
Second LSTM reads full sequence from above, outputs one vector at final step.
Dropout(0.2)
20% dropout after each LSTM to prevent overfitting.
Dense(1)
Predict one value (tomorrow's temperature).
loss='mse'
Regression task. MSE penalizes large prediction errors more.
sc.inverse_transform(pred)
De-normalize back to real temperatures (°C or °F).


What is LSTM and how is it better than SimpleRNN?
›
Long Short-Term Memory. Has gates (input, forget, output) that control what information to keep or discard. Solves vanishing gradient — can learn long-term dependencies.
What are LSTM gates?
›
Forget gate: what to erase from cell state. Input gate: what new info to add. Output gate: what to output. Cell state carries long-term memory.
Difference between RNN and LSTM?
›
RNN: simple hidden state, suffers vanishing gradient, short memory. LSTM: cell state + hidden state, gated mechanism, can remember across 100s of steps.
What is GRU?
›
Gated Recurrent Unit. Simplified LSTM with 2 gates instead of 3. Faster to train, similar performance.
Why is weather prediction a time series problem?
›
Temperature depends on recent history (trends, seasons). Past values predict future values.
What is the difference between regression and classification in output layer?
›
Regression: Dense(1) with no activation, loss=MSE. Classification: Dense(n_classes) with softmax, loss=cross-entropy.
