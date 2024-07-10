import pandas as pd
import time
from keras.models import Sequential
from keras.layers import Dense
import keras.optimizers
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
import warnings
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers  # type: ignore


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# from GenerateSpinningData import *
# The above import actually runs the generate function as well again, which is time consuming and tedious.

# t1 = time.time()
# df1_excel = pd.read_excel("full_dataset_3_Arducam.xlsx")
# t2 = time.time()
# print(f"Time taken to read excel = {t2-t1}")
# df1_excel.drop(labels=["Unnamed: 0"], inplace=True, axis=1)
# # Removing a pesky Unnamed column that appeared when we wrote the data to the excel sheet with the GenerateSpinningData.py file
# print(df1_excel.head())

# Looks like csv is better for speed of loading (~23 seconds vs 0.2 seconds!) and keeping floats as floats and not integers.

t3 = time.time()
df2_csv = pd.read_csv("full_dataset_5_Arducam.csv")
t4 = time.time()
print(f"Time taken to read csv = {t4-t3}")
df2_csv.drop(labels=["Unnamed: 0"], inplace=True, axis=1)
# Removing a pesky Unnamed column that appeared when we wrote the data to the csv file with the GenerateSpinningData.py file

print(df2_csv.shape)

print(df2_csv)
print(df2_csv.describe())

# Clean up the data!
# 1. Remove the rows which don't have any LEDs visible at all, as they will contaminate the training, also remove duplicates
t5 = time.time()
df2_csv = df2_csv[df2_csv["LED_1_c"] > 0]
df2_csv.drop_duplicates(inplace=True)
t6 = time.time()

print(f"Time taken for cleanup 1 = {t6-t5}")
print(df2_csv.shape)
print(df2_csv)

# COMPLETED AT THE GENERATE STAGE ITSELF
# 2. Remove the LEDs (or set them to -1) which go beyond the image size of 640 x 480.
# This problem arises due to incorrect FoV of the camera during generation. The previous method used arbitrary FoV.
# Now, I have found a better way to do the FoV test using the pixel size and image size. Need to implement that still, will take some time.

# Now we start processing the data
X = df2_csv.drop(labels=["q1", "q2", "q3", "q4", "range"], axis=1)
y = df2_csv[["q1", "q2", "q3", "q4", "range"]]

print(X)
print(y)

# Doing the train_test_split:
t7 = time.time()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=2
)
# We will have to try different random states, but currently using the random state to compare multiple networks with the same seed.
t8 = time.time()
print(f"Time taken for train_test_split = {t8-t7}")

# We do preprocessing using sklearn.preprocessing.StandardScaler: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
scaler_1 = StandardScaler()
scaler_2 = MaxAbsScaler()
scaler_3 = RobustScaler()
# Try other scalers as well. RobustScaler and MaxAbsScaler needs to be tried. From here: https://pieriantraining.com/complete-guide-to-feature-scaling-in-scikit-learn/#:~:text=One%20of%20the%20most%20commonly%20used%20feature%20scaling%20techniques%20in,into%20a%20machine%20learning%20algorithm.

t9 = time.time()
scaler_3.fit(X_train)

X_train_scaled = scaler_3.transform(X_train)
X_test_scaled = scaler_3.transform(X_test)
t10 = time.time()

print(f"Time taken for scaler = {t10-t9}")

print(X_train_scaled)
print(X_train_scaled.shape)

# Define the NN Model:
model_1 = Sequential()

# Hidden Layers should be in loop
model_1.add(Dense(128, activation="relu", input_dim=36))
model_1.add(Dense(64, activation="relu"))
model_1.add(Dense(32, activation="relu"))
model_1.add(Dense(16, activation="relu"))
model_1.add(Dense(8, activation="relu"))
model_1.add(Dense(4, activation="relu"))
model_1.add(Dense(2, activation="relu"))
model_1.add(Dense(36, activation="relu"))
model_1.add(Dense(28, activation="relu"))
model_1.add(Dense(98, activation="relu"))
model_1.add(Dense(76, activation="relu"))
#model_1.add(Dense(64, activation="adam"))
#model_1.add(Dense(32, activation="adam"))
#model_1.add(Dense(16, activation="adam"))
#model_1.add(Dense(8, activation="adam"))
# Need to try different activation functions: https://www.educative.io/answers/what-are-the-different-activation-functions-in-keras
# Need to try different number of layers
# Need to try different number of neurons each layer

# Output Layer
model_1.add(Dense(32, activation="linear"))

# Compile
# Use the optimizers in this way!: 
opt_4 = optimizers.legacy.Adam(learning_rate=0.001)
opt_5 = optimizers.legacy.RMSprop(learning_rate=0.001)
opt_6 = optimizers.legacy.SGD(learning_rate=0.001)
optimizer = keras.optimizers.Adam(learning_rate=0.005)
# Need to try different optimizers: https://keras.io/api/optimizers/

t11 = time.time()
model_1.compile(optimizer=opt_6, loss="huber_loss", metrics=["mae", "accuracy"])
# Need to try other losses: https://keras.io/api/losses/regression_losses/
# Need to try other metrics: https://keras.io/api/metrics/regression_metrics/
t12 = time.time()
print(f"Time taken to compile = {t12-t11}")
print(model_1.summary())

# Train and fit
t13 = time.time()
history = model_1.fit(
    X_train_scaled, y_train, validation_split=0.2, epochs=4, batch_size=64
)
# Need to try several epochs
t14 = time.time()
print(f"Time taken to train and fit = {t14-t13}")

# Plotting the metrics
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)

fig_1 = plt.figure()
plt.plot(epochs, loss, "y", label="Training Loss")
plt.plot(epochs, val_loss, "r", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()

mae = history.history["mae"]
val_mae = history.history["val_mae"]

fig_2 = plt.figure()
plt.plot(epochs, mae, "y", label="Training MAE")
plt.plot(epochs, val_mae, "r", label="Validation MAE")
plt.title("Training and Validation MAE")
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error (MAE)")
plt.legend()

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

fig_3 = plt.figure()
plt.plot(epochs, acc, "y", label="Training Accuracy")
plt.plot(epochs, val_acc, "r", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# Metrics to compare with other models
mse_model_1, mae_model_1, acc_model_1 = model_1.evaluate(X_test_scaled, y_test)
print(f"Mean Squared Error = {mse_model_1}")
print(f"Mean Absolute Error = {mae_model_1}")
print(f"Accuracy = {acc_model_1}")

# Predictions
predictions = model_1.predict(X_test_scaled[:5])  # type: ignore
print(f"Predictions={predictions}")
print(f"Actual Values = {y_test[:5]}")

model_1.save(r"NNModels/testsave.keras")
his = pd.DataFrame(history.history)
his.to_csv(r"NNHistories/testsave.csv")
