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
import numpy as np


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


df2_csv = pd.read_csv("full_dataset_5_Arducam.csv")
df2_csv.drop(labels=["Unnamed: 0"], inplace=True, axis=1)
# Removing a pesky Unnamed column that appeared when we wrote the data to the csv file with the GenerateSpinningData.py file

print(df2_csv.shape)

print(df2_csv)
print(df2_csv.describe())

# Clean up the data!
# 1. Remove the rows which don't have any LEDs visible at all, as they will contaminate the training, also remove duplicates
df2_csv = df2_csv[df2_csv["LED_1_c"] > 0]
df2_csv.drop_duplicates(inplace=True)

print(df2_csv.shape)
print(df2_csv)

# Now we start processing the data
X = df2_csv.drop(labels=["q1", "q2", "q3", "q4", "range"], axis=1)
y = df2_csv[["q1", "q2", "q3", "q4", "range"]]
y_2 = df2_csv[["q1", "q2", "q3", "q4"]]
y_3 = df2_csv[["range"]]

print(X)
print(y)

# Doing the train_test_split:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=2
)

# Doing the train_test_split:
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X, y_2, test_size=0.20, random_state=2
)

# Doing the train_test_split:
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(
    X, y_3, test_size=0.20, random_state=2
)

# We do preprocessing using sklearn.preprocessing.StandardScaler: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
scaler_1 = StandardScaler()
scaler_1.fit(X_train)

print(np.shape(X_train))

X_train_scaled = scaler_1.transform(X_train)
X_test_scaled = scaler_1.transform(X_test)

print(X_train_scaled)
print(X_train_scaled.shape)

scaler_2 = StandardScaler()
scaler_2.fit(X_train_2)

X_train_scaled_2 = scaler_2.transform(X_train_2)
X_test_scaled_2 = scaler_2.transform(X_test_2)

scaler_3 = StandardScaler()
scaler_3.fit(X_train_3)

X_train_scaled_3 = scaler_3.transform(X_train_3)
X_test_scaled_3 = scaler_3.transform(X_test_3)

# # Define the NN Model:
# # Model 1: MSE
# model_1 = Sequential()

# # Hidden Layers
# model_1.add(Dense(128, activation="relu", input_dim=36))
# model_1.add(Dense(64, activation="relu"))
# model_1.add(Dense(32, activation="relu"))

# # Output Layer
# model_1.add(Dense(5, activation="linear"))

# Compile
# Use the optimizers in this way!: https://stackoverflow.com/a/75556661
opt_4 = optimizers.legacy.Adam(learning_rate=0.001)
opt_5 = optimizers.legacy.RMSprop(learning_rate=0.001)
opt_6 = optimizers.legacy.SGD(learning_rate=0.001)

# model_1.compile(optimizer=opt_6, loss="mean_squared_error", metrics=["mae", "accuracy"])
# # Need to try other losses: https://keras.io/api/losses/regression_losses/

# # Train and fit
# history_1 = model_1.fit(
#     X_train_scaled, y_train, validation_split=0.2, epochs=2500, batch_size=64
# )

# model_1.save(r"NNModels/Model_1_MSE.keras")
# history_1_df = pd.DataFrame(history_1.history)
# history_1_df.to_csv(r"NNHistories/history_model_1.csv")


# Few more models
# Model 2: MAE Saving it as 12 to avoid overwriting
model_2 = Sequential()
model_2.add(Dense(128, activation="relu", input_dim=36))
model_2.add(Dense(64, activation="relu"))
model_2.add(Dense(32, activation="relu"))
model_2.add(Dense(5, activation="linear"))

model_2.compile(
    optimizer=opt_5, loss="mean_absolute_error", metrics=["mae", "accuracy"]
)

history_2 = model_2.fit(
    X_train_scaled, y_train, validation_split=0.2, epochs=350, batch_size=64
)

model_2.save(r"NNModels/Model_12_MAE.keras")
history_2_df = pd.DataFrame(history_2.history)
history_2_df.to_csv(r"NNHistories/history_model_12.csv")


# # Model 3: MSLE
# model_3 = Sequential()
# model_3.add(Dense(128, activation="relu", input_dim=36))
# model_3.add(Dense(64, activation="relu"))
# model_3.add(Dense(32, activation="relu"))
# model_3.add(Dense(5, activation="linear"))

# model_3.compile(
#     optimizer=opt_6, loss="mean_squared_logarithmic_error", metrics=["mae", "accuracy"]
# )

# history_3 = model_3.fit(
#     X_train_scaled, y_train, validation_split=0.2, epochs=2500, batch_size=64
# )

# model_3.save(r"NNModels/Model_3_MSLE.keras")
# history_3_df = pd.DataFrame(history_3.history)
# history_3_df.to_csv(r"NNHistories/history_model_3.csv")


# # Model 4: Cosine Similarity
# model_4 = Sequential()
# model_4.add(Dense(128, activation="relu", input_dim=36))
# model_4.add(Dense(64, activation="relu"))
# model_4.add(Dense(32, activation="relu"))
# model_4.add(Dense(5, activation="linear"))

# model_4.compile(optimizer=opt_6, loss="cosine_similarity", metrics=["mae", "accuracy"])

# history_4 = model_4.fit(
#     X_train_scaled, y_train, validation_split=0.2, epochs=2500, batch_size=64
# )

# model_4.save(r"NNModels/Model_4_Cosine.keras")
# history_4_df = pd.DataFrame(history_4.history)
# history_4_df.to_csv(r"NNHistories/history_model_4.csv")


# # Model 5: Huber
# model_5 = Sequential()
# model_5.add(Dense(128, activation="relu", input_dim=36))
# model_5.add(Dense(64, activation="relu"))
# model_5.add(Dense(32, activation="relu"))
# model_5.add(Dense(5, activation="linear"))

# model_5.compile(optimizer=opt_6, loss="huber_loss", metrics=["mae", "accuracy"])

# history_5 = model_5.fit(
#     X_train_scaled, y_train, validation_split=0.2, epochs=2500, batch_size=64
# )

# model_5.save(r"NNModels/Model_5_huber.keras")
# history_5_df = pd.DataFrame(history_5.history)
# history_5_df.to_csv(r"NNHistories/history_model_5.csv")


# # Model 6: logcosh
# model_6 = Sequential()
# model_6.add(Dense(128, activation="relu", input_dim=36))
# model_6.add(Dense(64, activation="relu"))
# model_6.add(Dense(32, activation="relu"))
# model_6.add(Dense(5, activation="linear"))

# model_6.compile(optimizer=opt_6, loss="log_cosh", metrics=["mae", "accuracy"])

# history_6 = model_6.fit(
#     X_train_scaled, y_train, validation_split=0.2, epochs=2500, batch_size=64
# )

# model_6.save(r"NNModels/Model_6_logcosh.keras")
# history_6_df = pd.DataFrame(history_6.history)
# history_6_df.to_csv(r"NNHistories/history_model_6.csv")

# Model 7: MAE but only for Quaternions Saving at 13 to avoid overwrite
model_7 = Sequential()
model_7.add(Dense(128, activation="relu", input_dim=36))
model_7.add(Dense(64, activation="relu"))
model_7.add(Dense(32, activation="relu"))
model_7.add(Dense(4, activation="linear"))

model_7.compile(
    optimizer=opt_5, loss="mean_absolute_error", metrics=["mae", "accuracy"]
)

history_7 = model_7.fit(
    X_train_scaled_2, y_train_2, validation_split=0.2, epochs=350, batch_size=64
)

model_7.save(r"NNModels/Model_13_MAE_Quat.keras")
history_7_df = pd.DataFrame(history_7.history)
history_7_df.to_csv(r"NNHistories/history_model_13_quat.csv")

# Model 8: Trying out some other models and saving them with new names.
# It seems like splitting the outputs into two (One for quaternions and one for range) is causing some issue.
# The validation accuracy does not go above 0.0156 for some reason.
model_8 = Sequential()
model_8.add(Dense(128, activation="relu", input_dim=36))
model_8.add(Dense(64, activation="relu"))
model_8.add(Dense(1, activation="linear"))

model_8.compile(
    optimizer=opt_5, loss="mean_absolute_error", metrics=["mae", "accuracy"]
)

history_8 = model_8.fit(
    X_train_3, y_train_3, validation_split=0.20, epochs=350, batch_size=64
)

model_8.save(r"NNModels/Model_14_MAE_range.keras")
history_8_df = pd.DataFrame(history_8.history)
history_8_df.to_csv(r"NNHistories/history_model_14_range.csv")
