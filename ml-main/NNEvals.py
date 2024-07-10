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
import tensorflow

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

model_1_MSE = tensorflow.keras.saving.load_model(r"NNModels/Model_1_MSE.keras")
model_2_MAE = tensorflow.keras.saving.load_model(r"NNModels/Model_2_MAE.keras")
model_3_MSLE = tensorflow.keras.saving.load_model(r"NNModels/Model_3_MSLE.keras")
model_4_Cosine = tensorflow.keras.saving.load_model(r"NNModels/Model_4_Cosine.keras")
model_5_Huber = tensorflow.keras.saving.load_model(r"NNModels/Model_5_huber.keras")
model_6_logcosh = tensorflow.keras.saving.load_model(r"NNModels/Model_6_logcosh.keras")
model_7_MAE_Quat = tensorflow.keras.saving.load_model(
    r"NNModels/Model_7_MAE_Quat.keras"
)
model_8_MAE_range = tensorflow.keras.saving.load_model(
    r"NNModels/Model_8_MAE_range.keras"
)
model_9_MSE_range = tensorflow.keras.saving.load_model(
    r"NNModels/Model_9_MSE_range.keras"
)
model_10_MAE_range = tensorflow.keras.saving.load_model(
    r"NNModels/Model_10_MAE_range.keras"
)
model_11_MAE_range = tensorflow.keras.saving.load_model(
    r"NNModels/Model_11_MAE_range.keras"
)
model_12_MAE = tensorflow.keras.saving.load_model(r"NNModels/Model_12_MAE.keras")
model_13_MAE_Quat = tensorflow.keras.saving.load_model(
    r"NNModels/Model_13_MAE_Quat.keras"
)
model_14_MAE_range = tensorflow.keras.saving.load_model(
    r"NNModels/Model_14_MAE_range.keras"
)


df2_csv = pd.read_csv("full_dataset_5_Arducam.csv")
df2_csv.drop(labels=["Unnamed: 0"], inplace=True, axis=1)
# Removing a pesky Unnamed column that appeared when we wrote the data to the csv file with the GenerateSpinningData.py file

# Clean up the data!
# 1. Remove the rows which don't have any LEDs visible at all, as they will contaminate the training, also remove duplicates
df2_csv = df2_csv[df2_csv["LED_1_c"] > 0]
df2_csv.drop_duplicates(inplace=True)

# Now we start processing the data
X = df2_csv.drop(labels=["q1", "q2", "q3", "q4", "range"], axis=1)
y = df2_csv[["q1", "q2", "q3", "q4", "range"]]
y_2 = df2_csv[["q1", "q2", "q3", "q4"]]
y_3 = df2_csv[["range"]]

# Doing the train_test_split:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=20
)

# Doing the train_test_split:
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X, y_2, test_size=0.20, random_state=55
)
# Doing the train_test_split:
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(
    X, y_3, test_size=0.20, random_state=59
)

# We do preprocessing using sklearn.preprocessing.StandardScaler: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
scaler_1 = StandardScaler()

scaler_1.fit(X_train)

X_train_scaled = scaler_1.transform(X_train)
X_test_scaled = scaler_1.transform(X_test)

scaler_2 = StandardScaler()
scaler_2.fit(X_train_2)

X_train_scaled_2 = scaler_2.transform(X_train_2)
X_test_scaled_2 = scaler_2.transform(X_test_2)

scaler_3 = StandardScaler()
scaler_3.fit(X_train_3)

X_train_scaled_3 = scaler_3.transform(X_train_3)
X_test_scaled_3 = scaler_3.transform(X_test_3)

# eval_model_1_MSE = model_1_MSE.evaluate(X_test_scaled, y_test)  # type: ignore
# eval_model_2_MAE = model_2_MAE.evaluate(X_test_scaled, y_test)  # type: ignore
# eval_model_3_MSLE = model_3_MSLE.evaluate(X_test_scaled, y_test)  # type: ignore
# eval_model_4_Cosine = model_4_Cosine.evaluate(X_test_scaled, y_test)  # type: ignore
# eval_model_5_huber = model_5_Huber.evaluate(X_test_scaled, y_test)  # type: ignore
# eval_model_6_logcosh = model_6_logcosh.evaluate(X_test_scaled, y_test)  # type: ignore
# eval_model_7_MAE_Quat = model_7_MAE_Quat.evaluate(X_test_scaled_2, y_test_2)  # type: ignore
# eval_model_8_MAE_range = model_8_MAE_range.evaluate(X_test_scaled_3, y_test_3)  # type: ignore
# eval_model_9_MSE_range = model_9_MSE_range.evaluate(X_test_scaled_3, y_test_3)  # type: ignore
# eval_model_10_MAE_range = model_10_MAE_range.evaluate(X_test_scaled_3, y_test_3)  # type: ignore
# eval_model_11_MAE_range = model_11_MAE_range.evaluate(X_test_scaled_3, y_test_3)  # type: ignore
# eval_model_12_MAE_range = model_12_MAE_range.evaluate(X_test_scaled_3, y_test_3)  # type: ignore

# print(f"Model 1 MSE results are (Loss, MAE, Accuracy) = {eval_model_1_MSE}")
# print(f"Model 2 MAE results are (Loss, MAE, Accuracy) = {eval_model_2_MAE}")
# print(f"Model 3 MSLE results are (Loss, MAE, Accuracy) = {eval_model_3_MSLE}")
# print(f"Model 4 Cosine results are (Loss, MAE, Accuracy) = {eval_model_4_Cosine}")
# print(f"Model 5 huber results are (Loss, MAE, Accuracy) = {eval_model_5_huber}")
# print(f"Model 6 logcosh results are (Loss, MAE, Accuracy) = {eval_model_6_logcosh}")
# print(f"Model 7 MAE_Quat results are (Loss, MAE, Accuracy) = {eval_model_7_MAE_Quat}")
# print(f"Model 8 MAE_range results are (Loss, MAE, Accuracy) = {eval_model_8_MAE_range}")
# print(f"Model 9 MSE_range results are (Loss, MAE, Accuracy) = {eval_model_9_MSE_range}")
# print(f"Model 10 MAE_range results are (Loss, MAE, Accuracy) = {eval_model_10_MAE_range}")
# print(f"Model 11 MAE_range results are (Loss, MAE, Accuracy) = {eval_model_11_MAE_range}")
# print(f"Model 12 MAE_range results are (Loss, MAE, Accuracy) = {eval_model_12_MAE_range}")


# # Predictions
# predictions = model_1_MSE.predict(X_test_scaled[:5])  # type: ignore
# print(f"Predictions={predictions}")
# print(f"Actual Values = {y_test[:5]}")

# # Predictions
# predictions = model_7_MAE_Quat.predict(X_test_scaled_2[:5])  # type: ignore
# print(f"Predictions={predictions}")
# print(f"Actual Values = {y_test_2[:5]}")

# # Predictions
# predictions = model_8_MAE_range.predict(X_test_scaled_3[:5])  # type: ignore
# print(f"Predictions={predictions}")
# print(f"Actual Values = {y_test_3[:5]}")

# # Predictions
# predictions = model_9_MSE_range.predict(X_test_scaled_3[:5])  # type: ignore
# print(f"Predictions={predictions}")
# print(f"Actual Values = {y_test_3[:5]}")

# # Predictions
# predictions = model_10_MAE_range.predict(X_test_scaled_3[:5])  # type: ignore
# print(f"Predictions={predictions}")
# print(f"Actual Values = {y_test_3[:5]}")

# # Predictions
# predictions = model_11_MAE_range.predict(X_test_scaled_3[10:15])  # type: ignore
# print(f"Predictions={predictions}")
# print(f"Actual Values = {y_test_3[10:15]}")

# Predictions
predictions = model_14_MAE_range.predict(X_test_scaled_3[10:15])  # type: ignore
print(f"Predictions={predictions}")
print(f"Actual Values = {y_test_3[10:15]}")

# Plots

history_1_df = pd.read_csv(r"NNHistories/history_model_1.csv")
history_2_df = pd.read_csv(r"NNHistories/history_model_2.csv")
history_3_df = pd.read_csv(r"NNHistories/history_model_3.csv")
history_4_df = pd.read_csv(r"NNHistories/history_model_4.csv")
history_5_df = pd.read_csv(r"NNHistories/history_model_5.csv")
history_6_df = pd.read_csv(r"NNHistories/history_model_6.csv")
history_7_df = pd.read_csv(r"NNHistories/history_model_7_quat.csv")
history_8_df = pd.read_csv(r"NNHistories/history_model_8_range.csv")
history_9_df = pd.read_csv(r"NNHistories/history_model_9_range.csv")
history_10_df = pd.read_csv(r"NNHistories/history_model_10_range.csv")
history_11_df = pd.read_csv(r"NNHistories/history_model_11_range.csv")
history_12_df = pd.read_csv(r"NNHistories/history_model_12.csv")
history_13_df = pd.read_csv(r"NNHistories/history_model_13_quat.csv")
history_14_df = pd.read_csv(r"NNHistories/history_model_14_range.csv")


def plot_histories(history_dataframe, model_name):
    loss = history_dataframe["loss"]
    val_loss = history_dataframe["val_loss"]
    epochs = epochs = range(1, len(loss) + 1)
    mae = history_dataframe["mae"]
    val_mae = history_dataframe["val_mae"]
    acc = history_dataframe["accuracy"]
    val_acc = history_dataframe["val_accuracy"]

    plt.figure(figsize=(15, 15))
    plt.suptitle(f"Model = {model_name}")

    ax_1 = plt.subplot(2, 2, 1)
    ax_1.plot(epochs, loss, "y", label="Training Loss")
    ax_1.plot(epochs, val_loss, "r", label="Validation Loss")
    ax_1.set_title("Training and Validation Loss")
    ax_1.set_xlabel("Epochs")
    ax_1.set_ylabel("Loss (MSE)")
    ax_1.legend()

    ax_2 = plt.subplot(2, 2, 2)
    ax_2.plot(epochs, mae, "y", label="Training MAE")
    ax_2.plot(epochs, val_mae, "r", label="Validation MAE")
    ax_2.set_title("Training and Validation MAE")
    ax_2.set_xlabel("Epochs")
    ax_2.set_ylabel("Mean Absolute Error (MAE)")
    ax_2.legend()

    ax_3 = plt.subplot(2, 2, 3)
    ax_3.plot(epochs, acc, "y", label="Training Accuracy")
    ax_3.plot(epochs, val_acc, "r", label="Validation Accuracy")
    ax_3.set_title("Training and Validation Accuracy")
    ax_3.set_xlabel("Epochs")
    ax_3.set_ylabel("Accuracy")
    ax_3.legend()

    plt.show()


# plot_histories(history_dataframe=history_1_df, model_name="MSE_Loss")
# plot_histories(history_dataframe=history_2_df, model_name="MAE_Loss")
# plot_histories(history_dataframe=history_3_df, model_name="MSLE_Loss")
# plot_histories(history_dataframe=history_4_df, model_name="Cosine Similarity_Loss")
# plot_histories(history_dataframe=history_5_df, model_name="Huber_Loss")
# plot_histories(history_dataframe=history_6_df, model_name="LogCosh_Loss")
# plot_histories(history_dataframe=history_7_df, model_name="MAE_Loss_Quaternions Only")
# plot_histories(history_dataframe=history_8_df, model_name="MAE_Loss_Range Only")
# plot_histories(history_dataframe=history_9_df, model_name="MSE_Loss Range Only")
# plot_histories(history_dataframe=history_10_df, model_name="MAE_Loss Range Only")
# plot_histories(history_dataframe=history_11_df, model_name="MAE_Loss Range Only")
plot_histories(history_dataframe=history_14_df, model_name="MAE_Loss Range Only")
