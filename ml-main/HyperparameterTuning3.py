import os
import pandas as pd
import time
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from skopt.space import Real, Categorical, Integer 
from tensorflow.keras import optimizers # type: ignore
from sklearn.linear_model import LinearRegression
from sklearn.tree import plot_tree
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


'''def read_all_csv(directory):
    """
    Reads all CSV files in a directory and concatenates them into a single DataFrame.
    """
    all_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    df_list = []
    for file in all_files:
        df = pd.read_csv(os.path.join(directory, file))
        df_list.append(df)
    concatenated_df = pd.concat(df_list, ignore_index=True)
    return concatenated_df

'''
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    X, y, test_size=0.20, random_state=42)
# We will have to try different random states, but currently using the random state to compare multiple networks with the same seed.
# adding a loop
t8 = time.time()
print(f"Time taken for train_test_split = {t8-t7}")

# We do preprocessing using sklearn.preprocessing.StandardScaler: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
scaler = StandardScaler()
# Try other scalers as well. RobustScaler and MaxAbsScaler needs to be tried. From here: https://pieriantraining.com/complete-guide-to-feature-scaling-in-scikit-learn/#:~:text=One%20of%20the%20most%20commonly%20used%20feature%20scaling%20techniques%20in,into%20a%20machine%20learning%20algorithm.

t9 = time.time()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
t10 = time.time()

print(f"Time taken for scaler = {t10-t9}")

print(X_train_scaled)
print(X_train_scaled.shape)

#######################################################################################################################################
# All of the above was from the NNTrainTest.py file, which is common protocol for most NNs, from now, we have the tuning part start.


# https://www.youtube.com/watch?v=Bc2dWI3vnE0 After the 9 min mark he goes over the below function. The enumerate stuff is pretty good.
# Also this video. The below constructor function is an amalgamation of these two videos. https://www.youtube.com/watch?v=lV0weESA0Sc

# Use the optimizers in this way!: https://stackoverflow.com/a/75556661
opt_1 = optimizers.legacy.Adam(learning_rate=0.005)
opt_2 = optimizers.legacy.RMSprop(learning_rate=0.005)
opt_3 = optimizers.legacy.SGD(learning_rate=0.005)
opt_4 = optimizers.legacy.Adam(learning_rate=0.001)
opt_5 = optimizers.legacy.RMSprop(learning_rate=0.001)
opt_6 = optimizers.legacy.SGD(learning_rate=0.001)
opt_7 = optimizers.legacy.Adam(learning_rate=0.01)
opt_8 = optimizers.legacy.RMSprop(learning_rate=0.01)
opt_9 = optimizers.legacy.SGD(learning_rate=0.01)


def create_model_1():
    model = Sequential()
    model.add(Dense(128, input_dim=36, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))

    model.add(Dense(5, activation="linear"))

    model.compile(
        optimizer=opt_1, loss="mean_squared_error", metrics=["accuracy", "mae"]
    )

    return model


def create_model_2():
    model = Sequential()
    model.add(Dense(128, input_dim=36, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))

    model.add(Dense(5, activation="linear"))

    model.compile(
        optimizer=opt_1, loss="mean_squared_error", metrics=["accuracy", "mae"]
    )

    return model


def create_model_3():
    model = Sequential()
    model.add(Dense(64, input_dim=36, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))

    model.add(Dense(5, activation="linear"))

    model.compile(
        optimizer=opt_1, loss="mean_squared_error", metrics=["accuracy", "mae"]
    )

    return model


def create_model_4():
    model = Sequential()
    model.add(Dense(32, input_dim=36, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(5, activation="linear"))

    model.compile(
        optimizer=opt_1, loss="mean_squared_error", metrics=["accuracy", "mae"]
    )

    return model


def create_model_5():
    model = Sequential()
    model.add(Dense(128, input_dim=36, activation="relu"))
    model.add(Dense(64, activation="relu"))

    model.add(Dense(5, activation="linear"))

    model.compile(
        optimizer=opt_1, loss="mean_squared_error", metrics=["accuracy", "mae"]
    )

    return model


def create_model_6():
    model = Sequential()
    model.add(Dense(128, input_dim=36, activation="relu"))
    model.add(Dense(128, activation="relu"))

    model.add(Dense(5, activation="linear"))

    model.compile(
        optimizer=opt_1, loss="mean_squared_error", metrics=["accuracy", "mae"]
    )

    return model


def create_model_7():
    model = Sequential()
    model.add(Dense(64, input_dim=36, activation="relu"))
    model.add(Dense(64, activation="relu"))

    model.add(Dense(5, activation="linear"))

    model.compile(
        optimizer=opt_1, loss="mean_squared_error", metrics=["accuracy", "mae"]
    )

    return model


def create_model_8():
    model = Sequential()
    model.add(Dense(16, input_dim=36, activation="relu"))
    model.add(Dense(16, activation="relu"))

    model.add(Dense(5, activation="linear"))

    model.compile(
        optimizer=opt_1, loss="mean_squared_error", metrics=["accuracy", "mae"]
    )

    return model


def create_model_9():
    model = Sequential()
    model.add(Dense(128, input_dim=36, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))

    model.add(Dense(5, activation="linear"))

    model.compile(
        optimizer=opt_1, loss="mean_squared_error", metrics=["accuracy", "mae"]
    )

    return model


param_grid = {
    "batch_size": [16, 32, 64],
    "optimizer": [opt_1, opt_2, opt_3, opt_4, opt_5, opt_6, opt_7, opt_8, opt_9],
    "loss": [
        "mean_squared_error",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
    ],
}

model_1 = KerasRegressor(build_fn=create_model_1, verbose=1, epochs=60)
model_2 = KerasRegressor(build_fn=create_model_2, verbose=1, epochs=60)
model_3 = KerasRegressor(build_fn=create_model_3, verbose=1, epochs=60)
model_4 = KerasRegressor(build_fn=create_model_4, verbose=1, epochs=60)
model_5 = KerasRegressor(build_fn=create_model_5, verbose=1, epochs=60)
model_6 = KerasRegressor(build_fn=create_model_6, verbose=1, epochs=60)
model_7 = KerasRegressor(build_fn=create_model_7, verbose=1, epochs=60)
model_8 = KerasRegressor(build_fn=create_model_8, verbose=1, epochs=60)
model_9 = KerasRegressor(build_fn=create_model_9, verbose=1, epochs=60)



def read_all_csv(directory):
    all_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    df_list = [pd.read_csv(os.path.join(directory, file)) for file in all_files]
    return pd.concat(df_list, ignore_index=True)

def create_model(optimizer='adam', neurons=56, layers=6, activation='relu', input_shape=36):
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_shape=(input_shape,)))
    for _ in range(1, layers):
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(5, activation='linear'))  # Assuming 5 output variables
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

param_dist = {
    'model__optimizer':  [opt_1, opt_2, opt_3, opt_4, opt_5, opt_6, opt_7, opt_8, opt_9],
    'model__neurons': [(750, 500, 250,750, 750, 500,500, 500, 500,750, 500, 500, 250,750, 750, 250, 250,500, 500, 250, 250)],
    'model__layers': [1, 2, 3],
    'model__activation': ['relu', 'tanh'],
    'batch_size': [16,32,64],
    'epochs': [50,100,150]  
}

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=1)

model = KerasRegressor(model=create_model, verbose=1, epochs=80)  
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=660, n_jobs=-1, cv=3, random_state=42)  # Reduced n_iter for example

random_search_result = random_search.fit(X_train, y_train)

#save to csv 
result_df = pd.DataFrame(random_search_result.cv_results_)
result_df.to_csv('random_search_results3.csv',index=False)


# Results summary
print("Best: %f using %s" % (random_search_result.best_score_, random_search_result.best_params_))
results = random_search_result.cv_results_
for mean, stdev, param in zip(results['mean_test_score'], results['std_test_score'], results['params']):
    print("%f (%f) with: %r" % (mean, stdev, param))