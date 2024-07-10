import pandas as pd

dfs = []
for i in range (1,10):
    df = pd.read_csv(f"hyperparameter_list_{i}.csv").drop(labels = ["Unnamed: 0"], axis=1)
    df["Model"] = f"CSV{i}"
    dfs.append(df)

# result
    result = pd.concat(dfs)
# selecting and reordering columns 
    results = result["mean_fit_time",
                    "mean_score_time",
                    "params",
                    "mean_test_neg_mean_absolute_percentage_error",
                    "rank_test_neg_mean_absolute_percentage_error",
                    "mean_test_r2",
                    "rank_test_r2",
                    "mean_test_neg_mean_absolute_error",
                    "rank_test_neg_mean_absolute_error",
                    "Model"
                    ]
    # sort by mean_test_r2 and save to csv 
    result.sort_values("mean_test_r2",inplace=True, ascending=False)
    result.to_csv("sorted_by_r2.csv", index = False)
    # sort by mean_test_neg_mean_absolute_percentage_error and save to csv 
    result.sort_values("mean_test_neg_mean_absolute_percentage_error", inplace=True, ascending=False)
    result.to_csv("sorted_by_MAPE.csv", index = False)
    # sort by mean_test_neg_mean_absolute_error and save to csv
    result.sort_values("mean_test_neg_mean_absolute_error, inplace = True, ascending = False")
    result.to_csv("sorted_by_MAE.csv", index = False)

df_1 = pd.read_csv("hyperparameters_list_1.csv").drop(labels=["Unnamed: 0"], axis=1)
df_2 = pd.read_csv("hyperparameters_list_2.csv").drop(labels=["Unnamed: 0"], axis=1)
df_3 = pd.read_csv("hyperparameters_list_3.csv").drop(labels=["Unnamed: 0"], axis=1)
df_4 = pd.read_csv("hyperparameters_list_4.csv").drop(labels=["Unnamed: 0"], axis=1)
df_5 = pd.read_csv("hyperparameters_list_5.csv").drop(labels=["Unnamed: 0"], axis=1)
df_6 = pd.read_csv("hyperparameters_list_6.csv").drop(labels=["Unnamed: 0"], axis=1)
df_7 = pd.read_csv("hyperparameters_list_7.csv").drop(labels=["Unnamed: 0"], axis=1)
df_8 = pd.read_csv("hyperparameters_list_8.csv").drop(labels=["Unnamed: 0"], axis=1)
df_9 = pd.read_csv("hyperparameters_list_9.csv").drop(labels=["Unnamed: 0"], axis=1)

df_1["Model"] = "CSV 1"
df_2["Model"] = "CSV 2"
df_3["Model"] = "CSV 3"
df_4["Model"] = "CSV 4"
df_5["Model"] = "CSV 5"
df_6["Model"] = "CSV 6"
df_7["Model"] = "CSV 7"
df_8["Model"] = "CSV 8"
df_9["Model"] = "CSV 9"

list_of_dataframes = [df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9]
result = pd.concat(list_of_dataframes)

result = result[
    [
        "mean_fit_time",
        "mean_score_time",
        "params",
        "mean_test_neg_mean_absolute_percentage_error",
        "rank_test_neg_mean_absolute_percentage_error",
        "mean_test_r2",
        "rank_test_r2",
        "mean_test_neg_mean_absolute_error",
        "rank_test_neg_mean_absolute_error",
        "Model",
    ]
]

result.sort_values("mean_test_r2", inplace=True, ascending=False)
result.to_csv("sorted_by_r2.csv")
result.to_excel("sorted_by_r2.xlsx")

result.sort_values(
    "mean_test_neg_mean_absolute_percentage_error", inplace=True, ascending=False
)
result.to_csv("sorted_by_MAPE.csv")
result.to_excel("sorted_by_MAPE.xlsx")

result.sort_values("mean_test_neg_mean_absolute_error", inplace=True, ascending=False)
result.to_csv("sorted_by_MAE.csv")
result.to_excel("sorted_by_MAE.xlsx")
