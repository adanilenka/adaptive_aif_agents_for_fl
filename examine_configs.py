from variable_categories import batch_size_mapping, lr_mapping
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go

timestamp = "2024_08_20_16_38"
experiment_name = ""
total_df = pd.DataFrame([])
for batch_size in [8, 32, 64, 256, 512]:
    for lr in list(lr_mapping.keys()):
        mode = f"{timestamp}_{batch_size}_{lr}"
        stat_df = pd.read_csv(
            f"./logs/{mode}_{experiment_name}/centralized_logs.csv", sep="|"
        )
        stat_df = stat_df[stat_df["ROUND"] > 10]
        stat_df = stat_df[stat_df["ROUND"] < 51]
        accumulated_df = stat_df.copy()
        accumulated_df["MODE"] = f"{batch_size}_{lr}"
        accumulated_df["ACCURACY_SMOOTHED"] = accumulated_df.groupby(["CLIENT"])[
            "VAL_PERFORMANCE"
        ].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        accumulated_df["IS_PERFORMANCE"] = accumulated_df.groupby(["CLIENT"])[
            "PERFORMANCE_SLO"
        ].cumsum()
        accumulated_df["IS_TIME"] = accumulated_df.groupby(["MODE", "CLIENT"])[
            "TIME_SLO"
        ].cumsum()
        accumulated_df["I"] = accumulated_df.groupby(["MODE", "CLIENT"]).cumcount() + 1
        accumulated_df["PERFORMANCE_FULFILLMENT"] = (
            accumulated_df["IS_PERFORMANCE"] / accumulated_df["I"]
        )
        accumulated_df["TIME_FULFILLMENT"] = (
            accumulated_df["IS_TIME"] / accumulated_df["I"]
        )
        total_df = pd.concat([total_df, accumulated_df])

total_df = total_df[total_df["ROUND"] == max(total_df["ROUND"])]

total_df_sum = total_df.groupby(["MODE"])[
    ["TIME_FULFILLMENT", "PERFORMANCE_FULFILLMENT"]
].mean()
total_df_sum = total_df_sum.reset_index()

fig = px.bar(
    total_df_sum,
    x="MODE",
    y="PERFORMANCE_FULFILLMENT",
    title=f"Performance SLO fulfillment for each configuration, mean over clients",
)
fig.show()

fig = px.bar(
    total_df_sum,
    x="MODE",
    y="TIME_FULFILLMENT",
    title=f"Time SLO fulfillment for each configuration, mean over clients",
)
fig.show()

total_df_sum["MEAN_FULFILLMENT"] = (
    total_df_sum["PERFORMANCE_FULFILLMENT"] + total_df_sum["TIME_FULFILLMENT"]
) / 2
fig = px.bar(
    total_df_sum,
    x="MODE",
    y="MEAN_FULFILLMENT",
    title=f"SLO fulfillment for each configuration, mean over clients",
)
fig.show()
