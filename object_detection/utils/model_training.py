import boto3
import sagemaker
import pandas as pd
import bokeh
import bokeh.io

bokeh.io.output_notebook()
from bokeh.plotting import figure, show
from bokeh.models import HoverTool


def get_tuning_job_status(tuning_job_name):

    region = boto3.Session().region_name
    sage_client = boto3.Session().client("sagemaker")

    # run this cell to check current status of hyperparameter tuning job
    tuning_job_result = sage_client.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuning_job_name
    )

    status = tuning_job_result["HyperParameterTuningJobStatus"]
    if status != "Completed":
        print("Reminder: the tuning job has not been completed.")

    job_count = tuning_job_result["TrainingJobStatusCounters"]["Completed"]
    print("%d training jobs have completed" % job_count)

    objective = tuning_job_result["HyperParameterTuningJobConfig"]["HyperParameterTuningJobObjective"]
    is_minimize = objective["Type"] != "Maximize"
    objective_name = objective["MetricName"]

    return objective, is_minimize, objective_name


def get_tuning_results(tuning_job_name):

    tuner = sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name)

    full_df = tuner.dataframe()

    if len(full_df) > 0:
        df_tuning_results = full_df[full_df["FinalObjectiveValue"] > -float("inf")]
        if len(df_tuning_results) > 0:
            df_tuning_results = df_tuning_results.sort_values("FinalObjectiveValue", ascending=False)
            print("Number of training jobs with valid objective: %d" % len(df_tuning_results))
            print({"lowest": min(df_tuning_results["FinalObjectiveValue"]), "highest": max(df_tuning_results["FinalObjectiveValue"])})
            pd.set_option("display.max_colwidth", None)  # Don't truncate TrainingJobName
        else:
            print("No training jobs have reported valid results yet.")

    return df_tuning_results


class HoverHelper:
    def __init__(self, tuning_analytics):
        self.tuner = tuning_analytics

    def hovertool(self):
        tooltips = [
            ("FinalObjectiveValue", "@FinalObjectiveValue"),
            ("TrainingJobName", "@TrainingJobName"),
        ]
        for k in self.tuner.tuning_ranges.keys():
            tooltips.append((k, "@{%s}" % k))

        ht = HoverTool(tooltips=tooltips)
        return ht

    def tools(self, standard_tools="pan,crosshair,wheel_zoom,zoom_in,zoom_out,undo,reset"):
        return [self.hovertool(), standard_tools]


def plot_performance_over_time(tuning_job_name, df_tuning_results):

    tuner = sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name)

    hover = HoverHelper(tuner)

    p = figure(width=900, height=400, tools=hover.tools(), x_axis_type="datetime")
    p.circle(source=df_tuning_results, x="TrainingStartTime", y="FinalObjectiveValue")
    show(p)


def plot_performance_vs_hyperparameter(tuning_job_name, df_tuning_results, objective_name):

    tuner = sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name)

    hover = HoverHelper(tuner)

    ranges = tuner.tuning_ranges
    figures = []
    for hp_name, hp_range in ranges.items():
        categorical_args = {}
        if hp_range.get("Values"):
            # This is marked as categorical.  Check if all options are actually numbers.
            def is_num(x):
                try:
                    float(x)
                    return 1
                except:
                    return 0

            vals = hp_range["Values"]
            if sum([is_num(x) for x in vals]) == len(vals):
                # Bokeh has issues plotting a "categorical" range that's actually numeric, so plot as numeric
                print("Hyperparameter %s is tuned as categorical, but all values are numeric" % hp_name)
            else:
                # Set up extra options for plotting categoricals.  A bit tricky when they're actually numbers.
                categorical_args["x_range"] = vals

        # Now plot it
        p = figure(
            width=500,
            height=500,
            title="Objective vs %s" % hp_name,
            tools=hover.tools(),
            x_axis_label=hp_name,
            y_axis_label=objective_name,
            **categorical_args,
        )
        p.circle(source=df_tuning_results, x=hp_name, y="FinalObjectiveValue")
        figures.append(p)
    show(bokeh.layouts.Column(*figures))