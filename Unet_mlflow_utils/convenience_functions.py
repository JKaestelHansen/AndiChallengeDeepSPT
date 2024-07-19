# %%
import mlflow
import os
import pathlib
from os.path import join

mlflow.set_tracking_uri('file:'+join(os.getcwd(), join("Unet_results", "mlruns")))


def make_experiment_name_from_tags(tags: dict):
    return "".join([t + "_" + tags[t] + "__" for t in tags.keys()])


def find_experiments_by_tags(tags: dict):
    exps = mlflow.tracking.MlflowClient().list_experiments()
    def all_tags_match(e):
        for tag in tags.keys():
            if tag not in e.tags:
                return False
            if e.tags[tag] != tags[tag]:
                return False
        return True
    return [e for e in exps if all_tags_match(e)]

def WIP():
    result_list = []
    algorithm_name_list = []
    exps = find_experiments_by_tags({'DATASET': data_naming+KLD_method,
                                    'METHOD': method, 'KLD_method': KLD_method})
    assert(len(exps) == 1)
    runs = mlflow.search_runs(experiment_ids=[exps[0].experiment_id])
    results = []
    for id in runs['run_id'].to_list():
        for metric in mlflow.tracking.MlflowClient().get_metric_history(id, 'TOTAL_TRAIN_LOSS'):
            results.append(metric.value)
    print(results)
    if len(result_list) > 0:
        assert(len(results) == len(result_list[-1]))
    result_list.append(results)