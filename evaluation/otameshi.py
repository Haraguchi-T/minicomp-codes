import wandb
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

def make_evaluation_sample_tables(path: Path, task_name: str) -> pd.DataFrame:
    # 例: parquet から dataframe を作成する簡易関数
    df = pd.read_parquet(path)
    df['dataset'] = task_name
    df = df.rename(columns={'input': 'example', 'prediction': 'predictions', 'answer': 'gold'})
    return df[['dataset', 'example', 'predictions', 'gold']]

def upload_to_wandb(run_name: str, all_results: List[Dict[str, Any]],
                     average_metrics: Dict[str, float], log_training_config: Dict[str, Any]):
    """安全に wandb に結果をアップロードし、サンプルテーブルを作る"""

    # 個人または一人チームアカウントで現在ログイン中の entity を取得
    api = wandb.Api()
    entity_name = api.viewer.username

    # プロジェクト名を固定してオンラインモードで初期化
    wandb.init(
        project="minicomp-test2",  # 新規プロジェクト名
        entity="haraguchi-chiba-university",
        name=run_name,
        config={
            "num_tasks": len(all_results),
            "tasks": [result.get("task_name", "unknown") for result in all_results]
        },
        mode="online"
    )

    # 各タスクのメトリクスをログ
    for result in all_results:
        if "metrics" in result:
            task_name_tmp = result.get("task_name", "unknown")
            metrics = result["metrics"]
            prefixed_metrics = {f"{task_name_tmp}_{k}": v for k, v in metrics.items()}
            wandb.log(prefixed_metrics)

    # 詳細ログファイルの最新を取得
    tasks = set([result.get("task_name", "unknown") for result in all_results])
    detailed_log_paths = {task: list(Path(f"./eval_results/{task}/details").glob("**/*.parquet")) for task in tasks}
    detailed_log_paths = {k: max(v, key=lambda x: x.stat().st_mtime) for k, v in detailed_log_paths.items() if v}

    # 各タスクメトリクスのテーブル作成
    each_metrics_data = []
    for result in all_results:
        if "metrics" in result:
            row = {
                "run_name": run_name,
                "task": result.get("task_name", "unknown"),
                "model": result.get("model_name", "unknown"),
                "evaluation_time": result.get("evaluation_time", "unknown")
            }
            row.update(result["metrics"])
            each_metrics_data.append(row)

    # 平均メトリクス行の作成
    average_metrics_data = []
    if average_metrics:
        avg_row = {"run_name": run_name}
        avg_row.update(average_metrics)
        average_metrics_data.append(avg_row)

    # wandb にテーブルとしてログ
    if log_training_config:
        table = wandb.Table(dataframe=pd.DataFrame([log_training_config]))
        wandb.log({"Training Config Table": table})

    if each_metrics_data:
        table = wandb.Table(dataframe=pd.DataFrame(each_metrics_data))
        wandb.log({"Detail Metrics Table": table})

    if average_metrics_data:
        table = wandb.Table(dataframe=pd.DataFrame(average_metrics_data))
        wandb.log({"Evaluation Score Table": table})

    # 評価サンプルテーブル作成
    if detailed_log_paths:
        df_list = []
        for task, path in detailed_log_paths.items():
            df = make_evaluation_sample_tables(path, task)
            df_list.append(df)
        table = wandb.Table(dataframe=pd.concat(df_list, ignore_index=True))
        wandb.log({"Evaluation Samples Table": table})

    wandb.finish()


if __name__ == "__main__":
    # テスト用ダミーデータ
    all_results = [
        {
            "task_name": "task1",
            "model_name": "dummy_model",
            "evaluation_time": "2025-08-20 13:00",
            "metrics": {"accuracy": 0.9, "loss": 0.1}
        },
        {
            "task_name": "task2",
            "model_name": "dummy_model",
            "evaluation_time": "2025-08-20 13:05",
            "metrics": {"accuracy": 0.8, "loss": 0.2}
        }
    ]

    average_metrics = {"avg_accuracy": 0.85, "avg_loss": 0.15}
    log_training_config = {"learning_rate": 0.001, "batch_size": 32}

    upload_to_wandb("test_run_001", all_results, average_metrics, log_training_config)
