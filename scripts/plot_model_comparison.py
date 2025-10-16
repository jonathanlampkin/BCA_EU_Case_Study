import os
import sys
import matplotlib
matplotlib.use('Agg')  # safe in headless environments; saves files
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Ensure src is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.price_model.modeling_pipeline import ModelingPipeline


def main():
    data_path = os.path.join(PROJECT_ROOT, 'data', 'clean_data.csv')
    artifacts_dir = os.path.join(PROJECT_ROOT, 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)

    pipe = ModelingPipeline(cv_folds=3)
    X, y = pipe.load_and_prepare_data(data_path)

    # Run evaluation quickly on raw engineered features (no extra transforms to avoid leakage)
    results_df = pipe.evaluate_models_timeseries_cv(X, y)

    # Long-to-wide melt for seaborn facetting
    metric_cols = [c for c in results_df.columns if c not in ('model',)]
    long_df = results_df.melt(id_vars='model', value_vars=metric_cols, var_name='metric', value_name='score')

    # Barplot per metric
    g = sns.catplot(
        data=long_df,
        x='model', y='score', col='metric', col_wrap=3,
        kind='bar', sharey=False, height=3.2, aspect=1.2
    )
    for ax in g.axes.flatten():
        ax.tick_params(axis='x', rotation=30)
        ax.set_xlabel('')
    plt.tight_layout()
    out_path = os.path.join(artifacts_dir, 'model_metric_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')

    # Focused RMSE and R2 charts for presentations
    for focus_metric in ['rmse', 'r2']:
        if focus_metric in metric_cols:
            plt.figure(figsize=(8, 4))
            order = results_df.sort_values(focus_metric, ascending=(focus_metric!='r2'))['model']
            sns.barplot(data=results_df, x='model', y=focus_metric, order=order)
            plt.xticks(rotation=30, ha='right')
            plt.xlabel('')
            plt.ylabel(focus_metric.upper())
            plt.title(f'Model comparison: {focus_metric.upper()}')
            fp = os.path.join(artifacts_dir, f'model_{focus_metric}_comparison.png')
            plt.tight_layout()
            plt.savefig(fp, dpi=300, bbox_inches='tight')
            plt.close()

    print(f"Saved charts to {artifacts_dir}")


if __name__ == '__main__':
    main()


