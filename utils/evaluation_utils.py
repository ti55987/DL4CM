import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
import seaborn as sns

def get_model_frequency(recovered_params):
    aic_model_frequency = {'data_model': [], 'fit_model': [], 'frequency': [], 'type': []}
    for (data_model, id), df in recovered_params.groupby(['data_model', 'id']):
        # Find the best fit model with lowest AIC for this agent
        for matrix_name in ['aic', 'bic']:
            best_models = df.loc[df[matrix_name] == df[matrix_name].min()]['fit_model']
            aic_model_frequency['data_model'].extend([data_model] * len(best_models))
            aic_model_frequency['fit_model'].extend(best_models)
            aic_model_frequency['frequency'].extend([1] * len(best_models))
            aic_model_frequency['type'].extend([matrix_name] * len(best_models))

    aic_model_frequency = pd.DataFrame(aic_model_frequency).groupby(['data_model', 'fit_model', 'type']).sum().reset_index()
    return aic_model_frequency

def plot_delta_aic_bic(recovered_params, matrix_name, ax, normalize=True, vmin=None, vmax=None):
    # Lower AIC values indicate better fit, so we'll use the mean AIC for each data_model/fit_model combination
    confusion_matrix = recovered_params.pivot_table(
        values=matrix_name,
        index="data_model",  # Changed to data_model for y-axis
        columns="fit_model",  # Changed to fit_model for x-axis
        aggfunc="mean",
    )

    # We can also create a normalized version where the best model for each data type has value 1.0
    # This makes it easier to see which model fits best for each data type
    if normalize:
        normalized_confusion = confusion_matrix.copy()
        for row in normalized_confusion.index:
            min_aic = normalized_confusion.loc[row].min()
            normalized_confusion.loc[row] = normalized_confusion.loc[row] - min_aic

        confusion_matrix = normalized_confusion

    ax = sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".1f",
        cmap="viridis_r" if matrix_name.startswith("aic") else "Greys",
        linewidths=0.5,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(f"Mean ($\\Delta$ {matrix_name})")
    ax.set_xlabel("Fit Model")
    ax.set_ylabel("Data Model")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    return ax

def plot_confusion_matrix(data, matrix_name, ax):
    confusion_matrix_freq = data.pivot_table(
        values='frequency',
        index='data_model',
        columns='fit_model',
        aggfunc='sum'
    )
    # Normalize the confusion matrix by rows (convert to percentages)
    confusion_matrix_norm = confusion_matrix_freq.copy()
    for row in confusion_matrix_norm.index:
        row_sum = confusion_matrix_norm.loc[row].sum()
        if row_sum > 0:  # Avoid division by zero
            confusion_matrix_norm.loc[row] = confusion_matrix_norm.loc[row] / row_sum

    # Plot the frequency-based confusion matrix
    ax = sns.heatmap(
        confusion_matrix_norm,
        annot=True,
        fmt='.2f',  # Use integer format for frequency counts
        cmap='Blues',
        linewidths=0.5,
        cbar_kws={'label': 'Frequency'},
        ax=ax
    )
    ax.set_title(matrix_name)
    ax.set_xlabel('Selected Model (Best Fit)')
    ax.set_ylabel('True Data-Generating Model')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    return ax

def add_corr_scores(ax, y_true, y_pred):
  r2 = r2_score(y_true, y_pred)
  ax.text(
      0.1,
      0.95,
      "$R^2$ = {:.3f}".format(r2),
      horizontalalignment="left",
      verticalalignment="center",
      color='black',
      fontweight='heavy',
      transform=ax.transAxes,
      size=9,
  )

  corr = np.corrcoef(y_true, y_pred)[0, 1]
  ax.text(
      0.1,
      0.9,
      "$r$ = {:.3f}".format(corr),
      horizontalalignment="left",
      verticalalignment="center",
      color='black',
      transform=ax.transAxes,
      size=9,
  )

  return ax


def analyze_switch(subdata, num_pre_switch_trials, num_post_switch_trials):
    """Analyzes switch trials in the data and computes accuracy around switch points.

    Args:
        subdata: A pandas DataFrame containing the data with 'isswitch' and 'iscorrectaction' columns.
        num_pre_switch_trials: Number of trials to consider before each switch.
        num_post_switch_trials: Number of trials to consider after each switch.

    Returns:
        A numpy array representing the mean accuracy around switch points.
    """
    # Reset index once for each group
    subdata = subdata.reset_index(drop=True)
    grouped = subdata.groupby(['block_no'])
    valid_switch_indices = []
    for (block_no), block_data in grouped:
        switch_indices = block_data.index[block_data['isswitch'] == 1].tolist()
        first, last = block_data.index[0], block_data.index[-1]
        # Identify indices of switch trials
        # switch_indices = [index for index, is_switch in enumerate(subdata['isswitch']) if is_switch]

        # Filter switch indices to ensure enough post-switch trials
        valid_switch_indices.extend([index for index in switch_indices
                                     if (index <= last - num_post_switch_trials) & (index > first)])

    if len(valid_switch_indices) == 0:
        print("No valid switch trials found.")
        return []

    #print(valid_switch_indices)
    # Convert correctness to integer array
    correctness = subdata['correct'].astype(int).to_numpy()
    # Initialize an array to store accuracy around each switch
    all_switch_accuracy = np.full((len(valid_switch_indices), num_pre_switch_trials + num_post_switch_trials), np.nan)

    # Calculate accuracy for each valid switch
    for i, switch_index in enumerate(valid_switch_indices):
        trial_range = np.arange(switch_index - num_pre_switch_trials, switch_index + num_post_switch_trials)
        all_switch_accuracy[i] = correctness[trial_range]

    # Compute the mean accuracy across all switches
    mean_accuracy = np.nanmean(all_switch_accuracy, axis=0)

    return mean_accuracy
