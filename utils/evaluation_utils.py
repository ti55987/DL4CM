import numpy as np
from sklearn.metrics import r2_score

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
