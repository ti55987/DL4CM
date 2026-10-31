import pandas as pd

def read_data(f):
  data = pd.read_csv(f)
  if 'agentid' not in data:
    data = data.rename(columns={"id": "agentid"})
  if 'trials' not in data:
    data = data.rename(columns={'trial': 'trials'})

  data['isswitch'] = data['trials'].apply(lambda x: 1 if x == 1 else 0)
  return data


def get_last_correct_trial(df):
  correct_trial_map = {}
  last_correct_trial = []
  for _, row in df.iterrows():
      s = row.stimuli
      if s in correct_trial_map:
          last_correct_trial.append(correct_trial_map[s])
      else:
          last_correct_trial.append(None)

      if row.rewards == 1:
          correct_trial_map[s] = int(row.trials)

  return last_correct_trial


def get_last_stimuli_trial(df):
    # Create a copy of the dataframe with trials and stimuli
    temp_df = df[["trials", "stimuli"]].copy()
    # Group by stimuli and shift trials to get previous trial number
    temp_df["last_trial"] = temp_df.groupby("stimuli")["trials"].shift(1)
    return temp_df["last_trial"].values