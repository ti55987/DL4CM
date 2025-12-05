import pandas as pd

def read_data(f):
  data = pd.read_csv(f)
  if 'agentid' not in data:
    data = data.rename(columns={"id": "agentid"})
  if 'trials' not in data:
    data = data.rename(columns={'trial': 'trials'})

  data['isswitch'] = data['trials'].apply(lambda x: 1 if x == 1 else 0)
  return data