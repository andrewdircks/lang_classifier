
'''
Predict and analyze a model's behavior on a custom code string. 
'''

import pickle
import argparse
import matplotlib.pyplot as plt
from settings import langs

def load_model(path):
  return pickle.load(open(path, 'rb'))

def predict(model, snippet):
    log_prob =  model.predict_log_proba([snippet])
    return log_prob[0]

if __name__ == '__main__':
  '''
  --mpath is the path to the model
  --snippet is the code snippet to be evaluated - include '\n' characters
  --out is where the probability plot should be saved
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument('--mpath', type=str)
  parser.add_argument('--snippet', type=str)
  parser.add_argument('--out', type=str)
  args = parser.parse_args()

  # load model and predict
  model = load_model(args.mpath)
  probs = predict(model, args.snippet)

  # plot and save the results
  fig, ax = plt.subplots()
  ax.bar(langs, probs)
  plt.xticks(fontsize=6)
  plt.ylabel('Log of class probability')
  plt.savefig(args.out)