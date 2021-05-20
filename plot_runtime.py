
'''
Plot runtime figures based on output of `runtime.py`
'''

import matplotlib.pyplot as plt
from ast import literal_eval

# `f_in` is an output of `runtime.py`
f_in = 'data/exp.txt'
f = open(f_in, 'r')
data = literal_eval(f.readline())
f.close()

classifiers = ['SGD', 'Perceptron', 'NB Multinomial', 'Passive-Aggressive']

def _acc(x):
  result = [0]
  acc = 0
  for t in x:
    acc += int(t)
    result.append(acc)
  return result

def plot_acc(x, y, label):
  plt.plot(_acc(x), [0] + y, label=label)

def plot(x_metric, fout):
  plt.clf()
  for alg in classifiers:
    _data = data[alg]
    plot_acc(_data[x_metric], _data['accuracies'], alg)
  plt.title('Accuracy vs. {}'.format(x_metric))

  if x_metric == 'times':
    xlabel = 'Time (seconds)'
  else:
    xlabel = 'Training Samples'
  plt.xlabel(xlabel)
  plt.title('Accuracy vs. {}'.format(xlabel))
  plt.ylabel('Test Accuracy')
  plt.legend()
  plt.savefig(fout)


# change the locations of `fout` if needed
plot('times', 'plots/time.png')
plot('n_examples', 'plots/samples.png')