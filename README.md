# Programming Language Classifier

Machine learning for detecting source code language from a string of text. This code utilizes the Kaggle dataset [Github Code Snippets](https://www.kaggle.com/simiotic/github-code-snippets), with 97,000,000 code samples, to train, test, and deploy classification models with accuracy **over 80%**.

Our system supports 20 programming languages: Bash, C, C++, CSV, DOTFILE, Go, HTML, JSON, Java, JavaScript, Jupyter, Markdown, PowerShell, Python, Ruby, Rust, Shell, TSV, Text, and Yaml.

### Auto syntax-highlight
With our fork of the [Ace](https://ace.c9.io) online text editor, run and develop code with **real time**, predictive syntax highlighting, as you type.

### Dependencies
- the Kaggle dataset
  - [full](https://www.kaggle.com/simiotic/github-code-snippets)
  - [dev](https://www.kaggle.com/simiotic/github-code-snippets-development-sample)
- Python >3.0
- [scikit-learn](https://scikit-learn.org/stable/)
- [matplotlib](https://matplotlib.org) for visuals

## Running
To build, train, and test a Naive Bayes model, use
```
python3 model.py --algorithm bayes --out models/bayes.sav
```

To perform runtime diagnostics on multiple ML classifiers, run 
```
python3 runtime.py
```
and visualize the results with 
```
python3 plot_runtime.py
```

To predict a model on a custom code snippet and visualize model output probabilities, use
```
python3 plot_prediction.py --mpath models/bayes.sav --out plots/prediction.png --snippet import numpy as np\n
```

## Documentation
See `docs\` for a research paper and presentation on this code.

## Authors
Andrew Dircks (abd93@cornell.edu) & Sam Kantor (sk2467@cornell.edu)