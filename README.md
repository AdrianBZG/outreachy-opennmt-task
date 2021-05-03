# Task for Outreachy project "Migration of natural language query translation code to OpenNMT 2.0"

This is a small Python application that makes use of **OpenNMT** to translate from a source language to a target language. The project follows an object-oriented programming approach and has been made as modular and flexible as possible. 

This application does the following tasks in order
1. Pre-processes the dataset
2. Trains a Transformer model
3. Translates a given input sentence into its corresponding target language 
4. Evaluates how well the model performs. 

To allow to query the model externally, it can be deployed as a REST server using the OpenNMT Server script.

## Getting Started
The easiest and most convenient way of starting development on this project is to use VirtualEnv.
Make sure you're using Python 3 and Pip 3 before installing VirtualEnv.

```bash
$ pip install --user virtualenv
```

Then create a new virtual environment and activate it 
```bash
$ virtualenv onmttask -p python3
> created virtual environment CPython3.6.9.final.0-64 in 7395ms
  creator CPython3Posix(dest=/home/user/outreachy-opennmt-task/onmttask, clear=False, no_vcs_ignore=False, global=False)
  seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=/home/user/.local/share/virtualenv)
    added seed packages: pip==21.0.1, setuptools==54.2.0, wheel==0.36.2
  activators BashActivator,CShellActivator,FishActivator,PowerShellActivator,PythonActivator,XonshActivator
  
$ source ./onmttask/bin/activate
```

Install the dependencies
```bash
(onmttask) 
$ pip install -r requirements.txt
```

## Datasets
The following datasets are available in this project. These datasets are from the [EMNLP 2017 conference](http://www.statmt.org/wmt17/translation-task.html#download). The dataset should be small so that it is easy to handle. 


| **Name**                          | **Tag**     | **Size**  | **Provided By** |
|-----------------------------------|-------------|-----------|-----------------|
| Toy English-German	              | `toy-ende`  | 1.6 MB    | ~               |
| Wiki Headlines	                  | `wiki`      | 9.1 MB    | CMU             |
| Rapid corpus of EU press releases	| `rapid2016` | 156 MB    | Tilde           | 


Download these by using the provided script
```bash
$ mkdir data
$ chmod +x download.sh
$ ./download.sh
```

In order to add more datasets to this project, one simply needs to add the appropriate download scipt and implement a reader for it.

```python
# Under src/pipeline/readers.py
class MyDataReader(DataReader):
  def __init__(self, datapath: str) -> None:
    super().__init__(datapath)
    # Give the dataset the folder name its in
    self.dataname = 'mydata'

  def _parse_data(self, tokenize, trainsplit, holdout) -> Dataset:
    # @todo Implement the read logic here
    pass

  def tokenize(self):
    # @todo Implement the tonkization logic here
    pass
```

Also add the path to the `datapaths` dictionary in the config module.
```python
datapaths = {
    "toy-ende": "data/toy-ende",
    "rapid2016": "data/rapid2016",
    "wiki": "data/wiki",
    # Your dataset
    "mydata": "..."
}
```

## Running from the Terminal
The terminal options allow the user to do the following
- Choose the dataset to use
- Choose the type of model to train
- Modify **any** model parameter
- Modify **any** training parameter

The script can be run with the default options provided in `config/defaults.py`
```bash
$ python __init__.py
```

However, you're free to give arguments to modify what the script does.
```bash
# Trains a LSTM model using default dataset and default parameters
$ python __init__.py model=lstm

# Trains a transformer model using default dataset and default parameters
$ python __init__.py model=transformer
```

You can also modify deep model parameters. For names and details, refer to `config/defaults.py`
```bash
# Trains a transformer model using default dataset and custom parameters
$ python __init__.py transformer-emb_size=128 transformer-decoder-heads=10
```
