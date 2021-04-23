# OpenNMT-App
Application exploring OpenNMT2.0 library
Author : Megha Jain
Git repo : https://github.com/meghajain-1711/OpenNMT-App 

## Run UI Locally

```
python app.py
```

## Retrieve Dataset by running this
```
chmod 755 retrieve_dataset.sh
./retrieve_dataset.sh

OR 

class PrepTrainingData has a module extractDataFromURL()
```

## Generation of training data and Training happens by running this
```
python TrainTransformer.py
```

## Major Goal : 
Build a language Translator using OpenNMT.py

### DataSet : 
"Wiki Headlines" http://www.statmt.org/wmt15/wiki-titles.tgz Finnish to English

### Frontend: 
- [x] UI for the user to enter a query and it returns the translation to the other language

### Rest Server deployment of API : 
- [x] REST server of OpenNMT.py 

### Backend : 
- [x] GenerateTrainingData : Class PrepTrainingData has the following functions
    1. extractDataFromURL : provide a URL and it will extract the dataset to the mentioned path
    2. preprocessTrainingData : removes serial numbers from line; separates the single parallele corpus dataset into 2 languages file
    3. generateTrainingData : uses pandas to convert text file to dataframe , calls scikit test_train_split to create train, validate, holdout datasets
    4. createTrainingYaml : creates yaml which will be used for both building vocab and training

- [x] Training the transformer Model : TrainTransformer.py, refer how to from USEMODEL above to see how to run
    1. sets training parameters
    2. creates PrepTrainingData object 
    3. Runs openNMT build vocab on the config generated above
    4. Uses the above vocabulary for training the model

- [x] Translates a given input sentence into its corresponding target language sentence : class LanguageTranslator needs a pretrained model 
    1. Takes Input sentence and Model dir as input arguments
    2. Translates the sentence based on pretrained model

- [x] Evaluates how well the model performs by having a hold-out test dataset : ModelPerformace    
    1. Hold out dataset was created during generating dataset
    2. The same dataset is used to generate predictions now 
    3. The predictions are compared with the tgt hold out dataset

An Object Oriented Programming approach has been used throughout. 


## Considerations 

1. Pandas need to be installed for train_test_split. 
2. Read csv from the text file : header should be set to None, line terminator ( different values for Linux/Windows systems , sep not comma but tab)
3. Subprocess.PIPE will hang indefinitely if stdout is more than 65000 characters. So removed the same 
4. subprocess.communicate() preferred over subprocess.wait() for the same reason
5. Train command is facing numpy version issues : https://github.com/pytorch/pytorch/issues/37377 . Fix is : remove os.environ['MKL_THREADING_LAYER'] = 'GNU' or os.environ['MKL_SERVICE_FORCE_INTEL'] = '1' and just have numpy 1.20.
6. Two approaches were evaluated for: using openNMT cmd or calling onmt modules.
7. Similarly for REST server, deployment through server.csh and server.py file or TranslationServer object was evaluated








