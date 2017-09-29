1. Download the preprocessed_db.gz file from Drive.
2. You should have this directory under the directory where that file is. For example,
```
cnn-stuff/ckp-cnn/generate_splits.py
cnn-stuff is the current directory, where preprocessed_db.gz is.
ckp-cnn is the checked-out repository
```
This is so that you don't have git keeping track of any large data files.
3. Run `python ckp-cnn/generate_splits.py` to create View 1.
4. Run `python ckp-cnn/train_model.py` to train the model.