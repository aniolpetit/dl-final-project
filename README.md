## Usage 


#### 1. Clone the repositories
```bash
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI/
make
python setup.py build
python setup.py install
cd ../../
git clone https://github.com/aniolpetit/dl-final-project.git
```

#### 2. Download the dataset

```bash
pip install -r requirements.txt
chmod +x download.sh
./download.sh
```

#### 3. Preprocessing

```bash
python build_vocab.py   
python resize.py
```

#### 4. Train the model

```bash
python train.py    
```

#### 5. Test the model 

```bash
python sample.py --image='png/example.png'
```

<br>

## Dataset
If you do not want to download and preprocess the whole dataset, you can download only the validation set (this one does not require resizing) and stop the download once you already have a few images to test with. Do this by commenting all lines in the download.sh file except for lines **1, 4, 10, 11** (if you stop execution early, you would need to unzip the images manually). You can also use images of your own, in acse of doing this and not running the download.sh file make sure to place them to './data/' which you should create yourself.

## Pretrained model
If you do not want to train the model from scratch, you can use our pretrained model. You can download the pretrained model [here](https://drive.google.com/drive/folders/1REjjWf08a11S_LEZZ57S1WFsuGY3idZd?usp=drive_link) and the vocabulary file [here](https://drive.google.com/file/d/1jHObH0FFAcu3vrbLwERlM1uEzFSScNFG/view?usp=drive_link). You should extract pretrained_model.zip to `./models/` and vocab.pkl to `./data/` using `unzip` command.
