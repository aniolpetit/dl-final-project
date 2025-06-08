## Usage 

In order to complete all the process as we did, following all steps, follow the guide below. At the end there's some information in case there are some steps you want to skip for simplicity.

#### 1. Clone the repositories
```bash
git clone https://github.com/pdollar/coco.git
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

You can test the model using our interface:
```bash
python app.py
```
Just upload the image in the required box and click Submit

<br>

## Dataset
If you do not want to download and preprocess the whole dataset (steps 2 and 3), you can download only the validation set (this one does not require resizing).It is still heavy to download, so just leave it running for a few seconds and stop execution using Ctrl+C and you'll already have some images to test. To do this, comment all lines in the download.sh file except for lines **1, 4, 10, 11** (if you stop execution early, you would need to unzip the images folder manually). You can also use images of your own, in case of doing this and not running the download.sh file make sure to place them to './data/' which you should create yourself.

## Pretrained model
If you do not want to train the model from scratch, you can use our pretrained model. You can download the pretrained model [here](https://drive.google.com/drive/folders/1REjjWf08a11S_LEZZ57S1WFsuGY3idZd?usp=drive_link) and the vocabulary file [here](https://drive.google.com/file/d/1jHObH0FFAcu3vrbLwERlM1uEzFSScNFG/view?usp=drive_link). You should extract models.zip to a `./models/` folder and vocab_nostre.pkl to `./data/` for everything to work correctly. If you opt for this, don't create the vocabulary yourself, use the one that's provided.
