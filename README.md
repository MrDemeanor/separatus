<br><br>

<h1 align="center">Separatus</h1>

<p align="center">
  <a href="/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"/></a>
  <a href="https://docs.python.org/3/index.html"><img src="https://img.shields.io/badge/python-3.6-blue.svg"/></a>
  <a href="https://www.python.org/dev/peps/pep-0008"><img src="https://img.shields.io/badge/code%20style-PEP8-brightgreen.svg"/></a>
  <a href="https://travis-ci.org/mingrammer/pyreportcard"><img src="https://travis-ci.org/mingrammer/pyreportcard.svg?branch=master"/></a>
</p>

<p align="center">
  Automatically generate segmentation annotations for objects of interest
</p>

<br><br><br>

It is relatively quick and easy to draw bounding boxes around objects of interest. However, manually annotating that object's segmentation mask is a lot more time consuming. We propose a novel approach to pipeline the process of automatically generating segmentation masks and adding those segmentations back into the annotations dataset using histogram inference. 

## Getting Started
Once in this repo, be sure to have virtualenv installed in your pip directory
```bash
pip install virtualenv
```

Create a python virtual environment and ensure that you are using at least Python v3.5 or higher
```bash
virtualenv -p python3 venv
```

Activate the virtual environment
```bash
source venv/bin/activate
```

Install dependencies
```bash
source requirements.txt
```

For each dataset that you add in the datasets folder, you must have the following structure, being sure to name your annotation files *annotations.json* and your images folder *images*. __Note__ that the subfolders (called subsets in *maskify_dataset.py*) can be named anything. However it is imperative that the subfolders contain the annotations file and images folder. 
```
datasets
|-- ipatch
|   |-- train
|   |   |-- annotations.json
|   |   `-- images
|   |-- validation
|   |   |-- annotations.json
|   |   `-- images
|   `-- test
|       |--annotations.json
|       `--images
`-- coco
    |-- train
    |   |-- annotations.json
    |   `-- images
    `-- validation
        |-- annotations.json
        `-- images
```
Now you can edit *maskify_config.json*. Set *dataset* equal to the name of your datasets folder, and a threshold value. For iPATCH, we found that a threshold of 0.48 was sufficient. 

```json
{
    "dataset": "IPATCH", 
    "threshold": 4.8E-1, 
    "subsets": [
        "train", 
        "valdiation", 
        "test"
    ], 
    "positive_histogram": "", 
    "negative_histogram": ""
}
```

Now you can process your dataset. 
```bash
python maskify_dataset.py
```

The script will generate the new annotation files and place them in your dataset's train and validation folders. 
```
datasets
|-- ipatch
    |-- train
    |   |-- annotations.json
    |   |-- annotations_maskified.json
    |   `-- images
    |-- validation
    |   |-- annotations.json
    |   |-- annotations_maskified.json
    |   `-- images
    `-- test
        |--annotations.json
        |--annotations_maskified.json
        `--images
```

## Demo.ipynb
To get a demonstration on how each image is being segmented, launch jupyter notebook and nativate to demo.ipynb. 

```bash
jupyter notebook
```
