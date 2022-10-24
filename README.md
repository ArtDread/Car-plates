# License plate recognition

Detection and recognition of Russian license plates.

## Datasets

- The dataset utilized for solving detection task: [VKCV_2022_Contest_02: Carplates Recognition](https://www.kaggle.com/competitions/vkcv2022-contest-02-carplates/data).

- The dataset utilized for solving OCR task: [AUTO.RIA Numberplate OCR Datasets (Russian)](https://github.com/ria-com/nomeroff-net#autoria-numberplate-ocr-datasets).

## Pipeline

![Pipeline demonstration](/data/Pipeline.png "Pipeline demonstration")

## Project structure

Detailed working directory structure presentation.

According to pipeline, there are two independent modules: detection and ocr (o. character recognition) each containing an individual interface for models. Main module unites these modules and provides both car plate detection and recognition for images.

```text
.
├── data                    <- Files such as images for testing pipeline
├── notebooks               <- Jupyter notebooks containing training and analysis
├── src
│   ├── configs             <- Parameters and configurations providing functionality
│   ├── models              
│   │   ├── detection       <- Defining detection models
│   │   └── ocr             <- Defining ocr models
│   ├── predict
│   │   ├── main.py         <- Main logic implementation
│   │   ├── detection.py    <- Implementation of car plate detection 
│   │   └── ocr.py          <- Implementation of car plate recognition
│   ├── tools               <- Extra utilities and tools
│   └── weights             <- Weights of trained models
├── .vscode                 <- Storage of language rules for project
├── mypy.ini                <- MyPy configuration file
├── setup.cfg               <- Project configuration file
├── pipeline.ipynb          <- License plate recognition pipeline
├── requirements.txt        <- The requirements file for making prediction along the pipeline  
├── README.md              
└── LICENSE
```

## Results

### Metrics

Metrics evaluation was performed using OCR test dataset.

<dl>
  <dt><strong>Accuracy</strong></dt>
  <dd>The fraction of truly recognized sequences.</dd>
  <dt><strong>CER</strong></dt>
  <dd>Сharacter error rate averaged over all sequences.</dd>
</dl>

| Model         | Accuracy, % | CER, %     |
| ------------- | ----------- | ---------- |
| CRNN          | 98,59       | 0.22       |
| Processing... |             |

### Performance

Processing...

## Installation and dependencies

### Dependencies

Dependencies are managed with `requirements.txt`.

```python
pip install -r requirements.txt 
```

### Models weights setting

#### For Linux users

1. After cloning the repo, open the terminal in `src/weights` directory

2. Give the execution rights to the script:

    ```bash
    sudo chmod +x get_weights.sh
    ```

3. Run the script:

    ```bash
    sh get_weights.sh
    ```

After the procedure your weights' directory structure must look like this:

```text
.
├── src
│   ├── ...
│   ├── weights
│   │   ├── detection
│   │   │   └── some weights files
│   │   ├── ocr
│   │   │   └── some weights files
│   │   └── get_weights.sh
```

If something goes wrong you could use an alternative way. Check the next section for details.

#### For Windows users

Follow the link to gdrive [weights](https://drive.google.com/drive/folders/1PNfxOkWIcPW4BmeNDNNesDb35mKrg70P?usp=sharing), download and unpack the content so the weights' directory looks exactly as it shown above.

## Running

Now, when all is settled, you could test the pipeline all by yourself running the `pipeline.ipynb` following the example.

**P.S.** The Install & Run was successfully tested using pyenv virtual environment builded upon python 3.10.5.

## TODO

### General

- Rate performance (time costs) especially after adding new fast detection models
- Implement detection by video (again after adding new detection models)
- Code refactoring: missing docstrings and static typing problems

### Detection Task

- Train and integrate YOLO models in the first place as they fast

### OCR Task

- Add transform logic (rectification) in order to rate change in recognition quality
- Add more models: as minimum, try to integrate existing frameworks such as OCR Tesseract or use models with another architecture comparing to CRNN
