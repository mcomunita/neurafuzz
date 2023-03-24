<div align="center">
<img src="images/title.png" width="300">
</div>

<div  align="center">

[Paper](https://arxiv.org/abs/2211.00497) | [Dataset (restricted access)](https://zenodo.org/record/7766959)

</div>


## Setup

Install the requirements.
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Pre-trained Model
You can find the pretrained best model in the ```results/1-GCNTF3-fuzz__1-10-16-3-2-128__prefilt-None-bs6``` folder. 
That is the model used to export the Neutone model.


## Neutone Model
The final model to run on the neutone plugin is ```export_model/model_fuzzring_v1.0.0.nm```

To export again the neutone model you can run ```src/gcntfilm_neutone_fuzzring.py```.


## Dataset
You can the dataset here: [Dataset (restricted access)](https://zenodo.org/record/7766959). You need to update the root path in the training configuration (inside ```train.py```) before start the trainin.

## Training

If you would like to re-train the model, you can run the training script. The training results will be saved in the ```results``` folder.

```
python train.py
```

## Process Audio

To process audio using a trained model you can run the ```process_file_with_params.py``` script.


## Credits
[https://github.com/Alec-Wright/Automated-GuitarAmpModelling](https://github.com/Alec-Wright/Automated-GuitarAmpModelling)

[https://github.com/csteinmetz1/micro-tcn](https://github.com/csteinmetz1/micro-tcn)

[https://github.com/kuleshov/audio-super-res](https://github.com/kuleshov/audio-super-res)
