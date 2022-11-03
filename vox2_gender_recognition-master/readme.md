# Gender recognition with LibriSpeech and Voxceleb

This repository implements gender recognition networks trained and evaluated on LibriSpeech and Voxceleb. 

## Setting up the project dependencies

We expect that [poetry](https://python-poetry.org/docs/) and [pyenv](https://github.com/pyenv/pyenv) are installed on the system. 

The python dependencies can be installed (in a project-specific virtual environment) by:

```bash
#! /usr/bin/env bash
set -e

# first remove if there's already a venv
echo "CREATION OF NEW CLEAN VIRTUAL ENVIRONMENT"
if [[ $(poetry env info -p) ]]; then
  echo "removing stale $(poetry env info -p)"
  rm -rf "$(poetry env info -p)"
else
  echo 'no existing virtual environment to delete'
fi

# install dependencies
PYTHON_VERSION=$(pyenv version-name)
echo "setting python to $PYTHON_VERSION"
poetry env use $PYTHON_VERSION
echo "UPDATING PIP TO LATEST VERSION"
poetry run pip install --upgrade pip
echo "INSTALLING DEPENDENCIES"
poetry update
echo "INSTALLING LOCAL PACKAGE"
poetry install
echo "INSTALLING CUDA DEPENDENCIES"
poetry run pip install \
  torch==1.12.1+cu113 \
  torchvision==0.13.1+cu113 \
  torchaudio==0.12.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

To access the virtual environment, activate it with `poetry shell` or prepend `poetry run` to the command you want to run in the virtual environment.

## Data preparation

See [data_utility](https://github.com/Loes5307/VocalAdversary2022/tree/main/data_utility-main).

## Setting up the environment

Copy the example environment variables:

```bash
$ cp .env.example .env 
```

You can than fill in `.env` accordingly. 

## Experiments

### X-vector

with voxceleb2 as train data:

```
python run.py +experiment=xvector data/module=voxceleb2
```

with the full 960h of librispeech as train data

```
python run.py +experiment=xvector data/module=librispeech_960h
```

with the clean-100h subset of librispeech as train data

```
python run.py +experiment=xvector data/module=librispeech_100h
```

### wavLM

with voxceleb2 as train data:

```
python run.py +experiment=wavlm data/module=voxceleb2
```

with the full 960h of librispeech as train data

```
python run.py +experiment=wavlm data/module=librispeech_960h
```

with the clean-100h subset of librispeech as train data

```
python run.py +experiment=wavlm data/module=librispeech_100h
```

### M5

with voxceleb2 as train data:

```
python run.py +experiment=m5 data/module=voxceleb2
```

with the full 960h of librispeech as train data

```
python run.py +experiment=m5 data/module=librispeech_960h
```

with the clean-100h subset of librispeech as train data

```
python run.py +experiment=m5 data/module=librispeech_100h
```
