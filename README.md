# Computer Vision Ops
A simple template for static image analysis operations in computer vision, with a list of automation tool used.  This can be a nice primer for anyone starting with in ML Ops to understanding the basics of setting up a production grade project. 

Project features: 
 - A fully functionning U-Net implementation 
 - Documented list of tools used to setup this project
 - Some tutorials for setting up tools used in this project

**This repo is in active development and still finding its purpose...**

## Setup your environment
Once you've cloned or forked this project, cd to the project root from your terminal and run the following commands. 
    
    pip install poetry
    poetry install

Running command `poetry install` will create a venv and install all dependencies using `poetry.lock` file.  

## Running training
To run training, use `poetry run` wich will ensure that the right venv is used.  

    poetry run python ./cvops/train.py

## Running tests
This project uses pytest for running tests.  To run test, run this command from the 

    poetry run pytest

# Project documentation 
Project documentation is located here : https://cvops.readthedocs.io/
