# SSD-MOBILENET-V2-EXAMPLE

## GENERAL INFORMATION
This is a SSD MOBILENET V2 network designed for object_detection. In this repo, there is an example
model made to detect a pencil sharpener.

*Check output.png for result from this example model*


## HOW TO USE


Download and Install labelimg
Properly annotate all images for the dataset
Drag and drop the image and xml files into images/training, validation, and test (80:15:5 split repectively) directory
(MAKE SURE OUTPUT IS XML)


## SETUP


Download anaconda and ensure the proper folders are listed in environment variables

Recommended to be run on VS Code with the "Jupyter" extension


Dependencies (In anaconda prompt, VSCODE Terminal, or Command Prompt)

conda create -n [ENVNAME] python == 3.9
conda activate [ENVNAME]

The following step should already be setup and it should be in the file
--> Download the tensorflow models github repo [https://github.com/tensorflow/models]

```
cd models/research
```
```
sudo apt-get install protobuf-compiler
```
 (NOTE: You may encounter an issue saying there is no "setup.py" file. In that case just manually download it from github instead of using "git clone")
```
protoc object_detection/protos/*.proto --python_out=.         
```
 (NOTE: This should download all necesary dependencies including tensorflow itself)
```
pip install .                                                 
```
WORKING AS OF 2025/03/28
