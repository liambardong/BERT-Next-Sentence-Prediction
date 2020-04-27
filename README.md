# bert_question_training

## Setting up Data
* on your local computer download
* To set up the data you need to use `question_answer_dataclean.py` to clean the data and get the testing, training, and development datasets.

* You need to also change the path for the data to be correct for your setup.

* It will only take specific number of samples from each question type so you need to change the number of samples you want to grab based on how many points of data you have in each question type.

* Once you have run the python script you should see `train.tsv` , `dev.tsv` , `test.tsv` .

* There should also be a test_results.tsv which is the data with the test results but with the actual classification.

## Create a Google Cloud Project
* First create a google cloud project following the steps in this link
https://cloud.google.com/resource-manager/docs/creating-managing-projects

## Creating GC Storage Bucket
* Follow the instructions on this link
https://cloud.google.com/storage/docs/creating-buckets#storage-create-bucket-console
  * use the steps for **Console**
## Setting up the TPU VM
* Follow the instructions on this link
<https://cloud.google.com/tpu/docs/quickstart>

* Once you are set up ssh into the TPU VM instance and clone the bert repository using this command and clone this repository to the VM

`git clone https://github.com/google-research/bert.git `

`git clone https://github.com/whooosreading/bert_question_training.git`

* Then move `next_sentence_prediction.py` from `bert_question_training` to the `bert` repository.

* run `sudo apt install emacs` so that you can edit the path for the data.

## Running the training model

*  Press the symbol at the top right of the screen and select upload file and download `train.tsv` , `dev.tsv` , and `test.tsv`  from your local directory.

<img src="images/download_files_to_vm.png" >

 Talk about the parameters that need to be passed for the program

 Talk about the output you get
