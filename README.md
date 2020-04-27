# bert_question_training

## Setting up Data
* Clean you data to have three .tsv files (train.tsv, test.tsv, dev.tsv)
* Each should have two columns with the two sentences
* The three files should be in the same directory

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

* Then move `next_sentence_prediction.py` from this reposistory to the `bert` repository.

* run `sudo apt install emacs` so that you can edit the path for the data.

## Running the training model

*  Press the symbol at the top right of the screen and select upload file and download `train.tsv` , `dev.tsv` , and `test.tsv`  from your local directory.

<img src="images/download_files_to_vm.png" >

## Parameters 
 * `--task_dir` - Directory path to folder with train.tsv, dev.tsv, and test.tsv
 *

 Talk about the parameters that need to be passed for the program

 Talk about the output you get
