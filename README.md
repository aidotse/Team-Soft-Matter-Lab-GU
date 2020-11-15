# Adipocyte Cell Challenge

## Instructions for staining

### Launching docker

First, check if the container is already running. Run `docker ps` and look for a container with IMAGE and NAMES _notebooks_

#### If the container is already running
Attach to the container by running `docker exec -it notebooks sh`

#### If the container is not running
Launch the docker container by running `docker run -it --name notebooks --rm -p 8890:8890 -v /home/group1/:/workspace/ notebooks`



### Starting Jupyter Server

Run `jupyter notebook --port=8890`

On the client computer, go to `10.80.4.52:8890` in a web-browser.

If prompted, enter the key shown by jupyter notebook on the host.

**REMEMBER TO SHUT DOWN THE KERNEL WHEN LAUNCHING A NEW NOTEBOOK**

### Predicting on 60X data

Launch the notebook named **Stain 60X Data**.

* Set `DATASET_PATH` to the path containing the 60X input images
* Set `OUTPUT_PATH` to the desired output path (can be the same as `DATASET_PATH`)

Run each cell in order. Under section **4**, cell 14, the execution time of the model is shown.

### Predicting on 40X data

Launch the notebook named **Stain 40X Data**.

* Set `DATASET_PATH` to the path containing the 40X input images
* Set `OUTPUT_PATH` to the desired output path (can be the same as `DATASET_PATH`)

Run each cell in order. Under section **4**, cell 14, the execution time of the model is shown.

### Predicting on 20X data

Launch the notebook named **Stain 20X Data**.

* Set `DATASET_PATH` to the path containing the 20X input images
* Set `OUTPUT_PATH` to the desired output path (can be the same as `DATASET_PATH`)

Run each cell in order. Under section **4**, cell 14, the execution time of the model is shown.

## Instructions for training

### Launching docker

Launch the docker container by running `docker run -it --rm -p 8890:8890 -v /home/group1/:/workspace/Team-Soft-Matter-Lab-GU  notebooks`

### Starting Jupyter Server

Run `jupyter notebook --port=8890`

On the client computer, go to `10.80.4.52:8890` in a web-browser.
If prompted, enter the key shown by jupyter notebook on the host.

### Training on 60X data

Launch the notebook named **Train 60X Virtual Stainer**.

* Set `DATASET_PATH` to the path containing the 60X input images
* Set `OUTPUT_PATH` to the desired output path (can be the same as `DATASET_PATH`)


### Training on 40X data

Launch the notebook named **Train 40X Virtual Stainer**.

* Set `DATASET_PATH` to the path containing the 40X input images
* Set `OUTPUT_PATH` to the desired output path (can be the same as `DATASET_PATH`)


### Training on 20X data

Launch the notebook named **Train 20X Virtual Stainer**.

* Set `DATASET_PATH` to the path containing the 20X input images
* Set `OUTPUT_PATH` to the desired output path (can be the same as `DATASET_PATH`)

Run each cell in order. 
