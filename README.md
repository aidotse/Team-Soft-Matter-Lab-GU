# Adipocyte Cell Challenge

## Expected folder structure for models

Before starting, make sure the folder structure for models matches the following:

    Team-Soft-Matter-Lab-GU/
    . . models/
    . . . . 20x/
    . . . . ...
    . . . . 40x/
    . . . . ...
    . . . . 60x/
    . . . . ...
    

## Instructions for staining

Move into the Team-Soft-Matter-Lab directory.

### Building the container

Only needed on a new computer system.

Run `docker build . -t notebooks`

### Launching docker

Check if the container is already running. Run `docker ps` and look for a container with IMAGE and NAMES _notebooks_

#### If the container is already running
Attach to the container by running `docker exec -it notebooks sh`

#### If the container is stopped

Start the container by running `docker start notebooks`

#### If the container is not running
Launch the docker container by running `docker run -it --name notebooks --rm -p 8890:8890 -v /home/group1/:/workspace/ notebooks`


### Starting Jupyter Server

Run `jupyter notebook --port=8890`

This will return an address in the form:

http://127.0.0.1:{port}/?token={key}

You may need to scroll up to see it.

Copy that address, replacing 127.0.0.1 with 10.80.4.52, and paste it into a web browser on the local computer.

An example url would be:

http://10.80.4.52:8890/?token=abcdefghijklmnopqrst0123456789abcdefghijklmnopqr

### Predicting on 60X data



Launch the notebook named **Stain 60X Data**.

* Set `DATASET_PATH` in section 1.2 to the path containing the 60X input images
* Set `OUTPUT_PATH` in section 1.2 to the desired output path (can be the same as `DATASET_PATH`)

Run each cell in order. Under section **4**, cell 14, the execution time of the model is shown.

**Make sure to shut down the kernel BEFORE moving on to the next notebook**
This is done by pressing kernel, followed by shutdown.

### Predicting on 40X data

Launch the notebook named **Stain 40X Data**.

* Set `DATASET_PATH` to the path containing the 40X input images
* Set `OUTPUT_PATH` to the desired output path (can be the same as `DATASET_PATH`)

Run each cell in order. Under section **4**, cell 14, the execution time of the model is shown.

**Make sure to shut down the kernel BEFORE moving on to the next notebook**
This is done by pressing kernel, followed by shutdown.

### Predicting on 20X data

Launch the notebook named **Stain 20X Data**.

* Set `DATASET_PATH` to the path containing the 20X input images
* Set `OUTPUT_PATH` to the desired output path (can be the same as `DATASET_PATH`)

Run each cell in order. Under section **4**, cell 14, the execution time of the model is shown.

**Make sure to shut down the kernel BEFORE moving on to the next notebook**
This is done by pressing kernel, followed by shutdown.

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

**Make sure to shut down the kernel BEFORE moving on to the next notebook**
This is done by pressing kernel, followed by shutdown.

### Training on 40X data

Launch the notebook named **Train 40X Virtual Stainer**.

* Set `DATASET_PATH` to the path containing the 40X input images
* Set `OUTPUT_PATH` to the desired output path (can be the same as `DATASET_PATH`)

**Make sure to shut down the kernel BEFORE moving on to the next notebook**
This is done by pressing kernel, followed by shutdown.

### Training on 20X data

Launch the notebook named **Train 20X Virtual Stainer**.

* Set `DATASET_PATH` to the path containing the 20X input images
* Set `OUTPUT_PATH` to the desired output path (can be the same as `DATASET_PATH`)

Run each cell in order. 

**Make sure to shut down the kernel BEFORE moving on to another notebook**
This is done by pressing kernel, followed by shutdown.
