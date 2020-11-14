# Adipocyte Cell Challenge

## Instructions for staining

### Launching docker

Launch the docker container by running `docker run -it -p 8890:8890 -v /home/group1/:/workspace  notebooks`

Move into theteam folder with `cd Team-Soft-Matter-Lab-Gu`

### Starting Jupyter Server

Run `jupyter notebook --port=8890`

On the client computer, go to `10.80.4.52:8890` in a web-browser.

### Predicting on 60X data

Launch the notebook named **Stain 60X Data**.

* Set `DATASET_PATH` to the path containing the 60X input images
* Set `OUTPUT_PATH` to the desired output path (can be the same as `DATASET_PATH`)
* Set `WELLS` to `["B02"] * 12`
* Set `SITES` to `range(1, 13)`

Run each cell in order. Under section **4**, cell 16, the execution time of the model is shown.

### Predicting on 40X data

Launch the notebook named **Stain 40X Data**.

* Set `DATASET_PATH` to the path containing the 40X input images
* Set `OUTPUT_PATH` to the desired output path (can be the same as `DATASET_PATH`)
* Set `WELLS` to `["B02"] * 12`
* Set `SITES` to `range(1, 9)`

Run each cell in order. Under section **4**, cell 16, the execution time of the model is shown.

### Predicting on 20X data

Launch the notebook named **Stain 20X Data**.

* Set `DATASET_PATH` to the path containing the 20X input images
* Set `OUTPUT_PATH` to the desired output path (can be the same as `DATASET_PATH`)
* Set `WELLS` to `["B02"] * 12`
* Set `SITES` to `range(1, 7)`

Run each cell in order. Under section **4**, cell 16, the execution time of the model is shown.

## Instructions for training

### Launching docker

Launch the docker container by running `TBD`

Move into theteam folder with `cd Team-Soft-Matter-Lab-Gu`

### Starting Jupyter Server

Run `jupyter notebook --port=8888`

On the client computer, go to `10.80.4.52:8888` in a web-browser.

### Training on 60X data

Launch the notebook named **Train 60X Virtual Stainer**.

* Set `DATASET_PATH` to the path containing the 60X input images
* Set `OUTPUT_PATH` to the desired output path (can be the same as `DATASET_PATH`)

Run each cell in order. Under section **4**, cell 16, the execution time of the model is shown.

### Training on 40X data

Launch the notebook named **Train 40X Virtual Stainer**.

* Set `DATASET_PATH` to the path containing the 40X input images
* Set `OUTPUT_PATH` to the desired output path (can be the same as `DATASET_PATH`)


### Training on 20X data

Launch the notebook named **Train 20X Virtual Stainer**.

* Set `DATASET_PATH` to the path containing the 20X input images
* Set `OUTPUT_PATH` to the desired output path (can be the same as `DATASET_PATH`)

Run each cell in order. 
