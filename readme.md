Official Tensorflow doc: https://www.tensorflow.org/get_started/get_started


OSX Getting start with tensorflow

Step 1: Install docker https://www.docker.com/

Step 2: Pull Tensorflow docker image `docker pull tensorflow/tensorflow`

Step 3: Create a new repo for all your tenserflow script, work `~/repos/tenserflow`

Step 4: Run Tensorflow docker container
        `docker run -it --rm -v ~/repos/tensorflow:/notebooks/repos/tensorflow tensorflow/tensorflow bash`
        This Will start container in interactive bash mode, and yor repo '~/repos/tenserflow' will be availabel inside '/notebooks/repos/tensorflow'

Step 5: Run first script: `python /notebooks/repos/tensorflow/test.py`
