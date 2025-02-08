# sample_fast_api

# if your are runnig this project in specific conda environment
conda activate sample-api

# to create requirement.txt file
pipreqs .

# if above not works than only use force parameter
pipreqs . --force

# to create docker image and container
docker build -t mlops_apis .

# or
docker buildx build -t mlops_apis .

# to run your container with port mapping
docker run -p 5000:5000 mlops_apis

# to run docker compose
docker-compose up --build -d
