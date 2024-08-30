# # Start with a Miniconda image as the base
# FROM continuumio/miniconda3

# # Set the working directory inside the container
# WORKDIR /app

# # Copy the environment.yml file into the container
# COPY environment.yml .

# # Create the conda environment with the dependencies
# RUN conda env create -f environment.yml

# # Set the environment variable for the Conda environment name (for better maintainability)
# ENV CONDA_ENV=myenv

# # Activate the environment
# SHELL ["conda", "run", "-n", "$CONDA_ENV", "/bin/bash", "-c"]

# # Make sure the environment is activated by default when the container starts
# # ENV PATH /opt/conda/envs/$CONDA_ENV/bin:$PATH
# ENV PATH="/opt/conda/envs/$CONDA_ENV/bin:$PATH"

# # Copy the rest of the application code into the container
# COPY . .

# # Command to run your training script
# CMD ["python", "test.py"]

FROM node:16-alpine

RUN apk add -U git curl