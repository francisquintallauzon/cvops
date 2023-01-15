# DVC

DVC is "Data Version Control is a data versioning, ML workflow automation, and experiment management tool that takes advantage of the existing software engineering toolset you're already familiar with (Git, your IDE, CI/CD, etc.). DVC helps data science and machine learning teams manage large datasets, make projects reproducible, and better collaborate."

[DVC Documentation](https://dvc.org/doc)


## Features
* Version control for datasets
* DVC Pipelines
* Works with multiple data storage solutions (gdrive, azure, S3, ssh, etc)
* Compare report between branches


## Setup project with remote repo in Google drive

First initialize DVC in your project

    DVC init

Setup dvc remote with google drive (the storage where data will lives).  Let's say the google drive link to where you want the DVC repo to live is something like `https://drive.google.com/drive/u/0/folders/1fV42SvMTkj0CxtFYgHsQPtNugdLjowfJ`, then copy the last part of the link and type

    DVC remote add -d gdriveremote gdrive://1fV42SvMTkj0CxtFYgHsQPtNugdLjowfJ

To start tracking your project project data with DVC, simply add the data path to dvc.  In the context of this project, it would be :

    DVC add ./data/datasets

At this point, you can push your data to the remote repo.

    DVC push

DVC then will add a *.dvc file at the base of your data folder.  In this case, it would then be called `datasets.dvc`.

Setting up a github action that pulls the DVC

Create a Google service account that will allow an app to login to google drive.
[Access Google Drive with a Service Account](https://www.labnol.org/google-api-service-account-220404)

Allowing github action access to google drive
[Secrets in github actions](https://docs.github.com/en/actions/security-guides/encrypted-secrets#accessing-your-secrets)


Declare pipeline for getting data

    DVC run -n get_data -d get_data.py -o data_raw.csv --no-exec python get_data.py

## Setting up Github actions
There is a nice stack overflow answer that explains how to setup Github actions with DCV and Google Drive using GCP service accounts:  [Automate DVC authentication when using github actions](https://stackoverflow.com/questions/74017026/automate-dvc-authentication-when-using-github-actions/75196751#75196751)

Doing it this ways solves the issue where you have to manually allow access to google drive to DVC, which is not possible with Github actions.

## Potential flow
* Current production algorithm is in master.
* Develop a new version of the algo in a branch
* DVC can compare metrics between branches and produce report.

For this, look into the [MLOps Tutorial #3: Track ML models with Git & GitHub Actions](https://www.youtube.com/watch?v=xPncjKH6SPk)



## References

Tutorials used for this documentation
* [MLOps Tutorial #2: When data is too big for Git](https://www.youtube.com/watch?v=kZKAuShWF0s)
* [MLOps Tutorial #3: Track ML models with Git & GitHub Actions](https://www.youtube.com/watch?v=xPncjKH6SPk&t)

Hosting
* [Self hosted GPUs with Github Actions](https://iterative.ai/blog/cml-self-hosted-runners-on-demand-with-gpus/)

Data version control workflow
* [Data Version Control With Python and DVC](https://realpython.com/python-data-version-control/#practice-the-basic-dvc-workflow)
* [Continuous Integration with CML and Github Actions](https://mlops-guide.github.io/CICD/cml_testing/)


Model Testing
* [Lecture 03: Troubleshooting & Testing (FSDL 2022)](https://youtu.be/RLemHNAO5Lw)
* [How Should We Test ML Models? with Data Scientist Jeremy Jordan](https://www.youtube.com/watch?v=k0naEYedv5I)
