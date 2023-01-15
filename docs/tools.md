# Project Tooling & Automations

## Poetry : python packaging
[Poetry][17] is a tool for making python packaging and dependency management easy and used int his project. 

## Markdown : documentation tooling
This project uses [Markdown](https://daringfireball.net/projects/markdown/) "Markdown is intended to be as easy-to-read and easy-to-write as is feasible"  according to John Gruber.  This is a nice guide to [Markdown basic syntax][19] 

Documation build automation and serving is done with [Read the Docs][12] and [mkdocs](https://www.mkdocs.org).  To setup documentation, the following steps were followed. 
1. Setup your [mkdocs project](https://docs.readthedocs.io/en/stable/intro/getting-started-with-mkdocs.html)
2. Create the `.readthedocs.yaml` file from by following those [instructions](https://docs.readthedocs.io/en/stable/config-file/v2.html)
2. Follow the [Read The Docs tutorial](https://docs.readthedocs.io/en/stable/tutorial/index.html) to link your github project with Read the DOcs. 


## Pylint : static Code Analysis (Linting)
Linting is a process of running a static code analysis witht he goal of flagging programming errors, bugs, stylistic errors and suspicious constructs [[2]].  An example of a rule enforced by linting in this project : use of [snake_case][6] which suggests that complex token names should be separated by underscored. The linter used in this project is [Pylint][3].  Development environment like Visual Studio Code integrates linting tools like pylint and automatically highlight issues. Pylint settings are located in the `.pylintrc` file

## Black : automatic code formatting
[Black][7] is used for automatic code formatting.  Automatic code formatting automatically modifies code to enforce a programming style, ensuring a uniform code style and making code maintainance easier. 

## Visual Studio Code : development environment
This project was developped using [Visual Studio Code][4] and includes a very limited set of workspace settings for activating tools used to develop this project.  As of writing this documentation, the following tools are integrated with Visual Studio Code.
* Automatic Testing : Pytest
* Automatic Code Formatting : Black (runs when saving files)
* Linting : Pylint (runs when saving files)

Visual Studio Code workspace settings are located in `./vscode/settings.json`.  Settings can be acces with `⌘,` on Mac and `ctrl+,` on other OS.  

By the way, the difference between user settings and workspace settings? From [Visal Studio Code documentation][5]:
* User Settings - Settings that apply globally to any instance of VS Code you open.
* Workspace Settings - Settings stored inside your workspace and only apply when the workspace is opened.

Of course, using Visual Studio Code is not required, nor recommended to work his project.  Use whatever tool you like!!

## Git : source control
[Git][9] is a "free and open source distributed version control system designed to handle everything from small to very large projects with speed and efficiency"[[9]]. 

## pre-commit : pre-commit verifications
Git allows automatically calling scripts at every commits (and other particular events). "Git hook scripts are useful for identifying simple issues before submission to code review" [[8]].  [pre-commit][8] is used for managing pre-commit verifications.  As of writing this documentation, it automatically 
* yaml files
* Fix end of files
* Trim Trailing Whitespaces
* Runs automatic code formatting (Black)
* Runs Pytest pre-commit tests

## bump2version : versionning automation
[bump2version][10] automates project versionning.  It is especially useful where version number appear in multiple locations in projects.  Not yet used in this project as of writing this, but it is planned.  This project uses [Semantic Versionning 2.0](https://semver.org/spec/v2.0.0.html)

See the [versionning page](./versionning.md) for more details

## Github workflow and automations
This project uses [Github flow][11] as a contribution workflow. A pull request template is implemented in `.github/pull_request_template.md`

[Github Actions][14] are used to automate workflows.  Github actions scripts are located in `.github/workflow/` 

* To create your own pull request template, [see this][16]
* To create github actions of your oww, [see this][15]

## CML : continuous machine learning
[CML][18] is a tool for continuous integration in ML.  Using this tool, a github action that trains the model, create a training report and adds as a pull request comment is located in `.github/workflow/cml.yaml`.  
* [A tutorial on the CML tool from Iterative](https://youtu.be/9BgIDqAzfuA)
* [Continuous machine learning explained](https://mlops-guide.github.io/CICD/cml_testing/)

## DVC : dataset version control, experiment tracking
[DVC][20] is a tool for dataset version control, experiment tracking and monitoring. A full page dedicated to DVC is [here](./dvc.md)


# Background documentation 

## Project Struture
The following articles were used as inspiration this project folder structure : 
* [Folder Structure for Machine Learning Projects](https://medium.com/analytics-vidhya/folder-structure-for-machine-learning-projects-a7e451a8caaa)
* [Machine Learning: Models to Production](https://medium.com/analytics-vidhya/machine-learning-models-to-production-72280c3cb479)

## Refactoring a data science project
Youtube series on refactoring a data science project by arjan_codes. 
* [Part 1](https://www.youtube.com/watch?v=ka70COItN40&t)
* [Part 2](https://www.youtube.com/watch?v=Tx4AxbQNv3U)
* [Part 3](https://www.youtube.com/watch?v=8fFqakxhW84)

## Learning Referecnes
Courses
* [Full Stack Deep Learning 2022](https://youtube.com/playlist?list=PL1T8fO7ArWleMMI8KPJ_5D5XSlovTW_Ur)
* [ML Ops Tutorials using iterative.io tools](https://www.youtube.com/playlist?list=PL7WG7YrwYcnDBDuCkFbcyjnZQrdskFsBz)
* [ML Ops Guide](https://mlops-guide.github.io/)

## Virtual environments
A python virtual environments is a "self-contained directory tree that contains a Python installation for a particular version of Python" [[1]] and   
* [Official Python Documentation](https://docs.python.org/3/tutorial/venv.html)
* [Python Virtual Environments Primer by Martin Breuss on RealPython](https://realpython.com/python-virtual-environments-a-primer/)
* [Managing Application Dependencies](https://www.fullstackpython.com/application-dependencies.html)


ML Ops Definitions
 * By [Databricks](https://www.databricks.com/glossary/mlops)
 * By [Arrikto](https://www.arrikto.com/mlops-explained/)

ML Ops Challenges
 * [Why Production Machine Learning Fails — And How To Fix It]( https://www.montecarlodata.com/blog-why-production-machine-learning-fails-and-how-to-fix-it/)
 * [The Ultimate Guide: Challenges of Machine Learning Model Deployment](https://towardsdatascience.com/the-ultimate-guide-challenges-of-machine-learning-model-deployment-e81b2f6bd83b)
 * [Model Deployment Challenges: 6 Lessons From 6 ML Engineers](https://neptune.ai/blog/model-deployment-challenges-lessons-from-ml-engineers)

 Maturity model in ML Ops
 * [Three Levels of ML Software](https://ml-ops.org/content/three-levels-of-ml-software)
 


# Could be usefull but not used in this project 

## Scikit-Learn Pipelines
* [Scikit-Learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
* [Basic Tutorial](https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.04-Feature-Engineering.ipynb)
* [Advanced Tutorial](https://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html)
* [Pipelines & Custom Transformers in scikit-learn](https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156)

## MLFlow : Machine Learning LifeCycle Management
An open source platform for the machine learning lifecycle, according to how they define themselves. 
* [MLFlow](https://mlflow.org/)
* [Youtube Series ML Lifecycle by Isaac Reis](https://youtube.com/playlist?list=PL6qNEZP_yH2mnbtwmvjuL6EmWhcPyaVlg)
* [Getting Started on Databricks Community Edition](https://docs.databricks.com/getting-started/index.html)

## Hypertops : Distributed Asynchronous Hyper-parameter Optimization
* [Hyperopt](http://hyperopt.github.io/hyperopt/)
* [Using MLFlow with HyperOpt for Automated Machine Learning](https://medium.com/fasal-engineering/using-mlflow-with-hyperopt-for-automated-machine-learning-f1f3e110500)



[1]:  ttps://docs.python.org/3/tutorial/venv.html
[2]:  https://en.wikipedia.org/wiki/Lint_%28software%29
[3]:  https://pylint.pycqa.org/en/latest/
[4]:  https://code.visualstudio.com/
[5]:  https://code.visualstudio.com/docs/getstarted/settings
[6]:  https://en.wikipedia.org/wiki/Snake_case
[7]:  https://github.com/psf/black
[8]:  https://pre-commit.com/
[9]:  https://git-scm.com/
[10]: https://github.com/c4urself/bump2version
[11]: https://docs.github.com/en/get-started/quickstart/github-flow
[12]: https://readthedocs.org/
[13]: https://docs.readthedocs.io/en/stable/tutorial/index.html
[14]: https://docs.github.com/en/actions
[15]: https://docs.github.com/en/actions/quickstart
[16]: https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository
[17]: https://python-poetry.org/
[18]: https://cml.dev/
[19]: (https://www.markdownguide.org/basic-syntax/) 
[20]: (https://dvc.org/doc)