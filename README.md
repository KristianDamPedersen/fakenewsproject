# Fake News Project
Our shared repository related to the Fake News Project in Data Science.

## Contributing
We've agreed to the following guidelines.
#### Code quality
Before merging any code into main, it should fulfil the following criteria:
* All tests (unit or otherwise) should be passing
* All new code should be properly tested
* All new code should be adequately documented
* If the changes affect the rapport (either in being able to write more, or modify what's already there), the rapport should be updated accordingly.
We aim to merge to main as frequently is possible, so please create pull requests and go through the above steps after each small change.

#### Branch rules
* We are unable to push to main without a pull req.
* In order to merge to main, create a pull req. your code should fulfil the code quality requirements.
* The pull req. must be reviewed by one other person before being able to be merged.


#### Folder strucure
Here is an overview of our folder-structure:
* fakenewsproject/
  * README.md
  * src/
    * my_module/
      * \_\_init\_\_.py
      * my_module.py
      * my_module.md
    * .../
  * report/
    * preamble.tex
    * master.tex
    * report.pdf
  * notebooks/
    * my_notebook.ipynb
  * tests/
    * my_module/
      * my_module_test.py

##### Managing modules 
The */src* folder is designed to be the place where we keep all of our python modules, such that we can reuse our code across the project.
Each module is meant to be its own contained piece of functionality. For example: You might have a module dedicated to
data gathering, another dedicated to data cleaning, and yet a third dedicated to visualization. The point of each module,
it to provide functions (and other functionality) that is useful and can be imported into other bits of the project.
###### Adding a module
When we want to add a module we have follow these steps:
1. Create a new directory under src/
2. Add an \_\_init\_\_.py file (this allows us to import functions from the files within).
3. Add a .py file, which name need to match the directory name (to make imports easier).
4. Add a .md file that likewise share its name with the directory name.
5. (Any other structure you introduce is up to you, however all the exported functions has to be from the original .py file)
6. Finally, unde the /tests directory, add a directory mathing your module name. Inside the directory, you can add the relevant tests.

##### Notebooks
All notebooks can be found under the notebooks/ folder. We have the following guidelines for them:
1. All notebooks should be self contained, they should not export functionality to any other notebook, or modules.
2. Useful pieces of code is encouraged to be extracted and made into modules (remember to test them!)

###### To illustrate the above, i have made some placeholder files designed the "hello_there" module.

