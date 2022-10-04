# Python Starter for *Computational Physics 2022*
Dear Students, \
welcome to the accompanying Python repository of the Computational Physics course of 2021.

You can use this Git repository as an 'upstream remote' (more on that later) while completing the course - as all the necessary information and data will be provided here, as well as in the TeachCenter.

## Setup and Usage
*Git* is a Version Control System (VCS), which means that not only the contents of this repository are stored, but also their change history.
*GitLab* on the other hand, is a service, in this case offered by TU Graz, that allows for collaborating on Git repositories with one another.
Also, it adds a simple to use web interface as well as an in-browser editor.

[Here](https://www.youtube.com/playlist?list=PLhW3qG5bs-L8YSnCiyQ-jD8XfHC2W1NL_) is an introduction tutorial into *Git*, including installing gitlab, ssh key generation, and explaining the functionality of add/commit/pull.

Each tuple `(state of the repository, point in time, author)` is referred to as a *commit*. They track what happens exactly and you can productively use them to store batches of 'complete' (and possibly working) changes to your code. When you are done with a coherent set of changes, make a commit and push it to the repository.

That also means you can go back to any stored state, any time!
And work together on the same set of files with multiple people, simultaneously.

**The first step** for using this repository is to **fork** it (button on the top right). This essentially creates a copy of the whole repo. You *must* set the visibility level to "**Private**" since you will complete the assignments on your own, or in groups (confer below).

You should now have a repository with the path `gitlab.tugraz.at/[Your User ID]/computational-physics-in-python-2021`.
Of course, you may rename it as you deem fitting, since it is now your own.

You may also **use your forked repo for handing in the assignments**, instead of the TeachCenter.
- Please leave a note in the corresponding deliverable ('Abgabe') on TeachCenter, linking to your TUGraz GitLab repository.
- As soon as you have your own repository (from forking the main one), from there please go to 'Project Information' -> 'Members' -> 'Invite Group'.
- Invite a group named "Computational Physics", which comprises the assessment team of Professors and Tutors.
- If you want to work in teams, you may invite your colleagues (up to 5 per group) to the repository. Do this via 'Project Information' -> 'Members' -> 'Invite per Mail'. Use the 'Maintainer' role.
- In case you want to switch back to using the TeachCenter for some reason, simply ZIP the project folder as usual and upload it to the TC.

### Cloning the repository for local use
First, [install git](https://github.com/git-guides/install-git). For Linux, simply `apt install git`. On Windows, consider using the *Windows Subsystem for Linux*. \
*You may be able to skip this step, if you use an IDE.*

To authenticate with the GitLab server for the upcoming *clone* process, add your public SSH key on GitLab. [This post](https://stackoverflow.com/a/50079018/5832850) explains it well.

In order to work on the repository locally, please *clone* it. \
`git clone git@gitlab.tugraz.at:[your-user-id]/computational-physics-in-python-2021.git`

This copies all the information from the online source to a local folder, where you can work on it.

### Working on the repository
Please make sure to install and use an *Integrated Development Environment* (IDE) such as

- VSCode (which is free and open source!). Find the installation instructions [here](https://code.visualstudio.com/). For Linux users, simply `snap install code`.
- or JetBrains PyCharm (who have a community version that is free, or with a student account you can also get the professional version). Find the installation instructions [here](https://www.jetbrains.com/pycharm/). For Linux users, simply `snap install pycharm-professional` or `snap install pycharm-community`. They have a bit better introspection tools for Python than VSCode does, but judge this by yourself :)
- Anaconda, which has a lot of beginner-friendly tooling, but is by far not as feature-rich.
- Use your own Editor - whichever you want! But make sure you like it :) It should be helpful for programming.

The two mentioned IDEs both have built-in support for Git.
Try the *action search* (`Ctrl-Shift-P` in VS Code or `Ctrl-Shift-A` in PyCharm) of both IDEs, they're the best tools for getting started.

**Start programming!**

When you are done with a set of changes (maybe when finishing work for tonight, etc. - it really depends on you - but in general, making many commits is often better than few), do: \
`git add ./ex1/myfile.py` \
`git commit` \
`git push`

If you want to get the updated files (maybe after a colleague worked on them) from GitLab to your local machine, use: \
`git pull` 

These steps can also be done using a graphical interface of an IDE.

Beware, the learning curve for all of this might be rather steep in the beginning - but well worth it in the end, surely even outside the scope of this course.

### Installing Packages (Modules)
We recommend using a virtual environment for isolating this project's packages from the rest of your system packages.
This approach allows for a great deal of flexibility and makes it possible to reproduce problems easier, nevertheless **this is an optional step**.
It might be unnecessary for beginners.

To create a virtual environment, we recommend you to use [pipenv](https://pipenv.pypa.io/en/latest/) or [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/). Refer to the documentation there.
IDEs also offer graphical tools for that!

(Note: you may use `Pipfile`s in addition to the `requirements.txt` file, even though the `requirements` approach is preferred.)

When you are done creating the *virtualenv*, make sure it is activated, either in your IDE or in the terminal.
Running `pip freeze` should list the currently installed packages and their exact versions (in the same way that the `requirements.txt` does it). As you just created the virtualenv, it should be an almost empty list.

To get started, install the packages in the `requirements.txt`, which per default includes numpy and matplotlib. \
`pip install -r requirements.txt`

If you don't need these packages, remove them from the `requirements.txt`, if you need other packages, install them via: \
`pip install [package name]`

Try for instance [tqdm](https://pypi.org/project/tqdm/), which can help you with generating progress bars for the upcoming iterative algorithms. \
`pip install tqdm`

To let others (tutors or colleagues) know you're using the package and which version, add a corresponding line to the `requirements.txt` file - for instance: \
`tqdm==4.46.0`

Other helpful libraries for the scientific context include *scipy* and *uncertainties*.
To make your numerical Python code much faster, use [Numba](https://numba.pydata.org/)!

### Linting
To run the linters locally, first make sure you ran `pip install -r requirements.txt` with `black` and `pylint` included.

Processing your files is easy then, simply \
`black ./ex1/myfile.py` and `pylint ./ex1/myfile.py`

They will give you warnings and tips on what to improve in your code. Both of them are highly configurable, from the `pyproject.toml` file - but this repo already includes a sensible default for the course. Black primarily reformats your code, while pylint gives hints.
Take the hints with a grain of salt however, they might not apply in your case - but usually it's good to follow them.

Some resources:
- [Integrate Black with your Editor](https://black.readthedocs.io/en/stable/integrations/editors.html)
- [The Different Options for PyLint](https://www.getcodeflow.com/pylint-configuration.html)

### Merging from Upstream
As there will be changes in the main ('upstream') repository, please make sure to merge them into your own fork from time to time. To do this, simply: \
`git remote add upstream git@gitlab.tugraz.at:computational-physics/computational-physics-in-python-2021.git` \
`git pull upstream master`

## Files in this Repository
This starter contains a few files:

- The `README.md`, which is the file you are reading right now.
- A `pyproject.toml` file, which configures some of the Python tools you will be using.
- The `requirements.txt` defines which Packages from the Python Package Index ([PyPi](https://pypi.org/)) you are using in your code, and what versions they use. For example, the current file says you are using numpy and matplotlib.
- A `.gitignore` file tells Git which types of files, or paths, it should ignore in general. Those files will not be uploaded to GitLab.

Please put your code into the `ex1`, `ex2`, etc. folders for each exercise, respectively.

### Adapt this repo for another language
You may of course use this repository for your preferred programming language as well, like C, C++, Java, Haskell, Julia, etc.!
Simply remove the `pyproject.toml` and `requirements.txt` as you will not need them.
Commit and push the files as you deem fitting in this case.

## Have Fun!
If you need any help, we will of course be there for you! \
And as always in the programming world, search engines are your best friend! Many people probably stumbled across your very problem before, just search for it online.

As a final note, try using [a debugger](https://docs.python.org/3/library/pdb.html) for finding errors in your code -
it can be very insightful (in almost all cases).
IDEs are very helpful with that
(Confer [VSCode](https://docs.microsoft.com/en-us/visualstudio/python/debugging-python-in-visual-studio?view=vs-2019)
and [PyCharm](https://www.jetbrains.com/help/pycharm/debugging-code.html))!

Good luck with the assignments!
