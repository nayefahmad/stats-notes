# stats-notes

## Contents 
1. [Calculating power of t-test of regression slope coefficient](https://github.com/nayefahmad/stats-notes/blob/main/src/2022-03-28_power-of-test-of-regression-slope.md)

2. [Optimization using the optim function in R](https://github.com/nayefahmad/stats-notes/blob/main/src/2022-11-28_simple-optimization-with-optim.md)

3. [Demonstrating the importance of correctly quantifying chance using systematic variables](https://github.com/nayefahmad/stats-notes/blob/main/src/2022-11-30_unexplained-variance-with-and-without-systematic-factors.ipynb)

4. [Notes on ridge, lasso and elastic net regression](https://github.com/nayefahmad/stats-notes/blob/main/src/2022-12-09_lasso-regression-demo.ipynb)

5. [Simple demonstration of a random effects model](https://github.com/nayefahmad/stats-notes/blob/main/src/2022-12-15_dummy-variable-vs-random-effect.md)

## Repo structure 

- `src` directory: code files 
- `.pre-commit-config.yaml`: config for use with `pre-commit`. It specifies what hooks to use. 
  Once this file is created, if you run `pre-commit install`, the pre-commit tool will populate the 
  `pre-commit` file in the `./.git/hooks` directory. Helpful references: 
    - [Automate Python workflow using pre-commits: black and flake8](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/)
    - [Keep your code clean using Black & Pylint & Git Hooks & Pre-commit](https://towardsdatascience.com/keep-your-code-clean-using-black-pylint-git-hooks-pre-commit-baf6991f7376)
    - [pre-commit docs](https://pre-commit.com/#)
- `.flake8`: config for Flake8. Mainly used to specify max-line-length=88, to match [Black's default](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
- `.isort.cfg`: config file used for `isort`. See [docs](https://pycqa.github.io/isort/docs/configuration/black_compatibility.html)
- `requirements.txt`: python packages used 
