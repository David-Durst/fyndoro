* How To Install

1. Install Conda - https://www.anaconda.com/download/
2. Checkout this repository - git clone git@github.com:David-Durst/fyndoro.git
3. Create the environment - conda env create -f environment.yml (or mac_env.yml if using a mac)
3. Get a Google Vision API key - https://cloud.google.com/vision/docs/auth
    a. Put the API key in a file called API_KEY at the top-level of this repository. .gitignore is configured to ignore this file.
4. Run the learner - python -m uncertain.learn
