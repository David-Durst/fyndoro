* How To Install

1. Install Conda - https://www.anaconda.com/download/
2. Checkout this repository - git clone git@github.com:David-Durst/fyndoro.git
3. Create the environment - conda env create -f environment.yml (or mac_env.yml if using a mac)
3. Install gcloud terminal tools - https://cloud.google.com/sdk/downloads
3. Get access to the Google Vision API - https://cloud.google.com/vision/docs/before-you-begin
3. Provide gcloud access to the application - gcloud auth application-default login --no-launch-browser
    a. documentation for command - https://cloud.google.com/sdk/gcloud/reference/auth/application-default/login
3. Install imagemagick - brew install imagemagick for mac
3. Install gshuf (or shuf on linux) - brew install coreutils for mac
4. Run the learner - python -m uncertain.learn
