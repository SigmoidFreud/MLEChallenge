# MLEChallenge

Installation Instructions
1. First it is recommended that you follow the instructions for installing anaconda for your platform here: https://docs.anaconda.com/anaconda/packages/pkg-docs/
2. Then clone the repository to your local machine and open a terminal window from inside the repo directory
3. run the command pip install -r requirements.txt
4. then run the following commands: 
5. cd featureData
6. python create_process_feature_data.py
7. This will create the database, do all of the data processing, create the test data, and produce the 2 logistic regression models
8. Now execute the following commands in the terminal
9. cd ..
10. flask run
11. now in a seperate terminal tab/window running python mle_challenge_test_script will produce the fully_tested_endpoint_data.json file that contains all of the input and output data to the various endpoints that are to be tested
