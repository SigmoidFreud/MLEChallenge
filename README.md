# MLEChallenge

Installation Instructions
1. First it is recommended that you follow the instructions for installing anaconda for your platform here: https://docs.anaconda.com/anaconda/packages/pkg-docs/
2. Then install docker so that when the flask app runs the redis image can be run in a cross platform env - https://docs.docker.com/engine/install/
3. Then clone the repository to your local machine and open a terminal window from inside the repo directory
4. run the command pip install -r requirements.txt
5. then run the following commands: 
6. cd featureData
7. python create_process_feature_data.py
8. This will create the database, do all of the data processing, create the test data, and produce the 2 logistic regression models
9. Now execute the following commands in the terminal
10. cd ..
11. flask run
12. now in a seperate terminal tab/window running python mle_challenge_test_script will produce the fully_tested_endpoint_data.json file that contains all of the input and output data to the various endpoints that are to be tested
