# FakeNews

Prequisite
pyhton3.6

install libraries with following commands

pip install django
pip install textblob
pip install pandas
pip install matplotlib
pip install tweepy
pip install numpy
pip install tweepy
pip install scikitlearn
pip install sklearn
pip install seaborn


If pip won't work installing above packages use pip3 install

INSTALLATION
clone ...
navigate to prediction.py 
fakenews-->prediction.py
change default path of file to your current absolute path. 
load_model = pickle.load(open(!set the filepath of final model-->!'D:/Django/FakeNews/test_fake/Dataset/final_model.sav', 
'rb'))

(optional part for training dataset)
navigate to classifier.py 
change file to your file path 
D:/Django/FakeNews/test_fake/Dataset/DataPrep.py-----> yourpath/DataPrep.py
Do same for Featureselection 
Also change path of DataPrep in Featureselection.py file


Implementation


open cmd
....test_fake>>python manage.py runserver
goto browser  type 127.0.0.1:8000 in url
You are ready to go...
if migration error occurs
open cmd
....test_fake>>python manage.py migrate


MAIN FILES
manage.py --> Runserver
prediction.py --> Fake news detection
main.py --> Twitter analysing tweets
home.html --> webpage
final_model.sav --> News analysier true or false
