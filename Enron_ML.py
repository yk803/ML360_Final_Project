import os;
import numpy as np;
#import pandas as pd; 
## didn't use pandas cuz found that df runs even slower than python list when building dataset
from collections import Counter;
from nltk.corpus import stopwords;
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score;
from sklearn.linear_model import LogisticRegression;
from sklearn.naive_bayes import MultinomialNB;
from sklearn.svm import SVC;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.ensemble import AdaBoostClassifier;
from sklearn.neural_network import MLPClassifier;
import time;


def make_dict(emails):
    words = [];
    stop = stopwords.words('english');

    for email in emails:
        f = open(email,encoding="utf8", errors='ignore');
        blob = f.read();
        words += blob.split(" ");

    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = "";
        if words[i] in stop:
            words[i] = "";

    dictionary = Counter(words);
    del dictionary[""]
    
    return dictionary.most_common(3000)


def make_dataset(dictionary,emails):

    print('There are {} emails in total.'.format(len(emails)));

    feature_set = [];
    labels = [];
    counter = 0;
    for email in emails:
        data = [];
        f = open(email, encoding="utf8", errors='ignore');
        words = f.read().split(' ');

        for entry in dictionary:
            data.append(words.count(entry[0]));
        feature_set.append(data)
        if "ham" in email:
            labels.append(0);
        else:
            labels.append(1);
        counter += 1;
        if not counter % 2000:
            print(counter,end = '\t');
    print();
    #print(feature_set[0])
    print('dataset built successfully');
    return feature_set, labels


def getScores(y, pred, name):
    print("--------------------- ",name," --------------------")
    print("Accuracy score")
    print(accuracy_score(y, pred))
    print("F1 score")
    print(f1_score(y, pred, average='macro'))
    print("Recall")
    print(recall_score(y, pred, average='macro'))
    print("Precision")
    print(precision_score(y, pred, average='macro'))
    return accuracy_score(y, pred)


def logisticRegression(train_x,train_y,val_x=None,val_y=None,test_x=None,test_y=None):
    acc_val,acc_test = None,None;
    print("----------------------Logistic Regression------------------------------")
    t0= time.time();
    clf = LogisticRegression()
    clf.fit(train_x, train_y);
    if val_x:
        validation_pred = clf.predict(val_x);
        acc_val = getScores(val_y,validation_pred, "Validation Scores");
    if test_x:
        prediction = clf.predict(test_x);
        acc_test = getScores(test_y,prediction,"Test Scores");
    t1 = time.time();
    
    print('time = ' + str(t1-t0)+'s');
    print();
    return acc_val,acc_test


def multinominalNB(train_x,train_y,val_x=None,val_y=None,test_x=None,test_y=None):
    acc_val,acc_test = None,None;
    print("---------------Multinominal Naive Bayes------------");
    t0 = time.time();
    clf = MultinomialNB();
    clf.fit(train_x,train_y);
    if val_x:
        validation_pred = clf.predict(val_x);
        acc_val = getScores(val_y,validation_pred, "Validation Scores");
    if test_x:
        test_pred = clf.predict(test_x);
        acc_test = getScores(test_y, test_pred,"Test Scores");
    t1 = time.time();
    
    print('time = ' + str(t1-t0)+'s');
    print();
    return acc_val,acc_test


def svc(train_x,train_y,val_x=None, val_y=None,test_x=None,test_y=None):
    acc_val,acc_test = None,None;
    print("----------------------SVC------------------------------")
    t0= time.time();
    svctest2 = SVC(kernel='linear', random_state=42, C=1)
    svctest2.fit(train_x, train_y)
    if val_x:
        validation_pred = svctest2.predict(val_x);
        acc_val = getScores(val_y,validation_pred, "Validation Scores");
    if test_x:
        test_pred = svctest2.predict(test_x);
        acc_test = getScores(test_y, test_pred,"Test Scores");
    t1 = time.time();
    
    print('time = ' + str(t1-t0)+'s');
    print();
    return acc_val,acc_test
    

def rfc(train_x,train_y,val_x=None, val_y=None,test_x=None,test_y=None):
    acc_val,acc_test = None,None;
    #for i in range(1,11):
    print("---------------Random Forest Classifier---------")
    #print('n_estimator='+str(10*i));
    t0 = time.time();
    rcf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None)
        #rcf = RandomForestClassifier(random_state=None, n_estimators=100, max_depth=None)
    rcf.fit(train_x, train_y)
    if val_x:
        validation_pred = rcf.predict(val_x);
        acc_val = getScores(val_y,validation_pred, "Validation Scores")
    if test_x:
        test_pred = rcf.predict(test_x);
        acc_test = getScores(test_y, test_pred,"Test Scores");
    t1 = time.time();
        
    print('time = ' + str(t1-t0)+'s');
    print();
    return acc_val,acc_test


def adaboost(train_x,train_y,val_x=None, val_y=None,test_x=None,test_y=None):
    acc_val,acc_test = None,None;
    print("----------------------AdaBoosting------------------------------");
    t0= time.time();
    ada = AdaBoostClassifier(n_estimators=5); #base_estimator=DecisionTreeClassifier()
    ada.fit(train_x, train_y)
    if val_x:
        validation_pred = ada.predict(val_x);
        acc_val = getScores(val_y,validation_pred, "Validation Scores");
    if test_x:
        test_pred = ada.predict(test_x);
        acc_test = getScores(test_y, test_pred,"Test Scores");
    t1 = time.time();
    
    print('time = ' + str(t1-t0)+'s');
    print();
    return acc_val,acc_test


def ANN(train_x,train_y,val_x=None, val_y=None,test_x=None,test_y=None):
    acc_val,acc_test = None,None;
    print("----------------------ANN------------------------------");
    t0= time.time();
    ann = MLPClassifier(hidden_layer_sizes=(10,));
    ann.fit(train_x, train_y)
    if val_x:
        validation_pred = ann.predict(val_x);
        acc_val = getScores(val_y,validation_pred, "Validation Scores");
    if test_x:
        test_pred = ann.predict(test_x);
        acc_test = getScores(test_y, test_pred,"Test Scores");
    t1 = time.time();
    
    print('time = ' + str(t1-t0)+'s');
    print();
    return acc_val,acc_test


#%% get the data and run cross validation
direc = 'Enron Dataset/';
os.chdir('/Users/zhouhaiqing/Downloads/Email-Classification-Master');
files = os.listdir(direc);

test_data = [];
score = [];
for i in range(101,len(files)+99): #enron101-106, with one invisible cache folder
    print('-----------------------');
    print('CV: ROUND '+str(i-100));
    print();
    train_data = [];
    epoch_score = []
    for folder in files:
        if 'enron' in folder and not '106' in folder and not str(i) in folder: 
            ## use previous 5 for train/validation and the 6th for test
            folders = os.listdir(direc+folder);
            train_data.extend([direc+folder+'/'+email for email in folders]);
        elif str(i) in folder:
            folders = os.listdir(direc+folder);
            val_data = [direc+folder+'/'+email for email in folders];
        elif '106' in folder and len(test_data) == 0: ## test data not formed yet
            folders = os.listdir(direc+folder);    
            test_data = [direc+folder+'/'+email for email in folders];
    del folders;
    
    dictionary = make_dict(train_data);
    
    print('dictionary successfully built.');
    print();
    train_x, train_y = make_dataset(dictionary,train_data);
    val_x, val_y = make_dataset(dictionary,val_data);

    epoch_score.append(logisticRegression(train_x,train_y,val_x,val_y));
    epoch_score.append(multinominalNB(train_x, train_y,val_x,val_y));
    epoch_score.append(svc(train_x,train_y,val_x, val_y)); ## very time-consuming?
    epoch_score.append(rfc(train_x, train_y, val_x, val_y));
    epoch_score.append(adaboost(train_x, train_y, val_x, val_y));
    epoch_score.append(ANN(train_x, train_y, val_x, val_y));
    score.append(epoch_score);
    print();

print(score);


#%% get average score over cross validation
score = np.array(score);
score_avg = [np.mean([score[i][j][0] for i in range(5)]) for j in range(6)];


#%% best from KFold: 101-104 train 105 validation 106 test
# so here we try 101-105 as train and 106 as test
direc = 'Enron Dataset/';
os.chdir('/Users/zhouhaiqing/Downloads/Email-Classification-Master');
files = os.listdir(direc);
test_data = [];
train_data = [];
for folder in files:
    if 'enron' in folder and not '106' in folder and not '105' in folder: 
        ## use previous 5 for train/validation and the 6th for test
        folders = os.listdir(direc+folder);
        train_data.extend([direc+folder+'/'+email for email in folders]);
    elif '105' in folder:
        folders = os.listdir(direc+folder);
        val_data = [direc+folder+'/'+email for email in folders];
    elif '106' in folder and len(test_data) == 0: ## test data not formed yet
        folders = os.listdir(direc+folder);    
        test_data = [direc+folder+'/'+email for email in folders];
del folders;
    
dictionary = make_dict(train_data);
    
print('dictionary successfully built.');
print();
train_x, train_y = make_dataset(dictionary,train_data);
val_x, val_y = make_dataset(dictionary,val_data);
test_x,test_y = make_dataset(dictionary,test_data);

#%% run all models (primitively)
stat_lr = logisticRegression(train_x,train_y,val_x,val_y,test_x,test_y);
stat_mnb = multinominalNB(train_x, train_y,val_x,val_y,test_x,test_y);
stat_svc = svc(train_x,train_y,val_x,val_y,test_x,test_y); ## very time-consuming?
stat_rfc = rfc(train_x, train_y,val_x,val_y,test_x,test_y);
stat_ada = adaboost(train_x, train_y,val_x,val_y,test_x,test_y);
stat_ann = ANN(train_x, train_y,val_x,val_y,test_x,test_y);
score_test = [stat_lr,stat_mnb,stat_svc,stat_rfc,stat_ada,stat_ann];
print(score_test);
print();


#%% tuning hyperparameters: Logistic Regression
## hyperparameters to be changed
solvers = ['newton-cg', 'lbfgs', 'liblinear']; #change solver may crush the computer
penalty = ['l2'];
c_values = [10, 1.0, 0.1, 0.01, 0.001];
lr_acc_stat = [];

## run for loops to test different hyperparameters
for i in range(len(solvers)):
    for j in range(len(c_values)):
        t0= time.time();
        log_reg = LogisticRegression(solver=solvers[i],C=c_values[j]);
        log_reg.fit(train_x,train_y);
        print("----------------------Logistic Regression------------------------------");
        print('Hyperparameters: solver = '+solvers[i]+', C = ' + str(c_values[j]));
        validation_pred = log_reg.predict(val_x);
        acc_val = getScores(val_y,validation_pred, "Validation Scores");
        test_pred = log_reg.predict(test_x);
        acc_test = getScores(test_y,test_pred,"Test Scores");

        t1 = time.time();
        print('time = ' + str(t1-t0)+'s');
        
        lr_acc_stat.append([solvers[i],c_values[j],acc_val,acc_test]);
        print('one loop done.');
        print();

#solver = 'lbfgs'; ## or 'newton-cg' score are the same, though lbfgs runs faster(12s) than newton method(36s)
#C = 0.1; ## C = 0.1 does show a better performance


#%% tuning hyperparameters: Naive Bayes
## hyperparameters to be changed
alphas = [0.5, 1, 1.5, 5];
fit_priors = [True,False];
acc_stat_nb = [];

## run for loops to test different hyperparameters
for i in range(len(alphas)):
    for j in range(len(fit_priors)):
        t0= time.time();
        mn_nb = MultinomialNB(alpha=alphas[i],fit_prior=fit_priors[j]);
        mn_nb.fit(train_x,train_y);
        print("-------------------Multinominal Naive Bayes----------------------------");
        print('Hyperparameters: alpha = '+ str(alphas[i]) +', fit_prior = ' + str(fit_priors[j]));
        validation_pred = mn_nb.predict(val_x);
        acc_val = getScores(val_y,validation_pred, "Validation Scores");
        test_pred = mn_nb.predict(test_x);
        acc_test = getScores(test_y,test_pred,"Test Scores");

        t1 = time.time();
        print('time = ' + str(t1-t0)+'s');
        
        acc_stat_nb.append([alphas[i],fit_priors[j],acc_val,acc_test]);
        print('one loop done.');
        print();


#%% tuning hyperparameters: SVM
c_values = [0.1, 1, 10];
gammas =['scale','auto'];
#kernels = ['linear','poly','rbf','sigmoid','precomputed']; too time consuming
acc_stat_svm = [];

for i in range(len(c_values)):
    for j in range(len(gammas)):
        t0= time.time();
        print("----------------------SVM----------------------------");
        svm = SVC(C=c_values[i],gamma=gammas[j]);
        svm.fit(train_x,train_y);
        print('Hyperparameters: C = '+ str(c_values[i]) +', gamma = ' + str(gammas[j]));
        validation_pred = svm.predict(val_x);
        acc_val = getScores(val_y,validation_pred, "Validation Scores");
        test_pred = svm.predict(test_x);
        acc_test = getScores(test_y,test_pred,"Test Scores");

        t1 = time.time();
        print('time = ' + str(t1-t0)+'s');

        acc_stat_svm.append([acc_val,acc_test,c_values[i],gammas[j]]);
        print('one loop done.');
        print();
        
#%% tuning hyperparameters: Random Forest
estimators = [100,400,700,1000];
criterions =['gini','entropy'];
#kernels = ['poly','rbf','sigmoid','precomputed']; too time consuming
acc_stat_rf = [];

for i in range(len(estimators)):
    for j in range(len(criterions)):
        t0= time.time();
        print("----------------------Random Forest----------------------------");
        rf = RandomForestClassifier(n_estimators=estimators[i],criterion=criterions[j]);
        rf.fit(train_x,train_y);
        print('Hyperparameters: n_estimators = '+ str(estimators[i]) +', criterion = ' + str(criterions[j]));
        validation_pred = rf.predict(val_x);
        acc_val = getScores(val_y,validation_pred, "Validation Scores");
        test_pred = rf.predict(test_x);
        acc_test = getScores(test_y,test_pred,"Test Scores");

        t1 = time.time();
        print('time = ' + str(t1-t0)+'s');

        acc_stat_rf.append([acc_val,acc_test,estimators[i],criterions[j]]);
        print('one loop done.');
        print();


#%% tuning hyperparameters: ANN
hidden_layers = [(10,),(100,),(500,)];
solvers = ['sgd','adam'];
early_stop = True;
acc_stat_ann = [];

for i in range(len(hidden_layers)):
    for j in range(len(solvers)):
        t0= time.time();
        print("----------------------ANN----------------------------");
        ann = MLPClassifier(hidden_layer_sizes=hidden_layers[i],solver=solvers[j],early_stopping=early_stop);
        ann.fit(train_x,train_y);
        print('Hyperparameters: hidden_layers = '+ str(hidden_layers[i]) +', solver = ' + solvers[j]+', early_stopping = True');
        validation_pred = ann.predict(val_x);
        acc_val = getScores(val_y,validation_pred, "Validation Scores");
        test_pred = ann.predict(test_x);
        acc_test = getScores(test_y,test_pred,"Test Scores");

        t1 = time.time();
        print('time = ' + str(t1-t0)+'s');

        acc_stat_ann.append([acc_val,acc_test,hidden_layers[i],solvers[j]]);
        print('one loop done.');
        print();

#%% store the results data into files
## did this part manually...


#%% Try tf-idf version ...