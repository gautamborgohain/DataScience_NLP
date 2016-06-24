import json
import pandas as pd
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report


variables = json.loads(open('/Volumes/Data/NLP/Utilities/variables.json').read())
path = variables['path']


# Grid Search for subjectivity classsifier
class clf_grid_searcher():

    def __init__(self,X,y,model_to_load):
        self.subjectivity_pipeline = pickle.load(open(path + model_to_load, mode='rb'))
        y = list(y)
        self.X, self.Xt, self.y, self.yt = train_test_split(X, y, test_size=0.33, random_state=1)


    def run(self, results_path , verbosity = 0):

        results = pd.DataFrame(columns=['Classifier', 'Desc', 'Accuracy', 'f1_Scores'])

        Classifier = []
        Desc = []
        Accuracy = []
        F1_Scores = []

        LSVC_parameters = [{'C': [0.01, 0.1, 0.3, 0.05, 0.08,2],
                            'class_weight':[{-1:1.001,0:1,1:1.001},{-1:1.005,0:1,1:1.005}, {-1:1.01,0:1,1:1.01},'balanced'],
                            'loss' : ['squared_hinge','hinge'],
                            'max_iter' : [3000]
                            }]

        # 6
        LR_parameters = [{'C': [0.1, 1, 0.05,2,2.5,3],
                          'class_weight':[{-1:1.1,0:1,1:1.1},{-1:1.15,0:1,1:1.15}, {-1:1.2,0:1,1:1.2},'balanced'],
                          'solver': ['liblinear']
                          }
            # ,
            #              {'C': [1, 2, 3],
            #               'solver': ['lbfgs', 'sag'],
            #               'max_iter': [1000]
            #               }
                         ]
        # 6+6
        RC_parameters = [{'alpha': [0.1, 1],
                          'tol': [0.001, 0.01],
                          'solver': ['lsqr', 'sparse_cg', 'sag']
                          }]
        # 12
        PAC_parameters = [{'C': [0.1, 1],
                           'n_iter': [30, 50, 100],
                           'loss': ['hinge', 'squared_hinge']

                           }]


        for clf, name in (
                (LinearSVC(verbose=verbosity), 'LinearSVC'),
                # (RidgeClassifier(), "Ridge Classifier"),
                # (PassiveAggressiveClassifier(), "Passive-Aggressive"),
                (LogisticRegression(verbose=verbosity), 'Logistic Regression')
        ):
            print('_' * 80)
            print(name)
            Classifier.append(name)
            if name == 'LinearSVC':
                params = LSVC_parameters
            elif name == 'Logistic Regression':
                params = LR_parameters
            elif name == 'Ridge Classifier':
                params = RC_parameters
            elif name == 'Passive-Aggressive':
                params = PAC_parameters

            grid_clf = GridSearchCV(clf, params, cv=10, verbose=3)

            grid_X = self.subjectivity_pipeline.transform(self.X)
            grid_Xt = self.subjectivity_pipeline.transform(self.Xt)

            grid_clf.fit(grid_X, self.y)
            print('Best params:')
            print(grid_clf.best_params_)
            clfs = grid_clf.best_estimator_
            pred = clfs.predict(grid_Xt)
            score = accuracy_score(self.yt, pred)
            f1score = f1_score(pred, self.yt, average='macro', labels=[1, -1])
            print("accuracy:")
            print(score)

            print("f1-score:")
            print(f1score)

            print("classification report:")
            print(classification_report(self.yt, pred))

            print("confusion matrix:")
            print(confusion_matrix(self.yt, pred))
            print('_' * 80)

            Desc.append(grid_clf.best_params_)
            Accuracy.append(score)
            F1_Scores.append(f1score)

        results['Classifier'] = Classifier
        results['Desc'] = Desc
        results['Accuracy'] = Accuracy
        results['f1_Scores'] = F1_Scores

        results.to_excel(results_path)


variables = json.loads(open('/Volumes/Data/NLP/Utilities/variables.json').read())
results_path =variables['Clf_grid_output']
modelspath = variables['path']

data = pd.read_pickle(modelspath + '/data.pck')
X = data.Sentences
y = data.Sentiment
model_to_load = '/polarity_classification_pipeline.pck'

cgs = clf_grid_searcher(data.Sentences, data.Sentiment, model_to_load)
cgs.run(results_path)
# self.y = list(map(lambda x: x if x != -1 else 1, self.y))
# self.yt = list(map(lambda x: x if x != -1 else 1, self.yt))
