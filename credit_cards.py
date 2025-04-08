import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier


df = pd.read_csv('../creditcard.csv')
# print(df.head())
# print(df.columns)
# print(df.info())
# print(df.describe())

X = df.drop(columns=['Class'], axis=1)
y = df['Class']

# scaler = StandardScaler()

# scaled_X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, stratify=y)

# model = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=6, min_samples_leaf=10)
# model.fit(X_train, y_train)

# params = {
#     'max_depth': [3, 5, 10],
#     'min_samples_leaf': [1, 5, 10],
#     'criterion': ['gini', 'entropy'],
#     'class_weight': ['balanced']
# }

# grid = GridSearchCV(DecisionTreeClassifier(), params, scoring='f1', cv=5)
# grid.fit(X_train, y_train)
# print(grid.best_params_)
# print(grid.best_score_)
# print(grid.best_estimator_)

# model = RandomForestClassifier(class_weight='balanced', max_depth=9, max_features=3)
# model.fit(X_train, y_train)

# param_grid = {'n_estimators':[100],
#               'max_features':[3],
#               'max_depth':[5,7,9]}
# grid = GridSearchCV(model, param_grid, verbose=2, scoring='f1', cv=3)
# grid.fit(X_train, y_train)
# print(grid.best_params_)
# print(grid.best_score_)
# print(grid.best_estimator_)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', learning_rate=0.1, n_estimators=200, max_depth=5, scale_pos_weight=100)
model.fit(X_train, y_train)

# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [3, 5],
#     'learning_rate': [0.05, 0.1],
#     'scale_pos_weight': [100, 300]
# }
# grid = GridSearchCV(model, param_grid, verbose=2, scoring='f1', cv=3, n_jobs=-1)
# grid.fit(X_train, y_train)
# print(grid.best_params_)
# print(grid.best_score_)
# print(grid.best_estimator_)

pred = model.predict(X_test)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

# RocCurveDisplay.from_estimator(model, X_test, y_test)
# PrecisionRecallDisplay.from_estimator(model,X_test,y_test)

cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.show()
