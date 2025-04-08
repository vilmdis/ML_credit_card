# ML_credit_cart
Credit Card Fraud Detection Project

Цель проекта:
Построить модель машинного обучения для выявления мошеннических транзакций по данным реальных операций с кредитными картами. Особенность задачи — сильный дисбаланс классов (менее 0.2% мошенничеств).

Использованные технологии:
Python (pandas, numpy, matplotlib, seaborn)
Scikit-learn (Decision Tree, Random Forest, метрики)
XGBoost
GridSearchCV
Визуализация: ROC-кривая, Precision-Recall curve

Этапы работы:
Загрузка и анализ данных
Данные с Kaggle: 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud'
Признаки уже предварительно обработаны (PCA), кроме Time и Amount.

Масштабирование данных
Использован StandardScaler для признаков Time и Amount.

Разделение выборки с учетом дисбаланса
Использован train_test_split(stratify=y) для сохранения пропорций классов.

Базовые модели
DecisionTreeClassifier (class_weight='balanced')
RandomForestClassifier (GridSearch + сравнение параметров)
XGBoost
scale_pos_weight подобран вручную и через GridSearch
Модель показала лучшие результаты по сравнению с деревьями

Оценка моделей
Использованы метрики:
precision, recall, f1-score
confusion matrix
ROC AUC, Precision-Recall AUC

Результаты XGBoost (лучшей модели):
Precision (мошенничество): 0.91
Recall (мошенничество): 0.87
F1-score: 0.89
Accuracy: 0.999
ROC AUC: ~0.99

![image](https://github.com/user-attachments/assets/5ce9e2ad-1e8f-436b-882a-c2e82144d45b)

![image](https://github.com/user-attachments/assets/890ee8b1-e6a3-43ac-a4ab-1ffa292543a2)

Ложноположительных: 8
Ложноприцательных: 13

![image](https://github.com/user-attachments/assets/179026ae-22a7-4779-b19f-468a0f7da159)


Выводы:
Использование scale_pos_weight и GridSearchCV позволило значительно повысить точность модели на несбалансированных данных.
XGBoost дал лучший компромисс между recall и precision.
Проект может служить примером решения задач с дисбалансом и высокими требованиями к точности предсказаний.
