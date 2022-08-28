def getScore(x_test,y_test,x_train,y_train):
  lst1=[]
  models = []
  models.append(['Logistic Regreesion', LogisticRegression(random_state=0)])
  models.append(['SVM', SVC(random_state=0)])
  models.append(['KNeighbors', KNeighborsClassifier()])
  models.append(['GaussianNB', GaussianNB()])
  models.append(['BernoulliNB', BernoulliNB()])
  models.append(['Decision Tree', DecisionTreeClassifier(random_state=0)])
  models.append(['Random Forest', RandomForestClassifier(random_state=0)])
  models.append(['XGBoost', XGBClassifier(eval_metric= 'error')])
  for m in range(len(models)):
    lst_2= []
    model = models[m][1]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)  #Confusion Matrix
    accuracies = cross_val_score(estimator = model, X = x_train, y = y_train, cv = 10)   #K-Fold Validation
    roc = roc_auc_score(y_test, y_pred)  #ROC AUC Score
    precision = precision_score(y_test, y_pred)  #Precision Score
    recall = recall_score(y_test, y_pred)  #Recall Score
    f1 = f1_score(y_test, y_pred)  #F1 Score
    lst_2.append(models[m][0])
    lst_2.append((accuracy_score(y_test, y_pred))*100) 
    lst_2.append(accuracies.mean()*100)
    lst_2.append(accuracies.std()*100)
    lst_2.append(roc)
    lst_2.append(precision)
    lst_2.append(recall)
    lst_2.append(f1)
    lst_1.append(lst_2)
  df = pd.DataFrame(lst_1, columns= ['Model', 'Accuracy', 'K-Fold Mean Accuracy', 'Std. Deviation', 'ROC AUC', 'Precision', 'Recall', 'F1'])
  df.sort_values(by= ['Accuracy', 'K-Fold Mean Accuracy'], inplace= True, ascending= False)
  return df