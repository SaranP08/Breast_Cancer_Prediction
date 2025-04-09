import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pylab as plt
import seaborn as sns
df = pd.read_csv("CAT 1 Breast cancer dataset.csv")

x = df.drop(columns=["id", "diagnosis"]) 

print(df.head())

print(df.info())

label_enc = LabelEncoder()


df['diagnosis'] = label_enc.fit_transform(df['diagnosis'])

preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  
    ('scaler', StandardScaler())                  
])

y = df['diagnosis']

df.hist(figsize=(12, 8))
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', square=True)
plt.show()

x_processed = preprocessing_pipeline.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_processed, y, test_size = 0.2, random_state = 0)

rf_model = RandomForestClassifier(random_state = 0)

rf_model.fit(x_train, y_train)

rf_predictions = rf_model.predict(x_test)

cm = confusion_matrix(y_test, rf_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.show()

print("random forest accuracy:", accuracy_score(rf_predictions, y_test))
print("random forest f1_score:", f1_score(rf_predictions, y_test))
print("random forest roc_auc_score:", roc_auc_score(rf_predictions, y_test))
print("random forest recall_score:", recall_score(rf_predictions, y_test))

print("Random Foreset Confusion Matrix:")

param_grid = {
    'n_estimators' : [10,50,100],
    'max_depth' : [10,20,None],
    'min_samples_split' : [2,4,5,6],
    'min_samples_leaf' : [1,2,3,4,5],
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_
print()
print("Best Parameters", best_params)
print()
best_random_forest = RandomForestClassifier(**best_params, random_state=0)
best_random_forest.fit(x_train, y_train)

best_random_forest_pred = best_random_forest.predict(x_test)
accuracy_best_rf = accuracy_score(best_random_forest_pred, y_test)
print("Accuracy of best random forest:", accuracy_best_rf)

joblib.dump(best_random_forest, "breast_cancer_model.pkl")

cm = confusion_matrix(y_test, best_random_forest_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Random Forest")
plt.show()

bagging_clf = BaggingClassifier(estimator = rf_model, n_estimators=100, random_state=42)
bagging_clf.fit(x_train, y_train)

y_pred_bagging = bagging_clf.predict(x_test)

print("Bagging Classifier Results")
print("Accuracy:", accuracy_score(y_test, y_pred_bagging))
print("f1_score:", f1_score(rf_predictions, y_test))
print("roc_auc_score:", roc_auc_score(y_pred_bagging, y_test))
print("recall_score:", recall_score(y_pred_bagging, y_test))

print(classification_report(y_test, y_pred_bagging))

plt.show()

def get_user_input():
    user_input = {}
    for column in x.columns:
        value = float(input(f"Enter the value for {column}: "))
        user_input[column] = [value]
    return pd.DataFrame(user_input)

user_input_df = get_user_input()
user_input_processed = preprocessing_pipeline.transform(user_input_df)
user_prediction = best_random_forest.predict(user_input_processed)

diagnosis = 'Malignant' if user_prediction[0] == 1 else 'Benign'
print(f"The predicted diagnosis is: {diagnosis}")

feature_to_plot = 'mean_radius'

plt.figure(figsize=(10, 6))
sns.boxplot(y=df[feature_to_plot])  # Distribution of the feature in the dataset
plt.scatter(1, user_input_df[feature_to_plot], color='red', label='User Input')  # Overlay user's input
plt.title(f"User Input vs Dataset Distribution for {feature_to_plot}")
plt.legend()
plt.show()