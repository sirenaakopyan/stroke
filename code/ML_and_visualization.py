import cleanup_ML
import datacleanup
import dataanalysis
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def fit_and_predit_stroke(ml_df: pd.DataFrame) -> list:
    '''
    create a logistic regression model to predict occurrence
    of stroke. Train the model and test it by splitting the
    dataframe into training and testing. Determine accuracy of model
    through a confusion matrix and accuracy.
    '''
    # features is the risk_row_corr
    features = ml_df.drop('stroke', axis=1)
    # label is stroke, which we want to predict
    labels = ml_df['stroke']
    # Breaks the data into 80% train and 20% test
    features_train, features_test,\
        labels_train, labels_test = train_test_split(features,
                                                     labels, test_size=0.2)
    logistic_model = LogisticRegression(solver='liblinear', random_state=101)
    # train model on training set
    logistic_model.fit(features_train, labels_train)
    predictions = logistic_model.predict(features_test)
    confusion_mat = confusion_matrix(labels_test, predictions)
    accuracy = logistic_model.score(features_test, labels_test)
    return confusion_mat, accuracy


def plot_confusion_matrix(cm: list) -> None:
    '''
    plot the results from the confusion matrix
    '''
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted No stroke',
                                           'Predicted Stroke'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual No Stroke',
                                           'Actual Stroke'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
    fig.savefig('ML_confusion_matrix.png',  bbox_inches='tight')


def main():
    risk_factor_data = datacleanup.create_risk_factor_df(
        'datasets/stroke_data_1.csv')
    normalized_df = dataanalysis.normalize_data(risk_factor_data)
    ml_data = cleanup_ML.risk_factor_df_ML(normalized_df)
    confusion_matrix = fit_and_predit_stroke(ml_data)
    print(confusion_matrix)
    print(f"True Negatives: {((((confusion_matrix[0])[0]))[0])},\
          False Positive: {((((confusion_matrix[0])[0]))[1])},\
          False Negatives: {((((confusion_matrix[0])[1]))[0])},\
          True Postives: {((((confusion_matrix[0])[1]))[1])}")
    print(f'Accuracy of predicting no stroke:{confusion_matrix[1]}')
    plot_confusion_matrix(confusion_matrix[0])


if __name__ == '__main__':
    main()
