import pandas as pd
import numpy as np

def analyze_data_distributions():
    # Load data
    train = pd.read_csv('cleaned_data/labels_train.csv')
    val = pd.read_csv('cleaned_data/labels_validation.csv')
    test = pd.read_csv('cleaned_data/labels_test.csv')

    # Analyze Y-coordinate distribution
    print('\nY-coordinate Distribution:')
    print('\nTraining Data:')
    y_train_dist = (train['y'] > 5.0).value_counts(normalize=True)
    print(y_train_dist)

    print('\nValidation Data:')
    y_val_dist = (val['y'] > 5.0).value_counts(normalize=True)
    print(y_val_dist)

    print('\nTest Data:')
    y_test_dist = (test['y'] > 5.0).value_counts(normalize=True)
    print(y_test_dist)

    # Analyze X-coordinate statistics
    print('\nX-coordinate Statistics:')
    print('\nTraining Data:')
    print(train['x'].describe())
    print('\nValidation Data:')
    print(val['x'].describe())

    # Analyze room distribution
    print('\nRoom Distribution:')
    print('\nTraining Data:')
    print(train['room'].value_counts())
    print('\nValidation Data:')
    print(val['room'].value_counts())

if __name__ == "__main__":
    analyze_data_distributions()
