import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin  # 100% unnecessary!!


class CleanData:
    data = None

    def __init__(self, file_name):
        """
        constructor to transform 'Diagnostic'column to numerical values
        :param file_name: name of the file
        """

        self.data = pd.read_csv(file_name)  # Pandas DataFrame

        self.data['Diagnostic'] = self.data['Diagnostic'].replace({'M': 0, 'B': 1}) #Replaces 'M' and 'B' for 0 and 1 respectively

    def print(self):
        """
        Print function for sanity check
        :return: the first ten rows of the dataframe
        """
        print(self.data.head(10))


class CancerClassification:

    method = None
    data = None
    x_cols = None
    y_cols = None

    X_test = None
    y_test = None

    __trained__ = False

    def __init__(self, clean_data_object: CleanData, method_: ClassifierMixin, training_columns, results_columns):
        """
        Constructor
        :param file_name: file name
        :param method: method to use (Scikit Learn object)
        """

        self.method = method_
        self.data = clean_data_object.data
        self.x_cols = training_columns
        self.y_cols = results_columns
        self.__trained__ = False

        X = self.data[self.x_cols].values
        y = self.data[self.y_cols].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    def train(self):
        """
        Train the data set classifier
        :return:
        """

        self.method.fit(self.X_train, self.y_train)

        self.__trained__ = True

    def score(self):
        """
        Score our classification
        :return:
        """
        y_pred = self.classify(self.X_test)

        comparison = y_pred == self.y_test[:, 0]

        score = sum(comparison) / len(comparison)

        return score

    def classify(self, X):
        """
        Classify some input
        :param X:
        :return:
        """

        # check that we have trained our method!
        if self.__trained__:
            pass
        else:
            self.train()

        # if not self.__trained__:
        #     self.train()

        # classify
        y_pred = self.method.predict(X)

        return y_pred


class CancerClassificationS(CancerClassification):

    def __init__(self, file_name, method_):
        """
        Constructor
        :param file_name: file name
        :param method: method to use (string)
        """

        self.f_name = file_name
        self.method = method_
        self.data = pd.read_csv(file_name)

    def classify(self):
        print()


if __name__ == '__main__':

    f = 'Wisconsin_breast_cancer.csv'

    knn = KNeighborsClassifier(n_neighbors=6)

    # a = ['radius', 'radius_std', 'radius_worst', 'texture', 'texture_std',
    #      'texture_worst', 'perimeter', 'perimeter_std', 'perimete_worst',
    #      'area', 'area_std', 'area_worst', 'smoothness', 'smoothness_std',
    #      'smoothness_worst', 'compactness', 'compactness_std', 'compactness_worst',
    #      'concavity', 'concavit_std', 'concavit_worst', 'concave', 'concav_std',
    #      'concav_worst', 'symmetry', 'symmetry_std', 'symmetry_worst', 'fractal',
    #      'fractal_std', 'fractal_worst']

    a = ['radius',  'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave',
         'symmetry', 'fractal']

    b = ['Diagnostic']

    clean_data_obj = CleanData(f)
    clean_data_obj.print()

    classification_object = CancerClassification(clean_data_object=clean_data_obj, method_=knn,
                                                 training_columns=a, results_columns=b)

    # clasification_object.train()

    s = classification_object.score()
    print('score', s)