import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin  # 100% unnecessary!!
from sklearn import manifold
from matplotlib import pyplot as plt
from enum import Enum
from warnings import warn

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Models(Enum):
    KMeans = 0,
    SVC = 1,
    RandomForest = 2,
    NN = 3


class CleanData:
    data = None

    def __init__(self, file_name):

        self.data = pd.read_csv(file_name)  # Pandas DataFrame

        self.data['Diagnostic'] = self.data['Diagnostic'].replace({'M': 0, 'B': 1})

    def print(self):

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

    def plot(self):

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111)

        model = manifold.TSNE(n_components=5, init='pca')
        # model = manifold.MDS(n_components=5)
        # model = manifold.Isomap(n_components=5)
        # model = manifold.SpectralEmbedding(n_components=self.X_train.shape[1], n_neighbors=5)

        reduced_data = model.fit_transform(self.X_train)

        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=self.y_train*150, cmap='spectral', s=200)
        ax.set_title("Train data")

        plt.show()


class CancerClassificationS(CancerClassification):

    def __init__(self, clean_data_object: CleanData,  training_columns, results_columns, method_: Models):
        """
        Constructor
        :param file_name: file name
        :param method: method to use (string)
        """

        self.data = clean_data_object.data
        self.x_cols = training_columns
        self.y_cols = results_columns
        self.__trained__ = False

        X = self.data[self.x_cols].values
        y = self.data[self.y_cols].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.05, random_state=42)

        if method_ is Models.KMeans:
            self.method = KNeighborsClassifier(5)
        elif method_ is Models.SVC:
            # 'poly', 'rbf', 'sigmoid'
            self.method = SVC(C=1.0, kernel='poly', degree=1, gamma='auto', coef0=0.0, shrinking=True,
                              probability=False, tol=0.001, cache_size=200, class_weight=None,
                              verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
        elif method_ is Models.RandomForest:
            self.method = RandomForestClassifier()
        elif method_ is Models.NN:
            # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
            self.method = MLPClassifier(hidden_layer_sizes=(100, 100, 50), activation='logistic', solver='adam',
                                        alpha=0.0001, batch_size='auto', learning_rate='constant',
                                        learning_rate_init=0.001, power_t=0.5, max_iter=400, shuffle=True,
                                        random_state=None, tol=1e-5, verbose=False, warm_start=False,
                                        momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                                        validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        else:
            warn('WTF!?')


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

    # classification_object = CancerClassification(clean_data_object=clean_data_obj, method_=knn,
    #                                              training_columns=a, results_columns=b)
    # classification_object.plot()

    method_list = [Models.RandomForest, Models.SVC, Models.KMeans, Models.NN]

    for _method in method_list:

        classification_object = CancerClassificationS(clean_data_object=clean_data_obj, method_=_method,
                                                      training_columns=a, results_columns=b)

        # clasification_object.train()

        s = classification_object.score()
        print(_method, 'score', s)