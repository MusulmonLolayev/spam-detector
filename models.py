from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import dill
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class SpamDetector:
    def __init__(self):
        self.vec = TfidfVectorizer()
        self.model = SVC(probability=True)
    
    def fit(self, X, y):
        # matn modelini o'rgatish
        self.vec.fit(X)
        # o'rgatuvchi to'plamni vektor ko'rnishiga o'tkazish
        X_vec = self.vec.transform(X)
        # asosiy modelni o'rgatish
        self.model.fit(X_vec, y)
    
    def predict(self, X):
        # o'rgatuvchi to'plamni vektor ko'rnishiga o'tkazish
        X_vec = self.vec.transform(X)
        # bashorat qilish
        y_ = self.model.predict_proba(X_vec)

        return y_
    
    def save(self, file_name="spam-det-model"):
        with open(f"{file_name}.pkl", 'wb') as f:
            dill.dump(self, f)
    
    def load(file_name="spam-det-model"):
        with open(f'{file_name}.pkl', 'rb') as f:
            model = dill.load(f)
            return model
        
def __train_spam_detector_model():
    # https://archive.ics.uci.edu/dataset/228/sms+spam+collection
    df = pd.read_csv('./spam.csv', sep='\t')
    # set - to'plammodel
    # train o'rgatish
    X_train, X_test, y_train, y_test = train_test_split(df['text'], 
                                                        df['label'], 
                                                        test_size=0.2,
                                                        random_state=42)
    y_train = np.array([0 if label == 'ham' else 1 for label in y_train])
    y_test = np.array([0 if label == 'ham' else 1 for label in y_test])
    print("O'rgatuvchi to'plamda: ", len(X_train))
    print("Test to'plamda: ", len(X_test))

    detector = SpamDetector()
    detector.fit(X_train, y_train)
    detector.save()


if __name__ == '__main__':
    __train_spam_detector_model()    