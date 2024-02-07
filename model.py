from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class SpamDetector:
    def __init__(self):
        self.vec = TfidfVectorizer()
        self.model = SVC()
    
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
        y_ = self.model.predict(X_vec)

        return y_
    
    def save(self, file_name="spam-det-model"):
        with open(f"{file_name}.pkl", 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load(file_name="spa,-det-model")->SpamDetector:
        with open(f'{file_name}.pkl', 'rb') as f:
            model = pickle.load(f)
            return model