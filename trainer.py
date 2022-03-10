from datetime import datetime
import json
import os
import pandas as pd
import pickle
from preprocessing import preprocessing_pipeline
from threading import Thread, get_ident
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


class Trainer():
    """Trainer class to train and store a logistic Regression model.
    
    Methods
    -------
    train(text: list[str], labels: list[str]) -> None
        train the model with the given data.

    predict(texts: list[str]) -> list[dict]:
        Predict the labels of the given data.

    get_status() -> dict:
        Get the status of the stored model (status, timestamp, classes, evaluation).
    """

    def __init__(self) -> None:
        self.__storage_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage')
        os.makedirs(self.__storage_path, exist_ok=True)

        self.__status_path = os.path.join(self.__storage_path, 'model_status.json')
        if os.path.exists(self.__status_path):
            with open(self.__status_path) as file:
                self.__model_status = json.load(file)
        else:
            self.__model_status = {"status": "No model found",
                                   "timestamp": datetime.now().isoformat(),
                                   "classes": [],
                                   "evaluation": {}}

        self.__model_path = os.path.join(self.__storage_path, 'model_pickle.pkl')
        if os.path.exists(self.__model_path):
            with open(self.__model_path, 'rb') as model_file:
                self.model = pickle.load(model_file)
        else: 
            self.model = None
        
        self._running_threads = []
        self._pipeline = None

    def _training_thread(self, x_train: list[str], x_test: list[str], y_train: list[str], y_test: list[str]):
        self._pipeline.fit(x_train, y_train)
        y_test_pred = self._pipeline.predict(x_test)
        
        report = classification_report(y_test, 
                                       y_test_pred, 
                                       output_dict=True)
        classes = self._pipeline.classes_.tolist()

        # update model status
        self._update_status("Model Ready", classes, report)
        
        self.model = self._pipeline
        self._pipeline = None

        with open(self.__model_path, "wb") as model_file:
            pickle.dump(self._pipeline, model_file)
        
        # When training is done, remove the thread from running-threads list
        thread_id = get_ident()
        for i, t in enumerate(self._running_threads):
            if t.ident == thread_id:
                self._running_threads.pop(i)
                break

    def train(self, text: list[str], labels: list[str]) -> None:
        """Train the model with the given data.

        Parameters
        ----------
        text: list[str]
            a List containing text data.

        labels: list[str]
            a list containing the corresponding text labels.
        """

        if len(self._running_threads):
            raise RuntimeError("A training process is already running.")

        x_train, x_test, y_train, y_test = train_test_split(text, 
                                                            labels, 
                                                            train_size=0.2)
        preprocess = preprocessing_pipeline(["normalization","tfidf"],
                                            victorizer_kwarg = dict(ngram_range=(1, 5), min_df=10))
        clf = LogisticRegression()
        self._pipeline = make_pipeline(preprocess, clf)

        # update model status
        self.model = None
        self._update_status("Training")

        t = Thread(target=self._training_thread,
                  args=(x_train, x_test, y_train, y_test))
        self._running_threads.append(t)
        t.start()

    def predict(self, texts: list[str]) -> list[dict]:
        """Predict the labels of the given data.

        Parameters
        ----------
        text: list[str]
            a List containing text data.

        Returns
        -------
        response: list[dict]
            a list containing each text and the corresponding probability of each label.
        """

        response = []
        if self.model:
            probs = self.model.predict_proba(texts)
            for i, row in enumerate(probs):
                row_pred = {}
                row_pred['text'] = texts[i]
                row_pred['predictions'] = {class_: round(float(prob), 3) 
                                           for class_, prob 
                                           in zip(self.__model_status['classes'], row)}
                response.append(row_pred)
        else:
            raise RuntimeError("No trained model was found.")
        return response

    def get_status(self) -> dict:
        """Get the status of the stored model (status, timestamp, classes, evaluation)."""
        return self.__model_status

    def _update_status(self, status: str, classes: list[str] = [], evaluation: dict = None) -> None:
        self.__model_status['status'] = status
        self.__model_status['timestamp'] = datetime.now().isoformat()
        self.__model_status['classes'] = classes
        self.__model_status['evaluation'] = evaluation if evaluation else {}

        with open(self.__status_path, 'w+') as file:
            json.dump(self.__model_status, file, indent=2)


if __name__ == "__main__":
    trainer = Trainer()
    df = pd.read_csv(r"Datasets/dialect_dataset.csv", header=0)

    trainer.train(df["text"], df["dialect"])
    print(trainer.get_status())