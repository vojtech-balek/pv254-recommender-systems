from abc import ABC, abstractmethod

class BaseRecommender(ABC):
    """
    Base class for recommender systems to enforce a common interface.
    """
    def __init__(self, top_n=10, **kwargs):
        self.top_n = top_n

    @staticmethod
    def _norm_book_id(x):
        """Normalize a book ID to a string."""
        return "" if x is None else str(x)

    @abstractmethod
    def fit(self, *args, **kwargs):
        """Train the recommender model or process the data."""
        pass

    @abstractmethod
    def recommend(self, user_id, *args, **kwargs):
        """Recommend items for a given user."""
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """Evaluate the recommender on a test set."""
        pass

