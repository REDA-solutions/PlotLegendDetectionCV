from abc import abstractmethod


class EvalInterface:
    """
    Abstract class as interface for algorithms/models for easy evaluation.
    """
    
    @abstractmethod
    def __init__(self, preprocessor, **args) -> None:
        pass

    @abstractmethod
    def predict(self, img) -> list:
        """ Returns the detected (legend) texts in a list of strings. """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """ Returns the name of the model. """
        pass