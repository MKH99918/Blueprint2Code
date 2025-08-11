import logging
import sys

import traceback



class BaseModel():
    def __init__(self, **kwargs):
        pass

    @abstractmethod

    def prompt(self, processed_input):
        pass

