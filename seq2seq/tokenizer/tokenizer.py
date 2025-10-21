from typing import List, Dict, Tuple, Optional


class Tokenizer:
    """
    Tokenizer class that tokenizes the input text.

    You will need TWO functions in this class. You
    do not need to have a body here, rather the body
    will be added in the CharacterTokenizer subclass.

    This class is simply a template that could be
    used in the future for other tokenizers.
    """

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError
