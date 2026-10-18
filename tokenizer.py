class CharacterTokenizer():
    def __init__(self, s: str) -> None:
        chars: list[str] = sorted(set(s))
        self.vocab_size: int = len(chars)
        
        self.forward_mapping: dict = {ch: i for i, ch in enumerate(chars)}
        self.backward_mapping: dict = {i: ch for i, ch in enumerate(chars)}


    def encode(self, s: str) -> list[int]:
        return [self.forward_mapping[ch] for ch in s]


    def decode(self, tokens: list[int]) -> str:
        return "".join([self.backward_mapping[t] for t in tokens])
