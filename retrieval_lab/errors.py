class LabError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class IngestError(LabError):
    pass


class RetrievalError(LabError):
    pass


class EmbeddingError(LabError):
    def __init__(self, message: str = "Embedding request failed.") -> None:
        super().__init__(502, message)


class DatabaseError(LabError):
    def __init__(self, message: str = "Database request failed.") -> None:
        super().__init__(503, message)
