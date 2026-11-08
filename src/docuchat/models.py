from pydantic import BaseModel, ConfigDict ,Field



class Chunk(BaseModel):
    model_config=ConfigDict(frozen= True)
    id: int
    text:str
    source:str
    page:int =Field(ge=1, description="Page number must be greater than or equal to 1")


class ScoredChunk(BaseModel):
    """A chunk with the score retrieval or reranking gave it."""
    chunk:Chunk
    score:float


class Citation(BaseModel):
    source:str
    page:int


class Answer(BaseModel):
    text:str
    citations:list[Citation]=[]
    found:bool=True




