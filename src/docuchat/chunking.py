from docuchat.config import settings
from docuchat.models import Chunk


def split_words(text:str , size:int , overlap:int)->list[str]:
    words=text.split()
    step=size-overlap
    pieces=[]
    for start in range(0,len(words), step):
        pieces.append(" ".join(words[start:start+size]))
        if start+size>=len(words):
            break
    return pieces


def chunk_pages(pages:list[int, str] , source:str , size:int|None= None , overlap:int |None=None) -> list[Chunk]:
    size=size or settings.chunk_size
    overlap= settings.chunk_overlap if overlap is None else overlap
    chunks=[]
    for page_number , text in pages:
        for piece in split_words(text , size , overlap):
            chunks.append(Chunk(id=len(chunks), text=piece, source=source, page=page_number))
    return chunks

