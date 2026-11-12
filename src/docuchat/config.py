from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_query_prefix: str = ""    # e5 models need "query: "
    embed_passage_prefix: str = ""  # e5 models need "passage: "
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    groq_model: str = "openai/gpt-oss-120b"
    groq_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GROQ_API_KEY_DOCU", "GROQ_API_KEY"),
    )

    chunk_size: int = 350
    chunk_overlap: int = 50

    retrieve_k: int = 10
    top_n: int = 3

    max_pages: int = 150
    use_gpu: bool = False

    @model_validator(mode="after")
    def check_values(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.top_n > self.retrieve_k:
            raise ValueError("top_n can't be larger than retrieve_k")
        return self


settings = Settings()