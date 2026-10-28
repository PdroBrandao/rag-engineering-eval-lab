import logging

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from openai import OpenAI

from retrieval_lab import __version__
from retrieval_lab.config import MAX_UPLOAD_BYTES, load_settings
from retrieval_lab.db import SupabaseDocumentStore, create_supabase
from retrieval_lab.embeddings import OpenAIEmbedder
from retrieval_lab.errors import LabError
from retrieval_lab.ingest import ingest_pdf
from retrieval_lab.retrieval import query_retrieval
from retrieval_lab.schemas import HealthResponse, IngestResponse, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Retrieval Lab",
        version=__version__,
        description=(
            "Ingest PDFs and compare vector, lexical, and hybrid retrieval. "
            "The API does not generate answers. Vector similarity, lexical rank, "
            "and RRF scores use different scales and are not directly comparable. "
            "The server is intended to run on 127.0.0.1."
        ),
    )
    app.state.store = None
    app.state.embedder = None
    app.state.settings = None
    app.state.max_upload_bytes = MAX_UPLOAD_BYTES

    def settings():
        if app.state.settings is None:
            app.state.settings = load_settings()
        return app.state.settings

    def store():
        if app.state.store is None:
            app.state.store = SupabaseDocumentStore(create_supabase(settings()))
        return app.state.store

    def embedder():
        if app.state.embedder is None:
            current = settings()
            app.state.embedder = OpenAIEmbedder(
                OpenAI(api_key=current.openai_api_key),
                current.embedding_model,
            )
        return app.state.embedder

    @app.exception_handler(LabError)
    async def handle_lab_error(_request, exc: LabError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.get("/health", response_model=HealthResponse)
    def health():
        try:
            store().ping()
        except Exception:
            logger.error("health check failed")
            return JSONResponse(
                status_code=503,
                content={"status": "ok", "database": "error"},
            )
        return HealthResponse(status="ok", database="ok")

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(
        file: UploadFile = File(...),
        chunk_size: int = Form(500),
        chunk_overlap: int = Form(50),
        replace: bool = Form(False),
    ):
        data = await file.read()
        current = app.state.settings
        model = current.embedding_model if current is not None else "text-embedding-3-small"
        result = ingest_pdf(
            data,
            file.filename,
            store(),
            embedder(),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            replace=replace,
            embedding_model=model,
            max_upload_bytes=app.state.max_upload_bytes,
        )
        return IngestResponse(**result)

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest):
        result = query_retrieval(
            request.question,
            store(),
            embedder(),
            top_k=request.top_k,
            rrf_k=request.rrf_k,
            candidate_k=request.candidate_k,
        )
        return QueryResponse(**result)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
