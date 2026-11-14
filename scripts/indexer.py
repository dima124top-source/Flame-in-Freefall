import os
import re
import shutil
from pathlib import Path
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore, EncoderBackedStore
from langchain_classic.retrievers import ParentDocumentRetriever
import pickle

# Конфигурация путей
PDF_DIR = Path("./data/pdf")
CHROMA_DIR = Path("./storage/chroma_db")
DOCSTORE_DIR = Path("./storage/docstore")
COLLECTION_NAME = "nasa_psi_microgravity_fire"

# Словари регулярных выражений для строгого булевого тегирования
MATERIAL_PATTERNS: Dict[str, str] = {
    "has_pmma": r"\b(pmma|polymethyl\s*methacrylate|plexiglas|cast\s*acrylic)\b",
    "has_nomex": r"\b(nomex|meta-aramid|poly\(m-phenylene\s*isophthalamide\))\b",
    "has_cotton": r"\b(cotton|cellulose\s*fabric|sibtex|cotton-fiberglass)\b",
    "has_ptfe": r"\b(ptfe|polytetrafluoroethylene|teflon)\b",
    "has_pvc": r"\b(pvc|polyvinyl\s*chloride)\b",
}

FACILITY_PATTERNS: Dict[str, str] = {
    "has_bass_ii": r"\b(bass-ii|bass\s*2|burning\s*and\s*suppression\s*of\s*solids)\b",
    "has_saffire": r"\b(saffire|spacecraft\s*fire\s*experiment)\b",
    "has_acme": r"\b(acme|advanced\s*combustion\s*via\s*microgravity\s*experiments)\b",
    "has_flex_2": r"\b(flex-2|flex\s*2|flame\s*extinguishment\s*experiment)\b",
    "has_sofie": r"\b(sofie|solid\s*fuel\s*ignition\s*and\s*extinction)\b",
}


def extract_boolean_metadata(text: str, filename: str) -> Dict[str, bool | str | int]:
    """Анализирует текст страницы и имя файла, проставляя булевы флаги материалов и установок."""
    combined_text = f"{filename} {text}".lower()
    meta: Dict[str, bool | str | int] = {"source_file": filename}

    for key, pattern in MATERIAL_PATTERNS.items():
        meta[key] = bool(re.search(pattern, combined_text, flags=re.IGNORECASE))

    for key, pattern in FACILITY_PATTERNS.items():
        meta[key] = bool(re.search(pattern, combined_text, flags=re.IGNORECASE))

    return meta


def get_cpu_embeddings() -> HuggingFaceEmbeddings:
    """Инициализирует мультиязычную модель эмбеддингов строго на CPU для экономии VRAM."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def get_parent_retriever() -> ParentDocumentRetriever:
    """Создает или подключает существующий ParentDocumentRetriever."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    DOCSTORE_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = get_cpu_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    fs = LocalFileStore(str(DOCSTORE_DIR))
    store = EncoderBackedStore(
        store=fs,
        key_encoder=lambda k: k,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads,
    )

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 6},
    )


def load_and_tag_pdfs(pdf_folder: Path) -> List[Document]:
    """Загружает PDF-файлы, очищает текст и обогащает страницы метаданными."""
    if not pdf_folder.exists():
        pdf_folder.mkdir(parents=True, exist_ok=True)
        return []

    pdf_files = list(pdf_folder.glob("*.pdf"))
    all_docs: List[Document] = []

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            cleaned_text = re.sub(r"\s+", " ", page.page_content).strip()
            if len(cleaned_text) < 50:
                continue

            tagged_meta = extract_boolean_metadata(cleaned_text, pdf_path.name)
            tagged_meta["page"] = page.metadata.get("page", 0) + 1

            all_docs.append(Document(page_content=cleaned_text, metadata=tagged_meta))

    return all_docs


def build_index(reset: bool = True) -> int:
    """Полный цикл индексации PDF-отчетов NASA PSI."""
    if reset:
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
        if DOCSTORE_DIR.exists():
            shutil.rmtree(DOCSTORE_DIR)

    documents = load_and_tag_pdfs(PDF_DIR)
    if not documents:
        print(f"[ВНИМАНИЕ] PDF-файлы не найдены в директории {PDF_DIR.resolve()}.")
        return 0

    retriever = get_parent_retriever()
    retriever.add_documents(documents)
    print(f"[УСПЕХ] Проиндексировано страниц NASA PSI: {len(documents)}")
    return len(documents)


if __name__ == "__main__":
    build_index(reset=True)