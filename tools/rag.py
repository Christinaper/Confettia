# tools/rag.py
# RAG Skill：Obsidian 笔记检索
#
# 依赖安装：
#   pip install chromadb sentence-transformers
#
# 首次运行会下载 bge-m3 模型（约 1.5GB），之后离线使用

import os
import re
import hashlib
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("rag")

# ── 配置（从 .env 读取）──────────────────────────────────────────────────────
VAULT_PATH   = os.getenv("OBSIDIAN_VAULT", "")          # Obsidian 库路径
CHROMA_PATH  = os.getenv("CHROMA_PATH", "./chroma_db")  # 向量库存储路径
EMBED_MODEL  = os.getenv("EMBED_MODEL", "BAAI/bge-m3")  # embedding 模型
COLLECTION   = "obsidian_notes"

# chunk 参数
CHUNK_MIN_CHARS = 50    # 低于此长度和上段合并
CHUNK_MAX_CHARS = 800   # 超过此长度按段落再切
TOP_K           = 3     # 检索返回最相关的 K 个 chunk


# ── 延迟初始化（避免启动时加载慢）───────────────────────────────────────────
_client     = None
_collection = None
_embedder   = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        log.info(f"加载 embedding 模型：{EMBED_MODEL}（首次加载约 10–30 秒）")
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
        log.info("Embedding 模型加载完成")
    return _embedder


def _get_collection():
    global _client, _collection
    if _collection is None:
        import chromadb
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},  # 余弦相似度
        )
        log.info(f"ChromaDB 已连接，当前 {_collection.count()} 个 chunks")
    return _collection


# ── Markdown 解析与 Chunking ─────────────────────────────────────────────────
def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """提取 YAML frontmatter，返回 (metadata_dict, 正文)。"""
    meta = {}
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            fm = content[3:end].strip()
            for line in fm.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip()
            content = content[end+3:].strip()
    return meta, content


def _clean_markdown(text: str) -> str:
    """清理不需要 embedding 的 Markdown 语法。"""
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)   # [[内链]] → 内链
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)         # 图片
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # [链接](url) → 链接
    text = re.sub(r"`{3}[\s\S]*?`{3}", "[代码块]", text)   # 代码块保留占位
    text = re.sub(r"\s{3,}", "\n\n", text)
    return text.strip()


def _chunk_by_heading(content: str, filename: str) -> list[dict]:
    """
    按 Markdown 二级标题切分，返回 chunk 列表。
    每个 chunk: {text, heading, position}
    """
    # 按 ## 标题分割
    sections = re.split(r"\n(?=#{1,3} )", content)
    chunks = []

    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        # 提取标题
        heading_match = re.match(r"^(#{1,3})\s+(.+)", section)
        if heading_match:
            heading = heading_match.group(2).strip()
            body    = section[heading_match.end():].strip()
        else:
            heading = filename
            body    = section

        body = _clean_markdown(body)
        if not body:
            continue

        # 太短：跳过（会被合并逻辑处理）
        if len(body) < CHUNK_MIN_CHARS:
            if chunks:
                chunks[-1]["text"] += "\n" + body
            continue

        # 太长：按段落再切
        if len(body) > CHUNK_MAX_CHARS:
            paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
            current = ""
            for para in paragraphs:
                if len(current) + len(para) > CHUNK_MAX_CHARS and current:
                    chunks.append({"text": current, "heading": heading, "position": i})
                    current = para
                else:
                    current = (current + "\n\n" + para).strip()
            if current:
                chunks.append({"text": current, "heading": heading, "position": i})
        else:
            # 在 chunk 开头加入标题，提升检索相关性
            full_text = f"[{heading}]\n{body}" if heading != filename else body
            chunks.append({"text": full_text, "heading": heading, "position": i})

    return chunks


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ── 索引构建 ──────────────────────────────────────────────────────────────────
def build_index(vault_path: str = None, force: bool = False) -> dict:
    """
    扫描 Obsidian 库，增量更新向量索引。
    返回统计信息 {added, updated, skipped, deleted}
    """
    vault = Path(vault_path or VAULT_PATH)
    if not vault.exists():
        raise FileNotFoundError(f"Obsidian 库路径不存在：{vault}")

    collection = _get_collection()
    embedder   = _get_embedder()

    # 获取已索引文件的 hash（存在 metadata 里）
    existing = collection.get(include=["metadatas"])
    indexed_hashes: dict[str, list[str]] = {}  # file_path → [chunk_id, ...]
    for i, meta in enumerate(existing["metadatas"]):
        fp = meta.get("file_path", "")
        if fp:
            indexed_hashes.setdefault(fp, []).append(existing["ids"][i])

    stats = {"added": 0, "updated": 0, "skipped": 0, "deleted": 0}
    current_files = set()

    md_files = list(vault.rglob("*.md"))
    log.info(f"扫描 Obsidian 库：{len(md_files)} 个 Markdown 文件")

    for md_file in md_files:
        rel_path = str(md_file.relative_to(vault))
        current_files.add(rel_path)

        file_hash = _file_hash(md_file)
        mtime     = datetime.fromtimestamp(
            md_file.stat().st_mtime, tz=timezone.utc
        ).isoformat()

        # 检查是否需要更新
        existing_ids = indexed_hashes.get(rel_path, [])
        if existing_ids and not force:
            # 对比第一个 chunk 的 hash
            first_meta = collection.get(ids=[existing_ids[0]], include=["metadatas"])
            if first_meta["metadatas"] and \
               first_meta["metadatas"][0].get("file_hash") == file_hash:
                stats["skipped"] += 1
                continue
            # hash 不同，删除旧 chunks
            collection.delete(ids=existing_ids)
            stats["updated"] += 1
        else:
            stats["added"] += 1

        # 解析文件
        content  = md_file.read_text(encoding="utf-8", errors="ignore")
        fm, body = _parse_frontmatter(content)
        tags     = fm.get("tags", fm.get("tag", ""))
        title    = md_file.stem
        folder   = str(md_file.parent.relative_to(vault))

        chunks = _chunk_by_heading(body, title)
        if not chunks:
            continue

        # 批量 embedding
        texts     = [c["text"] for c in chunks]
        embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()

        ids       = [f"{rel_path}::{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "file_path":  rel_path,
                "file_hash":  file_hash,
                "title":      title,
                "folder":     folder,
                "heading":    c["heading"],
                "tags":       tags,
                "mtime":      mtime,
                "position":   c["position"],
            }
            for c, _ in zip(chunks, embeddings)
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    # 清理已删除的文件
    for fp, ids in indexed_hashes.items():
        if fp not in current_files:
            collection.delete(ids=ids)
            stats["deleted"] += len(ids)
            log.info(f"已删除索引（文件不存在）：{fp}")

    log.info(
        f"索引完成：新增 {stats['added']}，"
        f"更新 {stats['updated']}，"
        f"跳过 {stats['skipped']}，"
        f"删除 {stats['deleted']}"
    )
    return stats


# ── 检索 ──────────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = TOP_K,
             folder_filter: Optional[str] = None) -> list[dict]:
    """
    检索最相关的 chunks。
    返回 [{text, title, file_path, heading, score}, ...]
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    embedder    = _get_embedder()
    query_vec   = embedder.encode([query], normalize_embeddings=True).tolist()

    where = {"folder": {"$contains": folder_filter}} if folder_filter else None

    results = collection.query(
        query_embeddings=query_vec,
        n_results=min(top_k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = 1 - dist  # cosine distance → similarity
        if score < 0.3:   # 低相关性过滤
            continue
        chunks.append({
            "text":      doc,
            "title":     meta.get("title", ""),
            "file_path": meta.get("file_path", ""),
            "heading":   meta.get("heading", ""),
            "tags":      meta.get("tags", ""),
            "score":     round(score, 3),
        })

    return chunks


def format_context(chunks: list[dict]) -> str:
    """将检索结果格式化为注入 prompt 的上下文。"""
    if not chunks:
        return ""
    parts = []
    for c in chunks:
        source = c["title"]
        if c["heading"] and c["heading"] != c["title"]:
            source += f" › {c['heading']}"
        parts.append(f"[来自笔记：{source}（相关度 {c['score']}）]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


# ── 异步包装（供 agent.py 调用）──────────────────────────────────────────────
async def async_retrieve(query: str, top_k: int = TOP_K) -> str:
    """异步检索，返回格式化后的上下文字符串。"""
    loop = asyncio.get_event_loop()
    chunks = await loop.run_in_executor(None, retrieve, query, top_k)
    if not chunks:
        return ""
    log.info(f"RAG 检索：'{query[:30]}' → {len(chunks)} 个相关 chunks")
    return format_context(chunks)


def is_rag_available() -> bool:
    """检查 RAG 是否已配置并可用。"""
    return bool(VAULT_PATH) and Path(VAULT_PATH).exists()