"""
parse_word_revisions.py — 提取Word修订记录，存入style_db
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import zipfile
import xml.etree.ElementTree as ET
import json
import sys
import os
import chromadb
from chromadb.config import Settings

NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


def extract_revisions(docx_path):
    """提取所有修订对（删除→插入）和批注"""
    revisions = []

    with zipfile.ZipFile(docx_path, 'r') as z:
        # 读批注
        comments = {}
        if 'word/comments.xml' in z.namelist():
            croot = ET.fromstring(z.read('word/comments.xml'))
            for c in croot.findall('.//w:comment', NS):
                cid = c.get(W + 'id')
                author = c.get(W + 'author', '')
                text = ''.join(t.text or '' for t in c.findall('.//w:t', NS))
                comments[cid] = {"text": text, "author": author}

        doc_root = ET.fromstring(z.read('word/document.xml'))

        # 遍历每个段落
        for para in doc_root.findall('.//w:p', NS):
            # 段落完整文本
            para_text = ''.join(t.text or '' for t in para.findall('.//w:t', NS))

            # 段落里的批注引用
            para_comment_ids = []
            for cref in para.findall('.//w:commentReference', NS):
                cid = cref.get(W + 'id')
                para_comment_ids.append(cid)

            # 删除的文本
            del_text = ''
            for dt in para.findall('.//w:del//w:delText', NS):
                if dt.text:
                    del_text += dt.text

            # 插入的文本
            ins_text = ''
            for it in para.findall('.//w:ins//w:t', NS):
                if it.text:
                    ins_text += it.text

            if del_text.strip() or ins_text.strip():
                comment_texts = [comments[cid]["text"] for cid in para_comment_ids if cid in comments]
                revisions.append({
                    "before": del_text.strip(),
                    "after": ins_text.strip(),
                    "context": para_text[:150],
                    "comment": " | ".join(comment_texts)
                })

        # 单独提取纯批注（没有对应修订的）
        mentioned_cids = set()
        for para in doc_root.findall('.//w:p', NS):
            for cref in para.findall('.//w:commentReference', NS):
                mentioned_cids.add(cref.get(W + 'id'))

        orphan_comments = []
        for cid, cdata in comments.items():
            if cid not in mentioned_cids:
                orphan_comments.append({
                    "before": "",
                    "after": "",
                    "context": "",
                    "comment": cdata["text"]
                })

    return revisions, orphan_comments


def build_style_db(revisions, orphan_comments):
    """把修改对存入ChromaDB"""
    client = chromadb.PersistentClient(path="data/chroma_db")

    # 删除旧集合（如果存在）
    try:
        client.delete_collection("style_db")
    except:
        pass

    collection = client.create_collection("style_db", metadata={"description": "导师写作修改风格库"})

    docs = []
    metadatas = []
    ids = []

    for i, r in enumerate(revisions):
        # 文本内容：before→after + 批注
        text = f"原句: {r['before']}\n改句: {r['after']}\n批注: {r['comment']}"
        docs.append(text)
        metadatas.append({
            "type": "revision_pair",
            "before": r["before"][:100],
            "after": r["after"][:100],
            "comment": r["comment"][:200],
        })
        ids.append(f"style_{i}")

    for i, c in enumerate(orphan_comments):
        docs.append(f"导师批注: {c['comment']}")
        metadatas.append({
            "type": "comment_only",
            "comment": c["comment"][:200],
        })
        ids.append(f"comment_{i}")

    if docs:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
        embeddings = model.encode(docs).tolist()
        collection.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=embeddings)

    print(f"style_db: 存入 {len(revisions)} 个修改对 + {len(orphan_comments)} 条批注")
    return collection


if __name__ == "__main__":
    docx_path = sys.argv[1]
    revisions, orphans = extract_revisions(docx_path)
    print(f"修改对: {len(revisions)}")
    print(f"纯批注: {len(orphans)}")
    for r in revisions:
        print(f"\n  删除: {r['before'][:80]}")
        print(f"  插入: {r['after'][:80]}")
        if r['comment']:
            print(f"  批注: {r['comment'][:80]}")

    build_style_db(revisions, orphans)
