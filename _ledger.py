import json
with open('evaluation/eval_v3_result.json','r',encoding='utf-8') as f:
    data = json.load(f)
for q in data:
    sr = q['strict_recall_at_10']
    if sr < 1.0:
        gold = q['gold_chunk_ids']
        top10 = q['pre_expansion_top10_ids']
        hit = [g for g in gold if g in top10]
        miss = [g for g in gold if g not in top10]
        print(f"{q['qid']} | {q['retrieval_mode']:20s} | R@10={sr:.0%} | gold={len(gold)} hit={len(hit)} miss={len(miss)}")
        for m in miss:
            print(f"    MISS: {m}")
        post = q.get('post_expansion_ids',[])
        for m in miss:
            if m in post:
                rank = post.index(m)+1
                print(f"    -> in post_expansion at rank {rank}")
            else:
                print(f"    -> NOT in post_expansion either")
