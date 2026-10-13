"""
写作助手多智能体工作流（LangGraph）
四节点：Retriever → Writer(VL) → De-duplication → Reviewer
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
import base64
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from openai import OpenAI
from config import (DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
                    SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, VL_MODEL_NAME)


class WritingState(TypedDict):
    user_request: str
    user_viewpoint: str         # 用户初步力学分析
    image_path: str
    template_text: str
    style_cases: str
    figure_features: str
    intent_mode: str            # 整体趋势/局部状态/参数对比/理论推演
    mechanism_chain: str
    draft: str
    final_text: str
    review_feedback: str
    iteration: int
    history: List[str]


class WritingAgent:
    def __init__(self, retriever=None):
        self.ds_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.vl_client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
        self.retriever = retriever
        self.max_iterations = 2
        self.graph = self._build_graph()

    def _retriever_node(self, state: WritingState) -> dict:
        """节点1：提取作者+章节，硬过滤ChromaDB"""
        print("  [节点1] 检索员：提取元数据，锁定模板章节...")

        # 用DeepSeek提取author+chapter
        resp = self.ds_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": f"""从用户写作需求中提取：
1. author：作者名（如"王鑫涛"、"雷杰鑫"）
2. chapter：章节号（如"4.2"、"第五章"）
3. task：写作任务类型

只输出JSON，如：{{"author":"王鑫涛","chapter":"4.2","task":"写作模板"}}

需求：{state['user_request']}"""}],
            temperature=0.1,
        )
        import json
        try:
            meta = json.loads(resp.choices[0].message.content)
        except:
            meta = {"author": "", "chapter": "", "task": ""}

        author = meta.get("author", "")
        chapter = meta.get("chapter", "")
        print(f"  提取到: author={author}, chapter={chapter}")

        # 硬过滤检索
        template_text = ""
        if self.retriever and author:
            try:
                # 在所有chunks里过滤
                all_chunks = self.retriever.all_chunks
                filtered = []
                for c in all_chunks:
                    paper_match = author in c.paper_name or author in c.section_path
                    section_match = chapter in c.section_path if chapter else True
                    if paper_match and section_match:
                        filtered.append(c)
                # 取前5个
                template_text = "\n\n".join([f"[{c.section_path}]\n{c.text[:400]}" for c in filtered[:5]])
                print(f"  锁定到{len(filtered)}个模板块")
            except Exception as e:
                print(f"  过滤检索失败: {e}")

        # 检索风格案例
        style_cases = ""
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            sc = chromadb.PersistentClient(path="data/chroma_db")
            style_col = sc.get_collection("style_db")
            model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
            q_emb = model.encode([state["user_request"]]).tolist()
            hits = style_col.query(query_embeddings=q_emb, n_results=3)
            style_cases = "\n---\n".join(hits['documents'][0])
        except:
            pass

        # 检索figure_db
        figure_features = ""
        try:
            fig_col = sc.get_collection("figure_db")
            fig_hits = fig_col.query(query_embeddings=q_emb, n_results=5)
            figure_features = "\n---\n".join(fig_hits['documents'][0])
            print(f"  figure_db命中{len(fig_hits['documents'][0])}个图")
        except:
            pass

        return {
            "template_text": template_text,
            "style_cases": style_cases,
            "figure_features": figure_features,
            "iteration": 0,
            "history": [f"[检索员] author={author}, chapter={chapter}, 模板{len(template_text)//400}块, 图{len(figure_features)//100}个"]
        }

    def _intent_router_node(self, state: WritingState) -> dict:
        """节点1.3：意图路由——判断写作场景类型"""
        print("  [节点1.3] 意图路由：分析写作场景...")

        resp = self.ds_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": f"""分析以下写作需求属于哪种模式，只输出一个词：

需求：{state['user_request']}

可选模式：
- 整体趋势：描述全过程、阶段划分、受力发展
- 局部状态：分析某一点(如B点)、某个构件的应力状态
- 参数对比：对比不同试件、不同参数(如轴压比、钢板厚度)
- 理论推演：全新构件、知识库没有的内容

只输出一个词："""}],
            temperature=0.1,
        )
        mode = resp.choices[0].message.content.strip()
        print(f"  识别为: {mode}")
        return {"intent_mode": mode, "history": state.get("history", []) + [
            f"[意图路由] {mode}"
        ]}

    def _mechanism_node(self, state: WritingState) -> dict:
        """节点1.5：机制推理员——根据意图模式用不同Prompt"""
        print("  [节点1.5] 机制推理员：分析图表特征...")

        features = state.get("figure_features", "")
        mode = state.get("intent_mode", "整体趋势")
        user_viewpoint = state.get("user_viewpoint", "")

        # 读mechanics_mapping
        try:
            import json
            with open("data/mechanics_mapping.json", "r", encoding="utf-8") as f:
                mapping = json.load(f)
        except:
            mapping = {}

        if not features and not user_viewpoint:
            return {"mechanism_chain": "", "history": state.get("history", [])}

        # 根据模式选Prompt
        prompts = {
            "整体趋势": "请按荷载步时序纵向对比，回答：触发屈服的构件是什么？塑性区如何扩展？哪个构件失稳导致刚度退化？",
            "局部状态": "请空间分析该时刻云图。回答：该构件的应力分布特征？屈服区在哪？主导应力是什么？是否发生局部屈曲？",
            "参数对比": "请对比不同试件的云图特征。回答：参数变化导致哪个区域应力显著改变？破坏模式有何转变？结合曲线数据解释机理。",
            "理论推演": "你是资深结构工程专家。当前分析的是全新构件，缺乏历史文献直接支撑。请基于基础力学原理进行推演。如果无法解释，必须明确输出'【需人工补充力学机制】'，严禁强行编造。"
        }
        reasoning_prompt = prompts.get(mode, prompts["整体趋势"])

        resp = self.ds_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": f"""【图表特征】
{features}

【用户初步分析】（用户已有的力学判断，请严格基于此展开）
{user_viewpoint or '（无）'}

【力学现象映射表】（常见现象→力学含义）
{mapping}

{reasoning_prompt}

输出300字以内，列出结构化的"力学事实清单"："""}],
            temperature=0.2,
        )
        chain = resp.choices[0].message.content.strip()
        print(f"  推理完成：{chain[:80]}...")

        return {"mechanism_chain": chain, "history": state.get("history", []) + [
            f"[机制推理] {mode}模式完成"
        ]}

    def _writer_node(self, state: WritingState) -> dict:
        """节点2：Qwen-VL看图写作"""
        print(f"  [节点2] 写作员(Qwen-VL)：看图写作第{state['iteration']+1}版...")

        # 编码图片
        image_b64 = ""
        if state.get("image_path") and os.path.exists(state["image_path"]):
            ext = os.path.splitext(state["image_path"])[1].lower()
            if ext in (".tif", ".tiff"):
                # TIF转PNG
                import cv2, numpy as np
                with open(state["image_path"], "rb") as f:
                    arr = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                if max(h, w) > 2000:
                    scale = 2000 / max(h, w)
                    img = cv2.resize(img, (int(w*scale), int(h*scale)))
                buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
                image_b64 = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
            else:
                mime = "image/png" if ext == ".png" else "image/jpeg"
                with open(state["image_path"], "rb") as f:
                    image_b64 = f"data:{mime};base64," + base64.b64encode(f.read()).decode()

        content = [
            {"type": "text", "text": f"""你是一位土木工程学术写作专家。

【任务】：请先仔细观察用户上传的图片（这是用户在某个荷载步下的实验云图/曲线图），提取图中的关键信息。
⚠️ 重要：用户提到的"A点/B点/C点"是荷载位移曲线上的荷载步，不是结构上的位置。你只分析当前这张图展示的状态。
⚠️ 重要：只描述用户图片中实际可见的内容。如果图片里没有混凝土板，就不要提混凝土。如果图片里只有钢板，就只分析钢板。

【结构模板】：以下是课题组论文的写作结构，仅供你参考其"逻辑组织方式"（如何引出数据、如何分析、如何下结论）。
⚠️ 严禁复制模板中的任何具体数据、数值、试件编号、宽高比、位移值、应力值。
你只能学习模板的段落结构和句式，所有数据必须来自用户上传的图片。
{state.get('template_text', '（无模板）')}

【导师风格】：以下是导师过往的修改偏好。请模仿这种改法，避免啰嗦和抽象描述。
⚠️ 同样，严禁复制风格案例中的任何具体数据。
{state.get('style_cases', '（无）')}

【审稿意见】（如果有）：{state.get('review_feedback', '（初稿）')}

【机制推理】：以下是根据图表特征推理出的塑性发展因果链，请融入写作中：
{state.get('mechanism_chain', '（无）')}

【用户初步分析】（用户已有的力学判断，请严格基于此润色，绝对禁止添加用户未提及的力学机制或结论）：
{state.get('user_viewpoint', '（无）')}

【硬性要求】：
1. 所有数据必须来自用户上传的图片，不得从模板或风格案例中复制任何数值。
2. 如果用户图片中没有某个数据（如不同宽高比对比），不要编造。
3. 严禁出现与模板文本连续超过8个字的重复片段，必须用自己的语言重组。
4. 被动语态为主，句式严谨。
5. 字数300-500字。

请直接输出正文："""}
        ]

        if image_b64:
            content.insert(0, {"type": "image_url", "image_url": {"url": image_b64}})

        resp = self.vl_client.chat.completions.create(
            model=VL_MODEL_NAME,
            messages=[{"role": "user", "content": content}],
            temperature=0.3,
        )
        draft = resp.choices[0].message.content.strip()

        history = state.get("history", []) + [
            f"[写作员] 第{state['iteration']+1}版初稿完成（{len(draft)}字）"
        ]
        return {"draft": draft, "history": history}

    def _dedup_node(self, state: WritingState) -> dict:
        """节点3：去重——n-gram对比模板文本"""
        print("  [节点3] 去重：与模板文本做n-gram对比...")

        draft = state.get("draft", "")
        template = state.get("template_text", "")

        if not template:
            return {"final_text": draft, "history": state.get("history", [])}

        # 8-gram对比
        def get_ngrams(text, n=8):
            text = re.sub(r'[^\w]', '', text)
            return set(text[i:i+n] for i in range(len(text)-n+1))

        draft_grams = get_ngrams(draft)
        template_grams = get_ngrams(template)
        overlap = draft_grams & template_grams

        if not overlap:
            print("  无重复片段，跳过改写")
            return {"final_text": draft, "history": state.get("history", []) + [
                "[去重] 与模板相似度低，无需改写"
            ]}

        # 有重复，调DeepSeek改写
        print(f"  发现{len(overlap)}个重复片段，同义改写...")
        resp = self.ds_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": f"""以下段落与参考文本有重复，请用自己的话改写，保持学术含义不变：

原文：{draft}

参考文本（避免重复）：{template[:500]}

输出改写后的版本："""}],
            temperature=0.4,
        )
        final = resp.choices[0].message.content.strip()

        return {"final_text": final, "history": state.get("history", []) + [
            f"[去重] 发现{len(overlap)}个重复片段，已改写"
        ]}

    def _reviewer_node(self, state: WritingState) -> dict:
        """节点4：DeepSeek审查"""
        print(f"  [节点4] 审稿人：审查第{state['iteration']+1}版...")

        resp = self.ds_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": f"""你是结构工程硕士生导师，审查学生论文写作。

写作需求：{state['user_request']}
初稿：{state.get('final_text', state.get('draft', ''))}

审查：1.数据是否合理 2.学术表达是否严谨 3.是否有明显抄袭痕迹
合格就输出"通过"，不合格输出修改意见。

审查："""}],
            temperature=0.1,
        )
        feedback = resp.choices[0].message.content.strip()
        passed = "通过" in feedback and len(feedback) < 20

        new_iter = state["iteration"] + 1
        history = state.get("history", []) + [
            f"[审稿人] 第{new_iter}版：{'通过' if passed else '需修改'}"
        ]

        return {
            "review_feedback": feedback if not passed else "",
            "iteration": new_iter,
            "history": history
        }

    def _should_continue(self, state: WritingState) -> str:
        if state["iteration"] >= self.max_iterations:
            return "end"
        if not state.get("review_feedback"):
            return "end"
        return "rewrite"

    def _build_graph(self):
        workflow = StateGraph(WritingState)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("intent_router", self._intent_router_node)
        workflow.add_node("mechanism", self._mechanism_node)
        workflow.add_node("writer", self._writer_node)
        workflow.add_node("dedup", self._dedup_node)
        workflow.add_node("reviewer", self._reviewer_node)

        workflow.set_entry_point("retriever")
        workflow.add_edge("retriever", "intent_router")
        workflow.add_edge("intent_router", "mechanism")
        workflow.add_edge("mechanism", "writer")
        workflow.add_edge("writer", "dedup")
        workflow.add_edge("dedup", "reviewer")

        workflow.add_conditional_edges(
            "reviewer",
            self._should_continue,
            {"rewrite": "writer", "end": END}
        )
        return workflow.compile()

    def run(self, user_request: str, image_path: str = "", user_viewpoint: str = "") -> dict:
        initial: WritingState = {
            "user_request": user_request,
            "user_viewpoint": user_viewpoint,
            "image_path": image_path,
            "template_text": "",
            "style_cases": "",
            "figure_features": "",
            "mechanism_chain": "",
            "draft": "",
            "final_text": "",
            "review_feedback": "",
            "iteration": 0,
            "history": []
        }
        return self.graph.invoke(initial)
