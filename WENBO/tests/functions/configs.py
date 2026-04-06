import numpy as np
from wegp_bayes.utils.variables import NumericalVariable,CategoricalVariable
from wegp_bayes.utils.input_space import InputSpace

def borehole():
    
    V0 = np.array(np.linspace(0.05,0.15,4))
    V1 = np.array(np.linspace(700,820,4))
    

    config = InputSpace()
    r = NumericalVariable(name='r',lower=100,upper=50000)
    Tu = NumericalVariable(name='T_u',lower=63070,upper=115600)
    Hu = NumericalVariable(name='H_u',lower=990,upper=1110)
    Tl = NumericalVariable(name='T_l',lower=63.1,upper=116)
    L = NumericalVariable(name='L',lower=1120,upper=1680)
    K_w = NumericalVariable(name='K_w',lower=9855,upper=12045)
    config.add_inputs([r,Tu,Hu,Tl,L,K_w])

    config.add_input(
        CategoricalVariable(name='r_w',levels=np.linspace(0.05,0.15,4))
    )
    config.add_input(
        CategoricalVariable(name='H_l',levels=np.linspace(700,820,4))
    )
    return config


def piston():
    config = InputSpace()
    M = NumericalVariable(name='M',lower=30,upper=60)
    S = NumericalVariable(name='S',lower=0.005,upper=0.02)
    V0 = NumericalVariable(name='V_0',lower=0.002,upper=0.01)
    Ta = NumericalVariable(name='T_a',lower=290,upper=296)
    T0 = NumericalVariable(name='T_0',lower=340,upper=360)
    config.add_inputs([M,S,V0,Ta,T0])
    config.add_input(
        CategoricalVariable(name='k',levels=np.linspace(1000,5000,4))
    )
    config.add_input(
        CategoricalVariable(name='P_0',levels=np.linspace(90000,110000,4))
    )
    return config

def func2C():
    config = InputSpace()
    x1 = NumericalVariable(name='x1', lower=-1, upper=1)
    x2 = NumericalVariable(name='x2', lower=-1, upper=1)
    ht1 = CategoricalVariable(name='ht1', levels=np.linspace(0, 2, 3))
    ht2 = CategoricalVariable(name='ht2', levels=np.linspace(0, 4, 5))
    config.add_inputs([x1,x2])
    config.add_inputs([ht1, ht2])
    return config


def Ackley3_4():
    config = InputSpace()
    x1 = NumericalVariable(name='x1', lower=-1, upper=1)
    x2 = NumericalVariable(name='x2', lower=-1, upper=1)
    x3 = NumericalVariable(name='x3', lower=-1, upper=1)
    ht1 = CategoricalVariable(name='ht1', levels=np.linspace(0, 1, 3))
    ht2 = CategoricalVariable(name='ht2', levels=np.linspace(0, 1, 3))
    ht3 = CategoricalVariable(name='ht3', levels=np.linspace(0, 1, 3))
    ht4 = CategoricalVariable(name='ht4', levels=np.linspace(0, 1, 3))
    config.add_inputs([x1, x2, x3])
    config.add_inputs([ht1,ht2,ht3,ht4])
    return config


def mlp_mse():
    config = InputSpace()
    ht1 = CategoricalVariable(name='ht1', levels=np.linspace(0, 2, 3))
    ht2 = CategoricalVariable(name='ht2', levels=np.linspace(0, 2, 3))
    ht3 = CategoricalVariable(name='ht3', levels=np.linspace(0, 2, 3))
    x1 = NumericalVariable(name='x1', lower=1, upper=100)
    x2 = NumericalVariable(name='x2', lower=0.0001, upper=1)
    x3 = NumericalVariable(name='x3', lower=0.0001, upper=0.1)
    config.add_inputs([x1,x2,x3])
    config.add_inputs([ht1,ht2,ht3])
    return config



def svm_mse():
    """
    Configuration function defining the hyperparameter search space for SVR.

    Returns:
    - InputSpace: The hyperparameter space configuration.
    """
    config = InputSpace()
    h1 = CategoricalVariable(name='h1', levels=np.linspace(0, 3, 4))

    C = NumericalVariable(name='C', lower=0.1, upper=10)
    epsilon = NumericalVariable(name='epsilon', lower=0.1, upper=1)


    # Add all hyperparameters to the configuration
    config.add_inputs([h1, C, epsilon])

    return config


def func3C():
    config = InputSpace()
    x1 = NumericalVariable(name='x1', lower=-1, upper=1)
    x2 = NumericalVariable(name='x2', lower=-1, upper=1)
    ht1 = CategoricalVariable(name='ht1', levels=np.linspace(0, 2, 3))
    ht2 = CategoricalVariable(name='ht2', levels=np.linspace(0, 4, 5))
    ht3 = CategoricalVariable(name='ht3', levels=np.linspace(0, 3, 4))

    config.add_inputs([x1,x2])
    config.add_inputs([ht1, ht2, ht3])
    return config

def svm_mse():
    """
    Configuration function defining the hyperparameter search space for SVR.

    Returns:
    - InputSpace: The hyperparameter space configuration.
    """
    config = InputSpace()
    h1 = CategoricalVariable(name='h1', levels=np.linspace(0, 3, 4))

    C = NumericalVariable(name='C', lower=0.1, upper=10)
    epsilon = NumericalVariable(name='epsilon', lower=0.1, upper=1)


    # Add all hyperparameters to the configuration
    config.add_inputs([h1, C, epsilon])

    return config


def mlp_mse():
    config = InputSpace()
    ht1 = CategoricalVariable(name='ht1', levels=np.linspace(0, 2, 3))
    ht2 = CategoricalVariable(name='ht2', levels=np.linspace(0, 2, 3))
    ht3 = CategoricalVariable(name='ht3', levels=np.linspace(0, 2, 3))
    x1 = NumericalVariable(name='x1', lower=1, upper=100)
    x2 = NumericalVariable(name='x2', lower=0.0001, upper=1)
    x3 = NumericalVariable(name='x3', lower=0.0001, upper=0.1)
    config.add_inputs([x1,x2,x3])
    config.add_inputs([ht1,ht2,ht3])

    return config
def nnYacht():
    """
    NN-Yacht 的搜索空间（改为分别声明类别超参，保持 function.py 的入参与返回不变）：
      - h1: 激活 {0:relu, 1:tanh, 2:sigmoid}
      - h2: 优化器 {0:sgd, 1:adam, 2:rmsprop, 3:adagrad}
      - h3: dropout 索引 {0..5} -> {0.001, 0.005, 0.01, 0.05, 0.1, 0.5}
      - x1: 学习率（归一化到 [-1,1]，function.py 内部反归一化到 10^[−5,−1]）
      - x2: 隐藏单元数（归一化到 [-1,1]，内部反归一化到 2^[4,7]）
      - x3: 观测噪声方差（归一化到 [-1,1]，内部反归一化到 [0.2,0.8]）
    """
    config = InputSpace()

    # 连续变量（归一化区间）
    x1 = NumericalVariable(name='x1', lower=-1, upper=1)
    x2 = NumericalVariable(name='x2', lower=-1, upper=1)
    x3 = NumericalVariable(name='x3', lower=-1, upper=1)
    config.add_inputs([x1, x2, x3])

    # 类别变量（按照你的“ht*”示例风格分别声明）
    h1 = CategoricalVariable(name='h1', levels=np.linspace(0, 2, 3))  # 0,1,2
    h2 = CategoricalVariable(name='h2', levels=np.linspace(0, 3, 4))  # 0,1,2,3
    h3 = CategoricalVariable(name='h3', levels=np.linspace(0, 5, 6))  # 0..5
    config.add_inputs([h1, h2, h3])

    return config




########################nas 101###########################

from typing import List

NUM_NODES = 7  # 节点 0..6，其中 0=input, 6=output

def _upper_tri_edge_names(n: int = NUM_NODES) -> List[str]:
    """
    生成所有上三角潜在边对应的变量名，顺序固定为 i<j 的字典序：
    edge_p_0_1, edge_p_0_2, ..., edge_p_0_6, edge_p_1_2, ..., edge_p_5_6
    一共 C(7,2)=21 条
    """
    return [f"edge_p_{i}_{j}" for i in range(n) for j in range(i + 1, n)]

def nas101():
    """
    返回混合输入空间配置：
      - 5 个类别变量：op1..op5 ∈ {'conv3x3','conv1x1','maxpool3x3'}
      - 21 个连续变量：edge_p_i_j ∈ [0.0, 1.0]
    """
    config = InputSpace()

    op_levels = ['conv3x3', 'conv1x1', 'maxpool3x3']
    for k in range(1, 6):
        config.add_input(
            CategoricalVariable(name=f'op{k}', levels=op_levels)
        )

    for name in _upper_tri_edge_names():
        config.add_input(
            NumericalVariable(name=name, lower=0.0, upper=1.0)
        )

    return config



def llm_gsm8k():
    """
    仅使用 top_p 解码的 LLM（GSM8K）搜索空间：
      - 连续：top_p ∈ [0.3,1.0]，temperature ∈ [0.2,0.9]
      - 类别：tpl(4)、style(4)
    固定：language='zh'，max_tokens=256；不含 decoding/top_k/few-shot
    """
    config = InputSpace()

    # 连续变量
    top_p = NumericalVariable(name='top_p', lower=0.7, upper=0.9)
    temperature = NumericalVariable(name='temperature', lower=0.2, upper=0.9)
    
    config.add_inputs([top_p, temperature])

    # 类别变量
    strategy_template = CategoricalVariable(name='strategy_template', levels=np.linspace(0, 2, 3))
    prompt_style = CategoricalVariable(name='prompt_style', levels=np.linspace(0, 2, 3))
    # max_tokens = CategoricalVariable(name='max_tokens', levels=np.array([128, 256, 512, 1024]))
    config.add_inputs([strategy_template, prompt_style])

    return config

def llm_gsm8k_1008():
    """
    仅使用 top_p 解码的 LLM（GSM8K）搜索空间：
      - 连续：top_p ∈ [0.3,1.0]，temperature ∈ [0.2,0.9]
      - 类别：tpl(4)、style(4)
    固定：language='zh'，max_tokens=256；不含 decoding/top_k/few-shot
    """
    config = InputSpace()

    # 连续变量
    top_p = NumericalVariable(name='top_p', lower=0.7, upper=0.9)
    temperature = NumericalVariable(name='temperature', lower=0.2, upper=0.9)
    
    config.add_inputs([top_p, temperature])

    # 类别变量
    strategy_template = CategoricalVariable(name='strategy_template', levels=np.linspace(0, 2, 3))
    prompt_style = CategoricalVariable(name='prompt_style', levels=np.linspace(0, 2, 3))
    # max_tokens = CategoricalVariable(name='max_tokens', levels=np.array([128, 256, 512, 1024]))
    config.add_inputs([strategy_template, prompt_style])

    return config


def llm_math():
    """
    仅使用 top_p 解码的 LLM（GSM8K）搜索空间：
      - 连续：top_p ∈ [0.3,1.0]，temperature ∈ [0.2,0.9]
      - 类别：tpl(4)、style(4)
    固定：language='zh'，max_tokens=256（固定项在外部传）
    """
    import numpy as np

    config = InputSpace()

    # 连续变量
    top_p = NumericalVariable(name='top_p', lower=0.3, upper=1.0)
    temperature = NumericalVariable(name='temperature', lower=0.2, upper=0.9)
    config.add_inputs([top_p, temperature])

    # 类别变量（4 档：0/1/2/3）
    strategy_template = CategoricalVariable(name='strategy_template', levels=np.arange(3))
    prompt_style = CategoricalVariable(name='prompt_style', levels=np.arange(3))
    config.add_inputs([strategy_template, prompt_style])

    return config



def llm_agnews():
    """
    仅使用 top_p 解码的 LLM（AG News）搜索空间：
      - 连续：top_p ∈ [0.3,1.0]，temperature ∈ [0.2,0.9]
      - 类别：tpl(3)、style(3)、max_tokens(4)
    固定：language='en'；不含 decoding/top_k/few-shot
    """
    config = InputSpace()

    # 连续变量
    top_p = NumericalVariable(name='top_p', lower=0.3, upper=1.0)
    temperature = NumericalVariable(name='temperature', lower=0.2, upper=0.9)
    config.add_inputs([top_p, temperature])

    # 类别变量（与你现有 evaluator 的枚举长度一致：3×3）
    strategy_template = CategoricalVariable(name='strategy_template', levels=np.linspace(0, 2, 3))
    prompt_style = CategoricalVariable(name='prompt_style', levels=np.linspace(0, 2, 3))

    # 短输出足够：8~64
    # max_tokens = CategoricalVariable(name='max_tokens', levels=np.array([8, 16, 32]))
    config.add_inputs([strategy_template, prompt_style])

    return config



def llm_translation():
    """
    LLM (Translation) 搜索空间：
      - 连续：temperature ∈ [0.2,0.9]，top_p ∈ [0.3,1.0]
      - 类别：prompt_style(4)、max_tokens(3)
    说明：
      prompt_style 索引：
        0 -> faithful
        1 -> concise
        2 -> formal
        3 -> creative
      max_tokens 建议较短；翻译句子稍长，默认给 64/128/256
    """
    config = InputSpace()

    # 连续变量
    temperature = NumericalVariable(name='temperature', lower=0.2, upper=0.9)
    top_p = NumericalVariable(name='top_p', lower=0.3, upper=1.0)
    config.add_inputs([temperature, top_p])

    # 类别变量
    prompt_style = CategoricalVariable(name='prompt_style', levels=np.linspace(0, 3, 4))
    max_tokens = CategoricalVariable(name='max_tokens', levels=np.array([64, 128, 256]))
    config.add_inputs([prompt_style, max_tokens])

    return config


def llm_paraphrase():
    """
    LLM (Paraphrase) 搜索空间：
      - 连续：temperature ∈ [0.2,0.9]，top_p ∈ [0.3,1.0]
      - 类别：prompt_style(4)、max_tokens(3)
    说明：
      prompt_style 索引：
        0 -> faithful
        1 -> concise
        2 -> formal
        3 -> creative
      改写输出建议更短，默认给 32/64/128
    """
    config = InputSpace()

    # 连续变量
    temperature = NumericalVariable(name='temperature', lower=0.2, upper=0.9)
    top_p = NumericalVariable(name='top_p', lower=0.3, upper=1.0)
    config.add_inputs([temperature, top_p])

    # 类别变量
    prompt_style = CategoricalVariable(name='prompt_style', levels=np.linspace(0, 3, 4))
    max_tokens = CategoricalVariable(name='max_tokens', levels=np.array([32, 64, 128]))
    config.add_inputs([prompt_style, max_tokens])

    return config


def llm_arc_mcq(params: dict) -> float:
    """
    ARC Challenge (MCQ) 并发评测 - 单指标：Accuracy
    - 任务：给问题 + 选项（A-D），只输出字母
    - 指标：accuracy（浮点数），函数返回该值
    - 关键优化点：使用 n>1 一次请求多样本 + 多数投票（Self-Consistency）
      -> 随机采样(高温度)单样本Acc较低，但投票后Acc显著上升
    - 连续参数：temperature, top_p（与现有搜索空间兼容）
    - 类别参数：三种“策略模板” × 三种“风格”，保留 3×3 索引使用方式
    - 运行控制：
        LLM_EVAL_N  : 评测样本数（默认 200）
        LLM_VOTE_N  : 一次请求生成的候选数 n（默认 1；设成 5/10/20 可显著提升）
        LLM_EVAL_WORKERS : 并发线程（默认 8）
        LLM_EVAL_RETRIES : 请求重试次数（默认 2）
    """
    import os, re, time
    from typing import Optional, Tuple
    from collections import Counter

    try:
        from datasets import load_dataset
    except Exception:
        return 0.0
    try:
        from openai import OpenAI
    except Exception:
        return 0.0

    # ====== 模板（categorical）——和你现有 3×3 索引逻辑兼容 ======
    STRATEGY_TEMPLATE_SET = ["direct_letter", "cot_brief_letter", "eliminate_then_letter"]
    STYLE_SET = ["terse", "formal", "confident"]

    TEMPLATE_TO_SYSTEM = {
        # 直接答：只输出字母，避免跑题/冗余
        "direct_letter": "Answer ONLY with a single letter among A, B, C, D. No explanation.",
        # 简短CoT：最多两步思考，但最后必须只给字母
        "cot_brief_letter": "Think in at most two very short steps, then answer ONLY with A, B, C, or D.",
        # 先排除再给字母：对模糊题有效，但仍强制最后只给字母
        "eliminate_then_letter": "Eliminate unlikely options briefly, then answer ONLY with A, B, C, or D."
    }
    STYLE_TO_PREFIX = {
        "terse": "Be concise. ",
        "formal": "Be precise and formal. ",
        "confident": "Be decisive; avoid hedging. "
    }
    LANG_TO_SUFFIX = {"en": "Answer in English."}

    # ====== label & 解析 ======
    LETTERS = ["A", "B", "C", "D"]
    LETTER_SET = set(LETTERS)

    def extract_letter(text: Optional[str]) -> Optional[str]:
        """尽量稳健解析：优先 #### <letter>；再找单独字母；再看最后一行收尾字母。"""
        if not text:
            return None
        # 1) #### <letter>
        m = re.findall(r'#{2,}\s*([A-D])\b', text, flags=re.IGNORECASE)
        if m:
            return m[-1].upper()
        # 2) 明确提示后的单字母
        m = re.findall(r'\b([A-D])\b', text, flags=re.IGNORECASE)
        if m:
            return m[-1].upper()
        # 3) 兜底：最后一行的结尾 A-D
        last = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if last:
            m = re.findall(r'([A-D])\b', last[-1], flags=re.IGNORECASE)
            if m:
                return m[-1].upper()
        return None

    # ====== 构造消息 ======
    def build_messages(q: str, options: Tuple[str, str, str, str],
                       prompt_style: str, strategy_template: str, language: str):
        style_prefix = STYLE_TO_PREFIX.get(prompt_style, "")
        strategy_text = TEMPLATE_TO_SYSTEM.get(strategy_template, "")
        lang_suffix = LANG_TO_SUFFIX.get(language, "")
        system_content = f"{style_prefix}{strategy_text} {lang_suffix}".strip()

        opt_block = "\n".join([f"{L}. {t}" for L, t in zip(LETTERS, options)])
        instruction = (
            "Choose the single best option and return ONLY its letter.\n"
            "Output format: #### <A|B|C|D>"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user",
             "content": f"{instruction}\n\nQuestion:\n{q}\n\nOptions:\n{opt_block}\n\nYour answer: #### <letter>"}
        ]

    # ====== 参数 ======
    strategy_template_idx = int(params.get('strategy_template', 0))
    style_idx = int(params.get('prompt_style', 0))
    strategy_template = STRATEGY_TEMPLATE_SET[strategy_template_idx % len(STRATEGY_TEMPLATE_SET)]
    prompt_style = STYLE_SET[style_idx % len(STYLE_SET)]
    temperature = float(params.get('temperature', 0.6))  # 建议默认稍高，利于投票
    top_p = float(params.get('top_p', 0.95))
    max_tokens = int(params.get('max_tokens', 4))       # 只需要 1 个字母，4 足够

    n_eval = int(os.getenv('LLM_EVAL_N', '200') or 200)
    n_samples = int(os.getenv('LLM_VOTE_N', '1') or 1)   # 一次请求 n 个候选；设 5/10/20 观察显著提升
    max_workers = max(1, int(os.getenv('LLM_EVAL_WORKERS', '8') or 8))
    max_retries = int(os.getenv('LLM_EVAL_RETRIES', '2') or 2)
    base_backoff = 0.5

    # ====== 数据 ======
    try:
        ds = load_dataset("ai2_arc", "ARC-Challenge")
        split = ds["validation"].select(range(min(n_eval, len(ds["validation"]))))
    except Exception:
        return 0.0
    items = list(split)
    if not items:
        return 0.0

    # ====== 单样本（一次请求 n 个候选） + 多数投票 ======
    def solve_one(i, item):
        q = item["question"]
        opts = item["choices"]["text"]
        labels = item["choices"]["label"]  # ['A','B','C','D',..]
        # 规范化成 A-D 顺序
        label2text = {lab: txt for lab, txt in zip(labels, opts)}
        options_tuple = tuple(label2text.get(L, "") for L in LETTERS)
        gt = item["answerKey"].strip().upper()

        msgs = build_messages(q, options_tuple, prompt_style, strategy_template, "en")
        t0 = time.perf_counter()
        preds = []

        for attempt in range(max_retries + 1):
            try:
                client = OpenAI(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
                out = client.chat.completions.create(
                    model="qwen2.5-7b-instruct",
                    messages=msgs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n_samples
                )
                # 兼容 choices 结构
                if hasattr(out, "choices"):
                    for ch in out.choices:
                        txt = getattr(getattr(ch, "message", ch), "content", None)
                        pred = extract_letter(txt)
                        if pred in LETTER_SET:
                            preds.append(pred)
                break
            except Exception:
                if attempt < max_retries:
                    time.sleep(base_backoff * (2 ** attempt))

        latency = time.perf_counter() - t0
        final_pred = None
        if preds:
            final_pred = Counter(preds).most_common(1)[0][0]
        correct = 1 if final_pred == gt else 0
        return i, correct, latency, preds, gt  # 把 preds/gt 带回便于你调试

    # ====== 并发执行 ======
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t_wall_start = time.perf_counter()
    workers = min(max_workers, len(items))

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(solve_one, i, it) for i, it in enumerate(items)]
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception:
                pass
    results.sort(key=lambda x: x[0])

    # ====== 统计（唯一指标：accuracy）======
    total = len(results) or 1
    acc = sum(r[1] for r in results) / total
    avg_lat = sum(r[2] for r in results) / total
    t_wall = time.perf_counter() - t_wall_start

    print(f"[time] wall={t_wall:.2f}s | [lat] avg={avg_lat:.3f}s | [acc]={acc:.3f}", flush=True)
    print(
        f"[params] strategy_template={strategy_template}, style={prompt_style}, "
        f"temperature={temperature:.3f}, top_p={top_p:.3f}, "
        f"n_samples={n_samples}, workers={workers}, retries={max_retries}, max_tokens={max_tokens}",
        flush=True,
    )

    # 如需查看错题/投票分布，可解开：
    # for _, ok, _, preds, gt in results[:10]:
    #     if not ok: print("[ERR] vote=", preds, " gt=", gt)

    return float(acc)


def llm_gsm8k_penalty():
    """
    仅使用 top_p 解码的 LLM（GSM8K）搜索空间：
      - 连续：top_p ∈ [0.3,1.0]，temperature ∈ [0.2,0.9]，presence_penalty ∈ [-2.0,2.0]
      - 类别：tpl(4)、style(4)
    固定：language='zh'，max_tokens=256；不含 decoding/top_k/few-shot
    """
    config = InputSpace()

    # 连续变量
    top_p = NumericalVariable(name='top_p', lower=0.7, upper=0.9)
    temperature = NumericalVariable(name='temperature', lower=0.2, upper=0.9)
    presence_penalty = NumericalVariable(name='presence_penalty', lower=0.0, upper=2.0)  # ← 新增
    
    config.add_inputs([top_p, temperature, presence_penalty])  # ← 修改：把 presence_penalty 一并加入

    # 类别变量
    strategy_template = CategoricalVariable(name='strategy_template', levels=np.linspace(0, 2, 3))
    prompt_style = CategoricalVariable(name='prompt_style', levels=np.linspace(0, 2, 3))
    # max_tokens = CategoricalVariable(name='max_tokens', levels=np.array([128, 256, 512, 1024]))
    config.add_inputs([strategy_template, prompt_style])

    return config


def llm_gsm8k_lg():
    """
    LLM（GSM8K）搜索空间（含语言作为第三类别）：
      - 连续：
          * top_p ∈ [0.7, 0.9]
          * temperature ∈ [0.2, 0.9]
      - 类别：
          * strategy_template ∈ {0,1,2,3} -> ['qa','rubric','critique','cot']
          * prompt_style ∈ {0,1,2,3} -> ['concise','step_by_step','formal','creative']
          * language ∈ {0,1,2} -> ['en','zh','mixed']
    """
    config = InputSpace()

    # 连续变量
    top_p = NumericalVariable(name='top_p', lower=0.7, upper=0.9)
    temperature = NumericalVariable(name='temperature', lower=0.2, upper=0.9)
    config.add_inputs([top_p, temperature])

    # 类别变量
    strategy_template = CategoricalVariable(name='strategy_template', levels=np.linspace(0, 3, 4))
    prompt_style = CategoricalVariable(name='prompt_style', levels=np.linspace(0, 3, 4))
    language = CategoricalVariable(name='language', levels=np.linspace(0, 2, 3))
    config.add_inputs([strategy_template, prompt_style, language])

    return config




# -*- coding: utf-8 -*-
"""
配置：返回 InputSpace；风格完全仿照你的示例：
  - 数值变量：eta（学习率）、lam（权重衰减）
  - 类别变量：a（激活，0/1/2）、o1..o6（算子，0..4）
"""

# 这里假定你项目里已定义下列类型；若模块名不同请改成你的路径
# from your_pkg import InputSpace, NumericalVariable, CategoricalVariable

def jash():
    config = InputSpace()

    # 数值变量（范围可按需调整）
    eta = NumericalVariable(name='eta', lower=1e-3, upper=4e-1)
    lam = NumericalVariable(name='lam', lower=1e-6, upper=1e-2)
    config.add_inputs([eta, lam])

    # 类别变量：激活 0/1/2 -> ReLU/Mish/Hardswish
    a = CategoricalVariable(name='a', levels=np.linspace(0, 2, 3))
    # 类别变量：6 条边的算子 0..4 -> zero/skip/1x1/3x3/avgpool
    o1 = CategoricalVariable(name='o1', levels=np.linspace(0, 4, 5))
    o2 = CategoricalVariable(name='o2', levels=np.linspace(0, 4, 5))
    o3 = CategoricalVariable(name='o3', levels=np.linspace(0, 4, 5))
    o4 = CategoricalVariable(name='o4', levels=np.linspace(0, 4, 5))
    o5 = CategoricalVariable(name='o5', levels=np.linspace(0, 4, 5))
    o6 = CategoricalVariable(name='o6', levels=np.linspace(0, 4, 5))
    config.add_inputs([a, o1, o2, o3, o4, o5, o6])

    return config