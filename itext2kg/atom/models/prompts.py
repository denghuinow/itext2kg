from enum import Enum


class Prompt(Enum):
    EXAMPLES = """ 
    FEW SHOT EXAMPLES \n

    * Michel served as CFO at Acme Corp from 2019 to 2021. He was hired by Beta Inc in 2021, but left that role in 2023.
    -> (Michel, is_CFO_of, Acme Corp, ["01-01-2019"], ["01-01-2021"]), (Michel, works_at, Beta Inc, ["01-01-2021"], ["01-01-2023"])

    * Subsequent experiments confirmed the role of microRNAs in modulating cell growth.
    -> (Experiments, confirm_role_of, microRNAs, [], []), (microRNAs, modulate, Cell Growth, [], [])

    * Researchers used high-resolution imaging in a study on neural plasticity.
    -> (Researchers, use, High-Resolution Imaging, [], []), (High-Resolution Imaging, is_used_in, Study on Neural Plasticity, [], [])

    * Sarah was a board member of GreenFuture until 2019.
    -> (Sarah, is_board_member_of, GreenFuture, [], ["01-01-2019"])

    * Dr. Lee was the head of the Oncology Department until 2022.
    -> (Dr. Lee, is_head_of, Oncology Department, [], ["01-01-2022"])

    * Activity-dependent modulation of receptor trafficking is crucial for maintaining synaptic efficacy.
    -> (Activity-Dependent Modulation, involves, Receptor Trafficking, [], []), (Receptor Trafficking, maintains, Synaptic Efficacy, [], [])

    * (observation_date = 2024-06-15) John Doe is no longer the CEO of GreenIT a few months ago.
    -> (John Doe, is_CEO_of, GreenIT, [], ["2024-03-15"])
    # "a few months ago" ≈ 3 months → 2024-06-15 minus 3 months = 2024-03-15

    * John Doe's marriage is happening on 26-02-2026.
    -> (John Doe, has_status, Married, ["2026-02-26"], [])

    * (observation_date = 2024-03-20) The AI Summit conference started yesterday and will end tomorrow.
    -> (AI Summit, has_status, Started, ["2024-03-19"], ["2024-03-21"])

    * The independence day of Morocco is celebrated on January 1st each year since 1956.
    -> (Morocco, celebrates, Independence Day, ["1956-01-01"], [])

    * (observation_date = 2024-08-10) The product launch event is scheduled for next month.
    -> (Product Launch, has_status, Scheduled, ["2024-09-01"], [])
    # "next month" = first day of September 2024
    
    """

    EXAMPLES_ZH = """
    FEW SHOT EXAMPLES（中文翻译版）\n

    * 米歇尔（Michel）在 2019 年到 2021 年间担任 Acme Corp 的首席财务官（CFO）。他在 2021 年被 Beta Inc 雇用，但在 2023 年离开了该职位。
    -> (Michel, 担任_CFO_于, Acme Corp, ["01-01-2019"], ["01-01-2021"]), (Michel, 就职于, Beta Inc, ["01-01-2021"], ["01-01-2023"])

    * 后续实验确认了 microRNAs 在调节细胞生长中的作用。
    -> (实验, 确认_作用_于, microRNAs, [], []), (microRNAs, 调节, 细胞生长, [], [])

    * 研究人员在一项关于神经可塑性的研究中使用了高分辨率成像。
    -> (研究人员, 使用, 高分辨率成像, [], []), (高分辨率成像, 被用于, 神经可塑性研究, [], [])

    * 萨拉（Sarah）在 2019 年之前一直是 GreenFuture 的董事会成员。
    -> (Sarah, 担任_董事会成员_于, GreenFuture, [], ["01-01-2019"])

    * 李博士（Dr. Lee）在 2022 年之前一直是肿瘤科（Oncology Department）的负责人。
    -> (Dr. Lee, 担任_负责人_于, 肿瘤科, [], ["01-01-2022"])

    * 活动依赖的受体转运调节对于维持突触效能至关重要。
    -> (活动依赖的调节, 涉及, 受体转运, [], []), (受体转运, 维持, 突触效能, [], [])

    * （observation_date = 2024-06-15）John Doe 在几个月前就不再是 GreenIT 的 CEO 了。
    -> (John Doe, 担任_CEO_于, GreenIT, [], ["2024-03-15"])
    # “几个月前”≈3 个月 → 2024-06-15 往前推 3 个月 = 2024-03-15

    * John Doe 的婚礼将在 26-02-2026 举行。
    -> (John Doe, 具有状态, 已婚, ["2026-02-26"], [])

    * （observation_date = 2024-03-20）AI Summit 会议昨天开始，明天结束。
    -> (AI Summit, 具有状态, 已开始, ["2024-03-19"], ["2024-03-21"])

    * 摩洛哥的独立日自 1956 年起每年 1 月 1 日庆祝。
    -> (摩洛哥, 庆祝, 独立日, ["1956-01-01"], [])

    * （observation_date = 2024-08-10）产品发布会计划在下个月举行。
    -> (产品发布会, 具有状态, 已计划, ["2024-09-01"], [])
    # “下个月”= 2024 年 9 月的第一天
    """
    
    @staticmethod
    def examples(output_language: str = "en") -> str:
        return Prompt.EXAMPLES_ZH.value if output_language.lower().startswith("zh") else Prompt.EXAMPLES.value

    @staticmethod
    def temporal_system_query(
        obs_timestamp: str,
        output_language: str = "en",
        entity_name_mode: str = "normalized",
        relation_name_mode: str = "en_snake",
    ) -> str:
        if output_language.lower().startswith("zh"):
            entity_rule = (
                "实体名称（Entity.name）必须与原文一致：中文就保持中文，不要翻译成英文/拼音，不要改写或“标准化”为英文别名。"
                if entity_name_mode == "source"
                else "实体名称（Entity.name）允许做轻度规范化（如去除多余空格），但不要将中文翻译成英文或引入原文未出现的别名。"
            )
            relation_rule = (
                "关系名称（Relationship.name）使用中文、简洁的动词/动宾短语，尽量直接来自原文表达；不要使用英文 snake_case（如 is_founder_of）。"
                if relation_name_mode == "source"
                else "关系名称（Relationship.name）使用英文、简洁的 snake_case（如 is_founder_of / works_at），保持现在时。"
            )
            return f"""
        Observation Time : {obs_timestamp}

        你是一个用于从文本中抽取结构化信息并构建知识图谱的算法。
        请尽可能全面抽取信息，但不要牺牲准确性；禁止添加任何原文未明确提及的信息。

        关键要求（必须遵守）：
        - {entity_rule}（entity_name_mode={entity_name_mode}）
        - 实体类别（Entity.label）可以使用通用英文类别（如 Person/Organization/Event/Field/Model 等），无需与原文一致。
        - {relation_rule}（relation_name_mode={relation_name_mode}）
        - 关系名保持“现在时含义”的规范表达，时间边界用 t_start/t_end 表达（若文本有明确时间信息）。
        """

        entity_extra = ""
        relation_extra = ""
        if entity_name_mode == "source":
            entity_extra = "- Keep entity names (Entity.name) exactly as in the source text; do not translate or romanize.\n"
        if relation_name_mode == "source":
            relation_extra = "- Use Chinese relation names (Relationship.name) as concise verb phrases from the source text; avoid English snake_case.\n"
        return f""" 
        Observation Time : {obs_timestamp}
        
        You are a top-tier algorithm designed for extracting information in structured 
        formats to build a knowledge graph.
        Try to capture as much information from the text as possible without 
        sacrificing accuracy. Do not add any information that is not explicitly mentioned in the text
        Remember, the knowledge graph should be coherent and easily understandable, 
        so maintaining consistency in entity references is crucial.
        {entity_extra}{relation_extra}
        """

'''
class AtomicFactsPrompt(Enum):
    system_query = """
    # Atomic Facts Evaluation Task

    You are an expert evaluator for factual information extraction. Your task is to meticulously compare a list of predicted atomic facts against a gold standard and calculate specific metrics.
    """
    @staticmethod
    def atomic_facts_system_query(ground_truth: list[str], predicted: list[str]) -> str:
        return f"""
    ## Input Format
    - **Gold Standard**: A set of reference atomic facts with their temporal information
    - **Predicted Atomic Facts**: A list of atomic facts with associated temporal information

    ## Evaluation Framework

    ### Phase 1: Content Evaluation
    Evaluate each predicted atomic fact's content against the gold standard without considering the temporal information:

    **MATCH**: A predicted atomic fact that accurately corresponds to a key fact explicitly stated in the gold standard.

    ### Phase 2: Temporal Evaluation
    **Only for atomic facts classified as MATCH in Phase 1**, evaluate their temporal components:

    **Temporal Match (MATCH_t)**: The predicted temporal information accurately corresponds to the temporal bounds stated or reasonably inferable from the gold standard for the matched fact.

    **Temporal Omission (OM_t)**: The gold standard specifies temporal bounds for a fact, but the predicted atomic fact either:
    - Provides no temporal information (null/empty temporal information)
    - Provides incomplete temporal information (missing t_start or t_end when both should be specified)

    ## Evaluation Instructions

    1. **First Pass**: Compare each predicted atomic fact against the gold standard
    - Count MATCH

    2. **Second Pass**: For each MATCH case only, evaluate temporal accuracy
    - Count MATCH_t, OM_t cases

    3. **Output the following counts**:
    - MATCH: [number]
    - MATCH_t: [number]
    - OM_t: [number]

    ## Important Notes
    - Be precise: semantic equivalence counts as a match (e.g., "John Smith" = "J. Smith" if referring to same entity)
    - Temporal evaluation only applies to content matches
    - Consider reasonable temporal tolerance based on the domain and precision of the gold standard

    ---
    Calculate for the following inputs:
    gold_standard: {ground_truth}
    predicted_atomic_facts: {predicted}
    """
'''

'''
class AtomicFactsPrompt(Enum):
    system_query = """
    # Atomic Facts Evaluation Task

    You are an expert evaluator for factual information extraction. Your task is to identify predicted atomic facts that match the gold standard in BOTH content and temporal accuracy.
    """
    
    @staticmethod
    def atomic_facts_system_query(ground_truth: list[str], predicted: list[str]) -> str:
        return f"""
## Evaluation Framework: Temporally-Aware Atomic Facts

Your task is to count how many predicted atomic facts fully match the gold standard (both content AND temporal information).

### What Constitutes a MATCH:

A predicted atomic fact is a **MATCH** if and only if:
1. **Content Accuracy**: The factual content accurately corresponds to a fact explicitly stated in the gold standard
   - Semantic equivalence counts (e.g., "John Smith" = "J. Smith" if same entity)
   - The core information must be the same
   
2. **Temporal Accuracy**: The temporal information is correct
   - If the gold standard specifies temporal bounds, the prediction must match them (within reasonable tolerance)
   - If the gold standard fact is atemporal, the prediction should not add temporal information
   - Both start and end dates must be accurate if specified in gold standard

### What is NOT a MATCH:

A predicted fact is NOT a match if:
- Content is not supported by the gold standard
- Content matches but temporal information is incorrect, hallucinated, or extends beyond what's stated
- Content matches but temporal information is missing when gold standard specifies it
- Content matches but temporal information is incomplete (e.g., missing start or end date when both should be present)

### Evaluation Instructions:

1. Go through each predicted atomic fact
2. For each prediction, ask:
   - Does the content match a gold standard fact?
   - Does the temporal information also match (or is appropriately absent)?
   - Only if BOTH are true → count as MATCH

3. **Output only the MATCH count**

### Important Notes:
- Be precise with temporal matching - consider domain-appropriate tolerance
- Atemporal facts in gold standard should remain atemporal in predictions
- A gold standard fact can only match one prediction (1-to-1 mapping)

---

**Input Data:**
- Gold Standard Facts: {ground_truth}
- Predicted Atomic Facts: {predicted}
"""
'''
