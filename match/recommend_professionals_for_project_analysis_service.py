from sapperrag.llm.oai import ChatOpenAI
from backend.core.conf import settings
from backend.app.techmanage.schema.profile import ProfileSchema
from backend.app.techmanage.schema.project_summary import ProjectSummarySchema
from backend.app.techmanage.schema.match_result import MatchResult
from backend.app.techmanage.service.project_service import project_service
from backend.app.techmanage.service.professional_personnel_service import professional_personnel_service
from typing import Optional
import json
import logging
from datetime import datetime


class RecommendProfessionalsForProjectAnalysisService:
    def __init__(self, project_id: int, professional_personnel_ids: list[int]):
        self.chatbot = ChatOpenAI(settings.OPENAI_KEY, settings.OPENAI_BASE_URL)
        self.project_id = project_id
        self.professional_personnel_ids = professional_personnel_ids
        self.project_summary: Optional[ProjectSummarySchema] = None
        self.profiles: list[ProfileSchema] = []
        self.match_results: list[MatchResult] = []

    async def initialize_profile_and_summary(self):
        project = await project_service.get(pk=self.project_id)
        project_summary = {
            "project_id": self.project_id,
            "project_name": project.project_name,
            "project_summary_content": await project_service.get_project_summary(pk=self.project_id),
            "leader_education_limit": project.leader_education_limit,
            "leader_age_limit": project.leader_age_limit,
            "leader_title_limit": project.leader_title_limit
        }
        self.project_summary = ProjectSummarySchema(**project_summary)
        for professional_personnel_id in self.professional_personnel_ids:
            professional_personnel_base_info = await professional_personnel_service.output_professional_personnel_details(
                pk=professional_personnel_id)
            profile = await professional_personnel_service.get_professional_personnel_profile(
                pk=professional_personnel_id)
            professional_personnel = await professional_personnel_service.get(pk=professional_personnel_id)
            profile_dict = {
                "professional_personnel_id": professional_personnel_id,
                "professional_personnel_name": professional_personnel.name,
                "professional_personnel_base_info": professional_personnel_base_info,
                "profile_content": profile
            }
            self.profiles.append(ProfileSchema(**profile_dict))

    async def project_match_professional_personnel_single_agent(self, project_summary_content: str,
                                                                profile_content: str):
        chatbot = self.chatbot
        professional_personnel_profile = profile_content
        project_summary = project_summary_content
        system_prompt = f"""
        ### 任务描述
        作为智能评估助手，你的任务是通过对提供的科研人员信息（researcher_text）和科研项目需求总结（summary_text）进行深度分析，从多维度评估科研人员与科研项目的匹配程度。具体要求包括全面审视科研人员的专业技能、学术背景、学术成果、实践经验、跨学科合作能力以及技术创新能力等关键指标，结合科研项目的具体目标与需求，对每一维度进行评分并给出明确的分析理由。

        你需要确保所有评分与分析严格基于输入信息，合理推导，并全面体现科研人员的能力优势或潜在不足，为科研项目的人员筛选与优先级建议提供科学依据。

        ### 角色
        作为一名高度专业化的智能评估助手，你不仅具备卓越的信息提取与逻辑推理能力，还能够进行细致严谨的对比分析。你的职责是结合科研人员和科研项目的核心要素，提供科学、系统且客观的评估结果，为复杂科研场景中的决策提供有力支持。你的输出需确保逻辑严密、条理清晰，具备高度的专业性与权威性。

        ### 输入内容
        #### 1. 科研人员信息（professional_personnel_profile）
        {professional_personnel_profile}

        #### 2. 科研项目总结（project_summary）
        {project_summary}

        ### 输出目标
        基于以上信息，你需要完成以下分析和输出：
        1. **科研人员与科研项目匹配程度分析（含评分及理由）**
           - **专业技能匹配（评分：0-10）**：  
             评估科研人员的技术能力、工具掌握情况是否满足科研项目的具体需求，并给出评分的依据，例如特定技术或编程能力的匹配情况。
             - **优点**：例如科研人员具备高级编程能力，掌握特定领域技术工具，符合项目要求。
             - **不足**：例如缺少某些必要的技能或技术，在该领域经验较少。
           - **学术背景匹配（评分：0-10）**：  
             根据科研人员的教育背景、研究领域和学术方向，与科研项目的研究目标和领域相关性进行对比，说明评分依据。
             - **优点**：例如拥有博士学位并在相关领域有丰富的学术研究背景。
             - **不足**：例如学术背景不完全匹配项目要求，缺乏相关研究领域的深度。
           - **学术成果匹配（评分：0-10）**：  
             分析科研人员发表的论文数量与质量、专利申请记录，以及是否有符合科研项目核心要求的重要研究成果，提供评分理由。
             - **优点**：例如拥有多篇高影响力论文，具有创新性研究成果。
             - **不足**：例如学术成果较少，或在相关领域的学术影响力较弱。
           - **实践经验匹配（评分：0-10）**：  
             根据科研人员的科研经历，评估其是否曾参与类似项目或解决过相关领域的技术问题，明确评分的来源和依据。
             - **优点**：例如曾参与多个相关项目，积累了丰富的项目经验。
             - **不足**：例如经验不足，未能涉及相关领域的深度实践。
           - **跨学科合作能力匹配（评分：0-10）**：  
             结合科研人员的团队合作经历与跨学科任务经验，分析其是否能够胜任项目中跨部门、跨领域协作的需求，并提供评分的具体理由。
             - **优点**：例如具备较强的跨学科合作经验，能够有效沟通与协调。
             - **不足**：例如缺少跨学科合作经验，或在团队协作中存在沟通障碍。
           - **技术创新能力匹配（评分：0-10）**：  
             从科研人员的创新成果、技术开发能力、解决技术难题的记录中提取信息，评估其是否符合科研项目对创新能力的具体需求，并说明评分依据。
             - **优点**：例如具有多项技术创新成果，能够应对项目中的技术挑战。
             - **不足**：例如缺乏创新性成果，未能体现出强大的技术开发能力。

        2. **科研项目需求来源分析**
           - 结合科研项目总结，具体分析每条对科研人员的能力和资格需求来源于科研项目的哪些部分（如核心任务、研究目标、技术需求等）。

        3. **匹配优先级建议**
           - **总体匹配评分**：将各维度评分按权重计算总评分，总评分范围为 0-10。
           - 每项维度的评分权重分配如下：专业技能（权重30%）、学术背景（权重20%）、学术成果（权重15%）、实践经验（权重15%）、跨学科合作能力（权重10%）、技术创新能力（权重10%）。
           - **计算步骤**：
             1. 依据每一维度的评分对其进行加权计算。例如：专业技能评分为8分，权重30%，则加权分数为 8 * 0.30 = 2.4。
             2. 将各维度的加权分数相加，得出总分。例如：若其他维度的加权分数分别为 3、2.5、3、1.5、1，则总分为 2.4 + 3 + 2.5 + 3 + 1.5 + 1 = 13.4，总分的计算范围为 0-10分。
           - 综合各项分析，明确科研人员与科研项目的整体匹配程度（高匹配：8-10分，中匹配：5-7分，低匹配：0-4分）。
           - 根据匹配评分和分析结果，提供具体的推荐意见，并说明优先级的理由。

        ### 评分规则
        以下是每一维度的具体评分标准：
        - **0-2 分（极低匹配）**：
          - 科研人员完全不具备该维度中项目所需的核心能力或背景。
          - 示例：项目要求掌握 AI 建模技能，而科研人员对此完全无涉。

        - **3-5 分（低匹配）**：
          - 科研人员在该维度有部分能力或经验，但未达到项目基本要求。
          - 示例：科研人员具备基本数据分析技能，但项目要求更高的算法开发能力。

        - **6-7 分（中等匹配）**：
          - 科研人员的能力或背景能够满足项目基本需求，但可能在深度或广度上略显不足。
          - 示例：科研人员参与过相关领域研究，但在某些细分技术上经验有限。

        - **8-9 分（高匹配）**：
          - 科研人员在该维度的能力与项目需求高度契合，能够胜任相关工作，且有显著优势。
          - 示例：科研人员具备丰富的跨学科合作经验，且有成功案例支持。

        - **10 分（完全匹配）**：
          - 科研人员的能力或背景与项目需求完全吻合，不仅满足需求，还能带来额外价值。
          - 示例：科研人员拥有相关领域深厚积累，并曾主导类似研究项目，取得显著成果。

        ### 输出格式
        请严格按照以下 JSON 结构输出评估结果，确保内容完整详细：
        ```json
        {{
        "matching_analysis": {{
        "skill_match": {{
        "score": 0-10(int型),
              "reason": "详细评分依据说明",
              "strengths": "具体优势描述",
              "weaknesses": "具体不足描述（若无则留空）"
            }},
            "background_match": {{...}},
            "achievement_match": {{...}},
            "experience_match": {{...}},
            "interdisciplinary_match": {{...}},
            "innovation_match": {{...}}
          }},
          "demand_source_analysis": [
            {{
                "requirement_description": "具体能力/资格要求",
                "source": "对应科研项目的具体部分（如：核心任务第X条）"
            }},
            // 其他需求条目
          ],
          "priority_recommendation": {{
                "overall_score": 0-10,
                "calculation_process": {{
                                "skill_match": {{
                                    "sore": X(int型), 
                                    "weight": 0.3(float型), 
                                    "weighted_score": X*0.3
                                    }},
                                "background_match": {{...}},
                                "achievement_match": {{...}},
                                "experience_match": {{...}},
                                "interdisciplinary_match": {{...}},
                                "innovation_match": {{...}},
                                "formula": "∑(维度评分×权重)写出加权求和的计算过程",
                                "total_value": X.XX(float型，保留两位小数)
            }},
            "matching_level": "高/中/低匹配",
            "recommendation": "具体推荐等级（如：优先推荐/谨慎推荐/不推荐）",
            "priority_reason": "结合分析维度的详细说明"
          }}
        }}
        注意事项
            确保所有分析和评估严格基于输入内容，避免主观臆断。
            如果某些信息不足，请合理依托现有数据进行推理，但不得编造或虚构信息。
            输出内容需逻辑清晰、层次分明，确保可读性与可信度。
            如果发现信息存在冲突或不足，应明确指出并说明可能的影响。
            一定要输出JSON格式，并严格按照JSON格式输出。
        """
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        return chatbot.generate(messages)

    async def project_match_professional_personnel_multiple_agent(self, project_summary_content: str,
                                                                  profile_content: str,
                                                                  ):
        chatbot = self.chatbot
        professional_personnel_profile = profile_content
        project_summary = project_summary_content
        system_prompt = f"""
        ### 任务描述
        作为智能评估助手，你的任务是通过对提供的科研人员信息（researcher_text）和科研项目需求总结（summary_text）进行深度分析，从多维度评估科研人员与科研项目的匹配程度。具体要求包括全面审视科研人员的专业技能、学术背景、学术成果、实践经验、跨学科合作能力以及技术创新能力等关键指标，结合科研项目的具体目标与需求，对每一维度进行评分并给出明确的分析理由。
        你需要确保所有评分与分析严格基于输入信息，合理推导，并全面体现科研人员的能力优势或潜在不足，为科研项目的人员筛选与优先级建议提供科学依据。

        ### 角色
        作为一名高度专业化的智能评估助手，你不仅具备卓越的信息提取与逻辑推理能力，还能够进行细致严谨的对比分析。你的职责是结合科研人员和科研项目的核心要素，提供科学、系统且客观的评估结果，为复杂科研场景中的决策提供有力支持。你的输出需确保逻辑严密、条理清晰，具备高度的专业性与权威性。
        你具备跨人员对比分析能力，能建立稳定的评分基准，确保多人员评估时的标准一致性。
        ### 输入内容
        #### 1. 科研人员信息（professional_personnel_profile）
        {professional_personnel_profile}
        #### 2. 科研项目总结（project_summary）
        {project_summary}

        ### 输出目标
        基于以上信息，你需要完成以下分析和输出：
        1. **科研人员与科研项目匹配程度分析（含评分及理由）**
           - **专业技能匹配（评分：0-10）**：  
             评估科研人员的技术能力、工具掌握情况是否满足科研项目的具体需求，并给出评分的依据，例如特定技术或编程能力的匹配情况。
             - **优点**：例如科研人员具备高级编程能力，掌握特定领域技术工具，符合项目要求。
             - **不足**：例如缺少某些必要的技能或技术，在该领域经验较少。
           - **学术背景匹配（评分：0-10）**：  
             根据科研人员的教育背景、研究领域和学术方向，与科研项目的研究目标和领域相关性进行对比，说明评分依据。
             - **优点**：例如拥有博士学位并在相关领域有丰富的学术研究背景。
             - **不足**：例如学术背景不完全匹配项目要求，缺乏相关研究领域的深度。
           - **学术成果匹配（评分：0-10）**：  
             分析科研人员发表的论文数量与质量、专利申请记录，以及是否有符合科研项目核心要求的重要研究成果，提供评分理由。
             - **优点**：例如拥有多篇高影响力论文，具有创新性研究成果。
             - **不足**：例如学术成果较少，或在相关领域的学术影响力较弱。
           - **实践经验匹配（评分：0-10）**：  
             根据科研人员的科研经历，评估其是否曾参与类似项目或解决过相关领域的技术问题，明确评分的来源和依据。
             - **优点**：例如曾参与多个相关项目，积累了丰富的项目经验。
             - **不足**：例如经验不足，未能涉及相关领域的深度实践。
           - **跨学科合作能力匹配（评分：0-10）**：  
             结合科研人员的团队合作经历与跨学科任务经验，分析其是否能够胜任项目中跨部门、跨领域协作的需求，并提供评分的具体理由。
             - **优点**：例如具备较强的跨学科合作经验，能够有效沟通与协调。
             - **不足**：例如缺少跨学科合作经验，或在团队协作中存在沟通障碍。
           - **技术创新能力匹配（评分：0-10）**：  
             从科研人员的创新成果、技术开发能力、解决技术难题的记录中提取信息，评估其是否符合科研项目对创新能力的具体需求，并说明评分依据。
             - **优点**：例如具有多项技术创新成果，能够应对项目中的技术挑战。
             - **不足**：例如缺乏创新性成果，未能体现出强大的技术开发能力。

        2. **科研项目需求来源分析**
           - 结合科研项目总结，具体分析每条对科研人员的能力和资格需求来源于科研项目的哪些部分（如核心任务、研究目标、技术需求等）。

        3. **匹配优先级建议**
           - **总体匹配评分**：将各维度评分按权重计算总评分，总评分范围为 0-10。
           - 每项维度的评分权重分配如下：专业技能（权重30%）、学术背景（权重20%）、学术成果（权重15%）、实践经验（权重15%）、跨学科合作能力（权重10%）、技术创新能力（权重10%）。
           - **计算步骤**：
             1. 依据每一维度的评分对其进行加权计算。例如：专业技能评分为8分，权重30%，则加权分数为 8 * 0.30 = 2.4。
             2. 将各维度的加权分数相加，得出总分。例如：若其他维度的加权分数分别为 3、2.5、3、1.5、1，则总分为 2.4 + 3 + 2.5 + 3 + 1.5 + 1 = 13.4，总分的计算范围为 0-10分。
           - 综合各项分析，明确科研人员与科研项目的整体匹配程度（高匹配：8-10分，中匹配：5-7分，低匹配：0-4分）。
           - 根据匹配评分和分析结果，提供具体的推荐意见，并说明优先级的理由。

        ### 评分规则
        以下是每一维度的具体评分标准：
        - **0-2 分（极低匹配）**：
          - 科研人员完全不具备该维度中项目所需的核心能力或背景。
          - 示例：项目要求掌握 AI 建模技能，而科研人员对此完全无涉。

        - **3-5 分（低匹配）**：
          - 科研人员在该维度有部分能力或经验，但未达到项目基本要求。
          - 示例：科研人员具备基本数据分析技能，但项目要求更高的算法开发能力。

        - **6-7 分（中等匹配）**：
          - 科研人员的能力或背景能够满足项目基本需求，但可能在深度或广度上略显不足。
          - 示例：科研人员参与过相关领域研究，但在某些细分技术上经验有限。

        - **8-9 分（高匹配）**：
          - 科研人员在该维度的能力与项目需求高度契合，能够胜任相关工作，且有显著优势。
          - 示例：科研人员具备丰富的跨学科合作经验，且有成功案例支持。

        - **10 分（完全匹配）**：
          - 科研人员的能力或背景与项目需求完全吻合，不仅满足需求，还能带来额外价值。
          - 示例：科研人员拥有相关领域深厚积累，并曾主导类似研究项目，取得显著成果。

        ### 输出格式
        请严格按照以下 JSON 结构输出评估结果，确保内容完整详细：
        ```json
        {{
        "matching_analysis": {{
        "skill_match": {{
        "score": 0-10(int型),
              "reason": "详细评分依据说明",
              "strengths": "具体优势描述",
              "weaknesses": "具体不足描述（若无则留空）"
            }},
            "background_match": {{...}},
            "achievement_match": {{...}},
            "experience_match": {{...}},
            "interdisciplinary_match": {{...}},
            "innovation_match": {{...}}
          }},
          "demand_source_analysis": [
            {{
                "requirement_description": "具体能力/资格要求",
                "source": "对应科研项目的具体部分（如：核心任务第X条）"
            }},
            // 其他需求条目
          ],
          "priority_recommendation": {{
                "overall_score": 0-10,
                "calculation_process": {{
                                "skill_match": {{
                                    "sore": X(int型), 
                                    "weight": 0.3(float型), 
                                    "weighted_score": X*0.3
                                    }},
                                "background_match": {{...}},
                                "achievement_match": {{...}},
                                "experience_match": {{...}},
                                "interdisciplinary_match": {{...}},
                                "innovation_match": {{...}},
                                "formula": "∑(维度评分×权重)写出加权求和的计算过程",
                                "total_value": X.XX(float型，保留两位小数)
            }},
            "matching_level": "高/中/低匹配",
            "recommendation": "具体推荐等级（如：优先推荐/谨慎推荐/不推荐）",
            "priority_reason": "结合分析维度的详细说明"
          }}
        }}
        注意事项
            确保所有分析和评估严格基于输入内容，避免主观臆断。
            如果某些信息不足，请合理依托现有数据进行推理，但不得编造或虚构信息。
            输出内容需逻辑清晰、层次分明，确保可读性与可信度。
            如果发现信息存在冲突或不足，应明确指出并说明可能的影响。
            一定要输出JSON格式，并严格按照JSON格式输出。
        """
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        return chatbot.generate(messages)

    async def type_change(self, answer: str):
        # 处理答案中的特殊字符
        new_text = answer.replace("```", "")
        new_text1 = new_text.replace("json", "")

        # 将 JSON 字符串转换为python对象
        data = json.loads(new_text1)
        return data

    async def project_hard_conditions_match_agent(self, project_hard_conditions: str,
                                                  professional_personnel_base_info: str):
        chatbot = self.chatbot
        current_time = datetime.now()
        system_prompt = f"""
        # 任务描述
        你是一个专业资质验证智能体，负责严格验证专业人员是否符合项目的硬性条件要求。需要基于项目硬性条件和专业人员信息进行精确匹配，当关键信息缺失时需明确标识。

        # 角色
        项目资质验证官

        # 输入内容
        1. 项目硬性条件（project_hard_conditions）:
           - 负责人学历要求
           - 负责人职称要求
           - 负责人年龄要求
           - 格式：字符串，每行一个条件（"无要求"表示该条件不限制）
           -{project_hard_conditions}

        2. 专业人员信息（professional_personnel_base_info）:

           - 包含学历、职称、出生日期等基本信息
           - 当前时间（用于计算年龄）是{current_time.strftime('%Y-%m-%d')}
           - {professional_personnel_base_info}

        # 输出目标
        判断专业人员是否符合项目所有硬性条件，出现以下情况时：
        1. 满足所有条件 → 符合
        2. 任意条件不满足 → 不符合
        3. 关键信息缺失导致无法验证 → 无法确定

        # 输出格式
        严格的JSON格式：
        {{
            "result": "符合硬性条件" | "不符合硬性条件" | "专业人员缺失信息无法确定",
            "reason": [
                "详细判断依据1",
                "详细判断依据2",
                ...
            ]
        }}

        # 验证规则
        1. 逐条检查项目硬性条件（忽略"无要求"的条件）
        2. 年龄计算：基于专业人员出生日期和当前时间精确计算周岁
        3. 严格匹配：学历/职称需完全匹配要求
        4. 信息缺失：当专业人员缺少必要字段时返回无法确定

        # 示例输出
        情况1：符合条件
        {{"result": "符合硬性条件", "reason": ["学历要求：博士（符合）", "职称要求：教授（符合）", "年龄要求：45岁（≤50岁）"]}}

        情况2：不符合条件
        {{"result": "不符合硬性条件", "reason": ["年龄要求：52岁（超过50岁限制）"]}}

        情况3：信息缺失
        {{"result": "专业人员缺失信息无法确定", "reason": ["缺失职称信息"]}}
        """
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        return chatbot.generate(messages)

    async def create_match_result(self, data: dict, personnel_id: int, project_id: int) -> MatchResult:
        """将评估数据转换为MatchResult对象

        Args:
            data: 评估结果字典数据
            personnel_id: 科研人员ID
            project_id: 科研项目ID

        Returns:
            MatchResult实例
        """
        # 提取匹配分析数据
        matching = data['matching_analysis']

        # 提取优先级建议和计算过程
        priority = data['priority_recommendation']
        calc_process = priority['calculation_process']

        # 创建MatchResult实例
        return MatchResult(
            # 基本信息
            professional_personnel_id=personnel_id,
            project_id=project_id,

            # 专业技能匹配维度
            skill_match_score=matching['skill_match']['score'],
            skill_match_weight=calc_process['skill_match']['weight'],
            skill_match_reason=matching['skill_match']['reason'],
            skill_match_strengths=matching['skill_match']['strengths'],
            skill_match_weaknesses=matching['skill_match']['weaknesses'],

            # 学术背景匹配维度
            background_match_score=matching['background_match']['score'],
            background_match_weight=calc_process['background_match']['weight'],
            background_match_reason=matching['background_match']['reason'],
            background_match_strengths=matching['background_match']['strengths'],
            background_match_weaknesses=matching['background_match']['weaknesses'],

            # 学术成果匹配维度
            achievement_match_score=matching['achievement_match']['score'],
            achievement_match_weight=calc_process['achievement_match']['weight'],
            achievement_match_reason=matching['achievement_match']['reason'],
            achievement_match_strengths=matching['achievement_match']['strengths'],
            achievement_match_weaknesses=matching['achievement_match']['weaknesses'],

            # 实践经验匹配维度
            experience_match_score=matching['experience_match']['score'],
            experience_match_weight=calc_process['experience_match']['weight'],
            experience_match_reason=matching['experience_match']['reason'],
            experience_match_strengths=matching['experience_match']['strengths'],
            experience_match_weaknesses=matching['experience_match']['weaknesses'],

            # 跨学科合作能力匹配维度
            interdisciplinary_match_score=matching['interdisciplinary_match']['score'],
            interdisciplinary_match_weight=calc_process['interdisciplinary_match']['weight'],
            interdisciplinary_match_reason=matching['interdisciplinary_match']['reason'],
            interdisciplinary_match_strengths=matching['interdisciplinary_match']['strengths'],
            interdisciplinary_match_weaknesses=matching['interdisciplinary_match']['weaknesses'],

            # 技术创新能力匹配维度
            innovation_match_score=matching['innovation_match']['score'],
            innovation_match_weight=calc_process['innovation_match']['weight'],
            innovation_match_reason=matching['innovation_match']['reason'],
            innovation_match_strengths=matching['innovation_match']['strengths'],
            innovation_match_weaknesses=matching['innovation_match']['weaknesses'],

            # 总体匹配评分信息
            overall_score=priority['overall_score'],
            matching_level=priority['matching_level'],
            recommendation=priority['recommendation'],
            priority_reason=priority['priority_reason'],

            # 计算过程信息
            calculation_process=str(priority['calculation_process']),  # 转换为字符串存储
            calc_formula=calc_process['formula'],
            calc_total_value=calc_process['total_value']
        )

    async def professional_personnel_match_project(self):

        # 项目画像
        project_summary_content = self.project_summary.project_summary_content
        # 项目硬性条件
        project_hard_conditions = ""
        if self.project_summary.leader_age_limit is not None:
            project_hard_conditions += "负责人学历要求:" + self.project_summary.leader_age_limit + "\n"
        else:
            project_hard_conditions += "负责人学历要求:无要求\n"
        if self.project_summary.leader_title_limit is not None:
            project_hard_conditions += "负责人职称要求:" + self.project_summary.leader_title_limit + "\n"
        else:
            project_hard_conditions += "负责人职称要求:无要求\n"
        if self.project_summary.leader_age_limit is not None:
            project_hard_conditions += "负责人年龄要求:" + self.project_summary.leader_age_limit + "\n"
        else:
            project_hard_conditions += "负责人年龄要求:无要求\n"

        # 对第一个科研人员进行匹配
        first_professional_personnel_profile = self.profiles[0]
        first_professional_personnel_id = self.profiles[0].professional_personnel_id
        first_professional_personnel_base_info = first_professional_personnel_profile.professional_personnel_base_info
        first_professional_personnel_profile_content = first_professional_personnel_profile.profile_content

        # 第一个科研人员匹配项目形成基准
        first_professional_personnel_match_result = await self.project_match_professional_personnel_single_agent(
            project_summary_content,
            first_professional_personnel_profile_content
        )
        first_professional_personnel_match_result_dict = await self.type_change(
            first_professional_personnel_match_result)
        if first_professional_personnel_match_result_dict:
            first_professional_personnel_match_result_obj = await self.create_match_result(
                data=first_professional_personnel_match_result_dict,
                personnel_id=first_professional_personnel_id,
                project_id=self.project_id
            )
            first_professional_personnel_match_result_obj.professional_personnel_name = first_professional_personnel_profile.professional_personnel_name
            first_professional_personnel_match_result_obj.project_name = self.project_summary.project_name
            first_professional_personnel_hard_condition_match_result = await self.project_hard_conditions_match_agent(
                project_hard_conditions,
                first_professional_personnel_base_info
            )
            first_professional_personnel_hard_condition_match_result_dict = await self.type_change(
                first_professional_personnel_hard_condition_match_result)
            if first_professional_personnel_hard_condition_match_result_dict is not None:
                first_professional_personnel_hard_condition_match_reason = ""
                for reason in first_professional_personnel_hard_condition_match_result_dict['reason']:
                    first_professional_personnel_hard_condition_match_reason += reason + "\n"
                first_professional_personnel_match_result_obj.hard_condition_match_result = \
                    first_professional_personnel_hard_condition_match_result_dict["result"]
                first_professional_personnel_match_result_obj.hard_condition_match_reason = first_professional_personnel_hard_condition_match_reason
            self.match_results.append(first_professional_personnel_match_result_obj)

        if len(self.profiles) > 1:
            # 对剩余科研人员进行匹配
            for i in range(1, len(self.profiles)):
                professional_personnel_profile = self.profiles[i]
                professional_personnel_id = professional_personnel_profile.professional_personnel_id
                professional_personnel_base_info = professional_personnel_profile.professional_personnel_base_info
                professional_personnel_profile_content = professional_personnel_profile.profile_content
                professional_personnel_match_result = await self.project_match_professional_personnel_multiple_agent(
                    project_summary_content=project_summary_content,
                    profile_content=professional_personnel_profile_content
                )
                professional_personnel_match_result_dict = await self.type_change(
                    professional_personnel_match_result)
                if professional_personnel_match_result_dict:
                    professional_personnel_match_result_obj = await self.create_match_result(
                        data=professional_personnel_match_result_dict,
                        personnel_id=professional_personnel_id,
                        project_id=self.project_id
                    )
                    professional_personnel_match_result_obj.professional_personnel_name = professional_personnel_profile.professional_personnel_name
                    professional_personnel_match_result_obj.project_name = self.project_summary.project_name
                    professional_personnel_hard_condition_match_result = await self.project_hard_conditions_match_agent(
                        project_hard_conditions,
                        professional_personnel_base_info
                    )
                    professional_personnel_hard_condition_match_result_dict = await self.type_change(
                        professional_personnel_hard_condition_match_result)
                    if professional_personnel_hard_condition_match_result_dict is not None:
                        professional_personnel_hard_condition_match_reason = ""
                        for reason in professional_personnel_hard_condition_match_result_dict['reason']:
                            professional_personnel_hard_condition_match_reason += reason + "\n"
                        professional_personnel_match_result_obj.hard_condition_match_result = \
                            professional_personnel_hard_condition_match_result_dict["result"]
                        professional_personnel_match_result_obj.hard_condition_match_reason = professional_personnel_hard_condition_match_reason
                    self.match_results.append(professional_personnel_match_result_obj)


async def main():
    project_id = 2
    professional_personnel_ids = [12, 13, 14]
    recommend_projects_for_professional_analysis_service = RecommendProfessionalsForProjectAnalysisService(project_id,
                                                                                                           professional_personnel_ids)
    await recommend_projects_for_professional_analysis_service.initialize_profile_and_summary()
    await recommend_projects_for_professional_analysis_service.professional_personnel_match_project()
    for match_result in recommend_projects_for_professional_analysis_service.match_results:
        print(match_result)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
