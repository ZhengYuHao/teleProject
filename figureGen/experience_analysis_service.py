from sapperrag.llm.oai import ChatOpenAI
from backend.core.conf import settings
from backend.app.techmanage.schema.profilesource import CreateProfileSourceParam, UpdateProfileSourceParam, \
    GetProfileSourceListDetails
from backend.app.techmanage.service.profilesource_service import profile_source_service
from backend.app.techmanage.service.experience_service import experience_service
from backend.app.techmanage.schema.experience import GetExperienceListDetails
from backend.app.techmanage.service.embedding_profilesource_service import embedding_profile_source_service
from backend.utils.serializers import select_as_dict
from langdetect import detect
from backend.common.exception import errors
import re
import json


class ExperienceAnalysisService:
    def __init__(self, experience_id: int) -> None:
        self.chatbot = ChatOpenAI(settings.OPENAI_KEY, settings.OPENAI_BASE_URL)
        self.experience_id = experience_id
        self.experience = None
        self.profilesource_experience_ids = []
        self.experience_analysis_dict: dict = {}

    async def initialize_experience(self):
        self.experience = await experience_service.get_with_profilesource(pk=self.experience_id)
        profilesources = [GetProfileSourceListDetails(**select_as_dict(profilesource)) for profilesource in
                          self.experience.profilesources]
        for profilesource in profilesources:
            if profilesource.source_type == '项目经历':
                self.profilesource_experience_ids.append(profilesource.id)

    async def build_experience_context(self):
        query = """
        请系统描述该科研项目的技术实施全流程，需包含：
        1. 摘要
        2. 关键词
        3. 研究背景与立项依据
        4. 关键技术路线选择逻辑 
        5. 核心技术原理及创新点
        6. 技术难点突破过程
        7. 研究成果的实践价值
        8. 项目目标达成度分析
        9. 领域关键词标引
        """
        embeddings = await embedding_profile_source_service.get_embeddings_by_query(
            profilesource_ids=self.profilesource_experience_ids,
            query=query, topK=20)
        system = ""
        index = 1
        for emb in embeddings:
            system += f"这是第{index}块\n" + emb.content + "\n"
            index += 1
        return system

    # 提取 JSON 对象的函数
    @staticmethod
    def extract_json_object(data):
        """
        从字符串中提取 JSON 对象（以花括号 {} 包围的部分）。
        :param data: 包含 JSON 对象的字符串
        :return: 提取的 JSON 对象字符串；如果未找到则返回 '{}'
        """
        # 修改正则表达式为跨行模式
        pattern = r'\{.*?\}'  # 如果 JSON 跨行，使用 re.DOTALL
        match = re.search(pattern, data, re.DOTALL)
        return match.group(0) if match else "{}"  # 返回匹配结果或空字典

    async def analyze_experience(self):
        await self.initialize_experience()
        experience_data = await self.build_experience_context()
        prompt = f"""
        作为资深技术文档分析师，您将根据项目经历切片进行结构化分析。输出需严格遵循以下JSON格式，字段值使用字符串或数组，空值字段保留为null。

        # 输入数据
        项目切片内容：
        {experience_data}

        # 角色与目标
        - 角色：技术文档分析师
        - 目标：从项目切片中提取关键信息并生成结构化分析报告

        # 分析指南（按输出字段说明）
        1. project_type（项目类型）：
           - 识别项目资助类型（如国家自然科学基金/重点研发计划）
           - 示例："国家自然科学基金面上项目"

        2. grant_number（项目批准号）：
           - 提取项目唯一标识编号
           - 示例："61976025"

        3. funding_amount（资助金额）：
           - 提取金额数值及单位
           - 示例："80万元"

        4. project_period（项目周期）：
           - 提取起止年月，格式：YYYY-MM至YYYY-MM
           - 示例："2023-09至2025-08"

        5. abstract（项目摘要）：
           - 项目摘要

        6. research_field（研究领域）：
           - 确定项目所属学科领域

        7. keywords（关键词）：
           - 提取核心专业术语
           - 格式：["术语1", "术语2", "术语3",...]
           - 示例：["深度学习", "目标检测", "模型压缩"]

        8. research_subject（研究主题）：
           - 核心研究内容

        9. research_problem（研究问题）：
           - 阐明待解决的关键技术瓶颈

        10. research_objective_and_significance（研究目标及意义）：
            - 描述技术目标与行业价值

        11. technical_approach（技术方案）：
            - 说明技术路线与实现方法

        12. core_technology（核心技术）：
            - 说明使用的核心技术

        13. technical_difficulty（技术难点）：
            - 分析主要技术挑战及对策

        # 输出要求
        - 严格使用JSON格式，字段顺序与示例保持一致
        - 技术描述需量化具体（如"提升30%效率"）
        - 空字段使用null表示
        - 禁止使用Markdown格式
        - 尽可能详细

        # 响应格式
        {{
          "project_type": "字符串/null",
          "grant_number": "字符串/null",
          "funding_amount": "字符串/null",
          "project_period": "字符串/null",
          "abstract": "字符串/null",
          "research_field": "字符串/null",
          "keywords": ["关键词1"， "关键词2"，...],
          "research_subject": "字符串/null",
          "research_problem": "字符串/null",
          "research_objective_and_significance": "字符串/null",
          "technical_approach": "字符串/null",
          "core_technology": "字符串/null",
          "technical_difficulty": "字符串/null"
        }}
        """

        messages = [
            {"role": "system", "content": prompt},

        ]

        return self.chatbot.generate(messages)

    async def analyze_experience_dict(self):
        experience_date = await self.analyze_experience()
        experience_json = self.extract_json_object(experience_date)
        experience_dict = json.loads(experience_json)
        self.experience_analysis_dict = experience_dict
