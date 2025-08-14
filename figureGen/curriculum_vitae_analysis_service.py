from sapperrag.llm.oai import ChatOpenAI
from backend.core.conf import settings
from backend.app.techmanage.schema.profilesource import CreateProfileSourceParam, UpdateProfileSourceParam, \
    GetProfileSourceListDetails
from backend.app.techmanage.service.profilesource_service import profile_source_service
from backend.app.techmanage.service.embedding_profilesource_service import embedding_profile_source_service
from backend.app.techmanage.service.professional_personnel_service import professional_personnel_service
from backend.utils.serializers import select_as_dict
from langdetect import detect
from backend.common.exception import errors
import re
import json
import asyncio

    
class CurriculumVitaeAnalysisService:
    def __init__(self, professional_personnel_id: int) -> None:
        self.chatbot = ChatOpenAI(settings.OPENAI_KEY, settings.OPENAI_BASE_URL)
        self.professional_personnel_id = professional_personnel_id
        self.professional_personnel = None
        self.professional_personnel_info = ""
        self.profilesource_curriculum_vitae_ids = []
        self.profilesources_curriculum_vitae: list[GetProfileSourceListDetails] = []
        self.curriculum_vitae_dict: dict = {}

    async def initialize_sources(self):
        self.professional_personnel = await professional_personnel_service.get_with_source(
            pk=self.professional_personnel_id)
        profilesources = [GetProfileSourceListDetails(**select_as_dict(profilesource)) for profilesource in
                          self.professional_personnel.profilesources]
        for profilesource in profilesources:
            if profilesource.source_type == '简历':
                self.profilesource_curriculum_vitae_ids.append(profilesource.id)
                self.profilesources_curriculum_vitae.append(profilesource)

    async def build_source_curriculum_vitae_context(self, query: str):
        length = len(self.profilesource_curriculum_vitae_ids)
        system = ""
        if length > 3:
            embeddings = await embedding_profile_source_service.get_embeddings_by_query(
                profilesource_ids=self.profilesource_curriculum_vitae_ids,
                query=query, topK=15)
            for emb in embeddings:
                system += emb.content + "\n"
            return system
        else:
            for profilesource in self.profilesources_curriculum_vitae:
                system += profilesource.content + "\n"
            return system

    # 专业人员基础信息
    async def build_professional_personnel_base_context(self):
        professional_personnel_info = await professional_personnel_service.output_professional_personnel_details(
            pk=self.professional_personnel_id)
        self.professional_personnel_info = professional_personnel_info

    async def base_info_agent(self):
        await self.initialize_sources()
        query = """个人基本信息（如姓名、联系方式、教育背景等）、科研项目、学术论文、研究方法与技术、跨学科合作项目、科研成果、科研获奖等"""
        curriculum_vitae_info = await self.build_source_curriculum_vitae_context(query)
        await self.build_professional_personnel_base_context()
        basic_info_str = self.professional_personnel_info

        prompt = f"""
        # 角色
        您是专业人才信息处理专员，负责根据简历信息(中文简历向量化提取的碎片信息)更新专业人员的基本数据。

        # 目标
        根据提供的简历信息(curriculum_vitae_info:中文简历向量化提取的碎片信息)更新专业人员的基础数据(basic_info_str)，输出完整更新后的JSON格式数据（必须包含所有13个字段）。

        ## 输入数据
        1. 当前基础数据{basic_info_str}结构：
        {{
          "姓名": "...",
          "证件号码": "...",
          "性别": "男/女",
          "出生日期": "...", # 日期格式字符串
          "民族": "...",
          "工作单位": "...",
          "最高学历": "...", # 只能为：博士研究生、硕士研究生、本科、大专
          "人才称号": "...",
          "称号级别": "...", # 只能为：国家级、省级
          "职称": "...", # 只能为：教授、副教授、讲师、助教
          "研究领域": "["A", "B", "C"]",
          "专业领域": "...",
          "人员类型": "..." # 只能为：研究人员、技术人员、管理人员
        }}

        2. 新简历信息{curriculum_vitae_info}：包含最新人员信息的结构体

        ## 处理规则
        1. 必须输出所有13个字段（即使没有更新也要保留原值）
        2. 禁止更新姓名(name)字段 - 无论简历信息如何，必须使用原basic_info_str中的姓名
        3. 字段值验证：
           • 最高学历：必须为[博士研究生、硕士研究生、本科、大专]，无效值保留原数据

           • 称号级别：必须为[国家级、省级]，无效值保留原数据

           • 职称：必须为[教授、副教授、讲师、助教]，无效值保留原数据

           • 人员类型：必须为[研究人员、技术人员、管理人员]，无效值保留原数据

        4. 出生日期格式化为"YYYY-MM-DD"

        ## 输出要求
        1. 必须输出完整13个字段（使用英文键名）
        2. JSON格式：
        {{
          "name": "原姓名",
          "identification_number": "证件号码(原值或更新值)",
          "gender": "性别(原值或更新值，中文)",
          "birth_date": "YYYY-MM-DD(原值或更新值)",
          "ethnicity": "民族(原值或更新值)",
          "organization": "工作单位(原值或更新值)",
          "highest_degree": "最高学历(原值或更新值)",
          "talent_honor": "人才称号(原值或更新值)",
          "honor_level": "称号级别(原值或更新值)",
          "professional_rank": "职称(原值或更新值)",
          "research_domain": "研究领域(原值或更新值)",
          "professional_field": "专业领域(原值或更新值)",
          "personnel_category": "人员类型(原值或更新值)"
        }}

        3. 不要包含任何解释性文本，只输出纯净JSON
        """
        messages = [
            {"role": "system", "content": prompt},

        ]

        return self.chatbot.generate(messages)

    # 提取 JSON 对象的函数
    @staticmethod
    async def extract_json_object(data):
        """
        从字符串中提取 JSON 对象（以花括号 {} 包围的部分）。
        :param data: 包含 JSON 对象的字符串
        :return: 提取的 JSON 对象字符串；如果未找到则返回 '{}'
        """
        # 修改正则表达式为跨行模式
        pattern = r'\{.*?\}'  # 如果 JSON 跨行，使用 re.DOTALL
        match = re.search(pattern, data, re.DOTALL)
        return match.group(0) if match else "{}"  # 返回匹配结果或空字典

    async def analyze_curriculum_vitae_dict(self):
        curriculum_vitae_data = await self.base_info_agent()
        curriculum_vitae_json = await self.extract_json_object(curriculum_vitae_data)
        self.curriculum_vitae_dict = json.loads(curriculum_vitae_json)


async def main():
    cv_analysis_service = CurriculumVitaeAnalysisService(1)
    result = await cv_analysis_service.base_info_agent()
    print(result)


if __name__ == '__main__':
    asyncio.run(main())
