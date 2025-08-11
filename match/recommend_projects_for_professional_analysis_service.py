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
import os

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_DIR = os.path.join(PROJECT_ROOT, 'prompts')


class RecommendProjectsForProfessionalAnalysisService:
    def __init__(self, project_ids: list[int], professional_personnel_id: int):
        self.chatbot = ChatOpenAI(settings.OPENAI_KEY, settings.OPENAI_BASE_URL)
        self.project_ids = project_ids
        self.professional_personnel_id = professional_personnel_id
        self.project_summaries: list[ProjectSummarySchema] = []
        self.profile: Optional[ProfileSchema] = None
        self.match_results: list[MatchResult] = []

    async def initialize_profile_and_summary(self):
        professional_personnel = await professional_personnel_service.get(pk=self.professional_personnel_id)

        profile_dict = {
            "professional_personnel_id": self.professional_personnel_id,
            "professional_personnel_name": professional_personnel.name,
            "professional_personnel_base_info": await professional_personnel_service.output_professional_personnel_details(
                pk=self.professional_personnel_id),
            "profile_content": await professional_personnel_service.get_professional_personnel_profile(
                pk=self.professional_personnel_id)
        }
        self.profile = ProfileSchema(**profile_dict)
        for project_id in self.project_ids:
            project = await project_service.get(pk=project_id)
            project_summary = await project_service.get_project_summary(pk=project_id)
            project_summary_dict = {
                "project_id": project_id,
                "project_name": project.project_name,
                "project_summary_content": project_summary,
                "leader_education_limit": project.leader_education_limit,
                "leader_age_limit": project.leader_age_limit,
                "leader_title_limit": project.leader_title_limit
            }
            self.project_summaries.append(ProjectSummarySchema(**project_summary_dict))

    async def project_match_professional_personnel_single_agent(self, project_summary_content: str,
                                                                profile_content: str):
        chatbot = self.chatbot
        professional_personnel_profile = profile_content
        project_summary = project_summary_content
        
        # 从外部文件读取提示词模板
        with open(os.path.join(PROMPT_DIR, 'project_personnel_match_prompt.txt'), 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        # 填充模板中的变量
        system_prompt = prompt_template.format(
            professional_personnel_profile=professional_personnel_profile,
            project_summary=project_summary
        )
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        return chatbot.generate(messages)

    async def project_hard_conditions_match_agent(self, project_hard_conditions: str,
                                                  professional_personnel_base_info: str):
        chatbot = self.chatbot
        current_time = datetime.now()
        
        # 从外部文件读取提示词模板
        with open(os.path.join(PROMPT_DIR, 'hard_conditions_match_prompt.txt'), 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        # 填充模板中的变量
        system_prompt = prompt_template.format(
            project_hard_conditions=project_hard_conditions,
            current_time=current_time.strftime('%Y-%m-%d'),
            professional_personnel_base_info=professional_personnel_base_info
        )
        
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

    async def project_match_professional_personnel(self):

        for project_summary in self.project_summaries:
            project_summary_content = project_summary.project_summary_content
            profile_content = self.profile.profile_content
            match_result = await self.project_match_professional_personnel_single_agent(project_summary_content,
                                                                                        profile_content)
            match_result_dict = await self.type_change(match_result)
            if match_result_dict:
                match_result_obj = await self.create_match_result(data=match_result_dict,
                                                                  personnel_id=self.professional_personnel_id,
                                                                  project_id=project_summary.project_id)
                match_result_obj.project_name = project_summary.project_name
                match_result_obj.professional_personnel_name = self.profile.professional_personnel_name
                project_hard_conditions = ""
                if project_summary.leader_age_limit is not None:
                    project_hard_conditions += "负责人学历要求:" + project_summary.leader_education_limit + "\n"
                else:
                    project_hard_conditions += "负责人学历要求:无要求\n"
                if project_summary.leader_title_limit is not None:
                    project_hard_conditions += "负责人职称要求:" + project_summary.leader_title_limit + "\n"
                else:
                    project_hard_conditions += "负责人职称要求:无要求\n"
                if project_summary.leader_age_limit is not None:
                    project_hard_conditions += "负责人年龄要求:" + project_summary.leader_age_limit + "\n"
                else:
                    project_hard_conditions += "负责人年龄要求:无要求\n"

                hard_condition_match_result = await self.project_hard_conditions_match_agent(project_hard_conditions,
                                                                                             self.profile.professional_personnel_base_info)
                hard_condition_match_result_dict = await self.type_change(hard_condition_match_result)
                if hard_condition_match_result_dict is not None:
                    hard_condition_match_reason = ""
                    for reason in hard_condition_match_result_dict['reason']:
                        hard_condition_match_reason += reason + "\n"
                    match_result_obj.hard_condition_match_result = hard_condition_match_result_dict['result']
                    match_result_obj.hard_condition_match_reason = hard_condition_match_reason
                self.match_results.append(match_result_obj)


async def main():
    project_ids = [2, 4, 5]
    professional_personnel_id = 12
    recommend_projects_for_professional_analysis_service = RecommendProjectsForProfessionalAnalysisService(project_ids,
                                                                                                           professional_personnel_id)
    await recommend_projects_for_professional_analysis_service.initialize_profile_and_summary()

    await recommend_projects_for_professional_analysis_service.project_match_professional_personnel()
    for match_result in recommend_projects_for_professional_analysis_service.match_results:
        print(match_result)


async def main2():
    project_ids = [2, 4, 5]
    professional_personnel_id = 12
    recommend_projects_for_professional_analysis_service = RecommendProjectsForProfessionalAnalysisService(project_ids,
                                                                                                           professional_personnel_id)
    await recommend_projects_for_professional_analysis_service.initialize_profile_and_summary()
    print(recommend_projects_for_professional_analysis_service.profile.professional_personnel_base_info)
    for project_summary in recommend_projects_for_professional_analysis_service.project_summaries:
        print(project_summary.leader_age_limit)
        print(project_summary.leader_education_limit)
        print(project_summary.leader_title_limit)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
