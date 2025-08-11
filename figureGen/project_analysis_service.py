from sapperrag.llm.oai import ChatOpenAI
from backend.core.conf import settings
from backend.app.techmanage.schema.summarysource import GetSummarySourceListDetails
from backend.app.techmanage.service.summarysource_service import summary_source_service
from backend.app.techmanage.schema.project import GetProjectListDetails
from backend.app.techmanage.service.project_service import project_service
from backend.app.techmanage.service.embedding_summarysource_service import embedding_summary_source_service
from backend.utils.serializers import select_as_dict
from langdetect import detect
from backend.common.exception import errors
import re
import json
from typing import Optional
import logging


class ProjectAnalysisService:
    def __init__(self, project_id: int, summarysource_ids: list[int] = None):
        self.chatbot = ChatOpenAI(settings.OPENAI_KEY, settings.OPENAI_BASE_URL)
        self.project_id = project_id
        self.summarysource_ids = summarysource_ids if summarysource_ids is not [] else []
        self.project_analysis_dict = {}

    async def initialize_project(self):
        if not self.summarysource_ids:
            project_info = await project_service.get_with_source(pk=self.project_id)
            project = GetProjectListDetails(**select_as_dict(project_info))
            self.summarysource_ids = [summarysource.id for summarysource in project.summarysources]
        else:
            project_info = await project_service.get_with_source(pk=self.project_id)
            project = GetProjectListDetails(**select_as_dict(project_info))
            project_summarysource_ids = [summarysource.id for summarysource in project.summarysources]
            # 修改5：过滤非法ID（交集操作）
            self.summarysource_ids = [
                sid for sid in self.summarysource_ids
                if sid in project_summarysource_ids
            ]

    async def build_source_context(self):
        query = "从项目申报书中提取关键信息，包括但不限于：1. 项目总结（概述项目背景、目标及预期成果）；2. 技术要求（项目实施所需的关键技术和规范）；3. 研究目标（核心研究方向和预期突破）；4. 研究领域（涉及的学科或应用方向）；5. 关键词（能概括项目核心内容的关键词）；6. 核心技术（项目依赖或创新的关键技术）；7. 技术难点（研究过程中需要解决的主要技术挑战）；8. 创新点（项目在技术、方法或应用上的创新贡献）；9. 工具与平台（研究或开发过程中使用的软件、硬件或实验环境）。请确保检索出的内容准确、完整，并便于技术分析和申报材料整理。"
        embeddings = await embedding_summary_source_service.get_embeddings_by_query(
            summarysource_ids=self.summarysource_ids,
            query=query, topK=20)
        system = ""
        index = 1
        for emb in embeddings:
            system += f"这是第{index}块\n" + emb.content + "\n"
            index += 1
        return system

    @staticmethod
    async def type_change(answer: str):
        """
                将包含JSON的字符串转换为Python对象，处理常见的代码块标记和格式问题。

                Args:
                    answer: 包含JSON的字符串，可能被代码块标记包围

                Returns:
                    解析后的Python对象，如果解析失败则返回None

                Raises:
                    ValueError: 如果输入的字符串无法被清理为有效的JSON格式
                """
        if not isinstance(answer, str):
            logging.error(f"type_change 函数收到了非字符串类型的输入: {type(answer)}")
            return None

        try:
            # 移除常见的Markdown代码块标记
            cleaned_text = answer.strip()

            # 检查是否包含标准的三重反引号代码块
            if cleaned_text.startswith("```") and cleaned_text.endswith("```"):
                # 移除开头的```和可能的语言标识（如```json）
                lines = cleaned_text.split('\n')
                content_lines = lines[1:-1]  # 去掉第一行和最后一行
                cleaned_text = '\n'.join(content_lines).strip()

            # 移除任何剩余的json标签（如开头的json或结尾的json）
            cleaned_text = cleaned_text.replace("```json", "").replace("json```", "").strip()

            # 检查处理后的字符串是否为空
            if not cleaned_text:
                logging.warning("处理后的JSON字符串为空")
                return None

            # 尝试解析JSON
            return json.loads(cleaned_text)

        except json.JSONDecodeError as e:
            # 记录详细的错误信息，帮助调试
            logging.error(f"JSON解析失败: {str(e)}")
            logging.error(f"原始输入: {answer}")
            logging.error(f"处理后的文本: {cleaned_text}")
            return None
        except Exception as e:
            logging.error(f"type_change 函数发生未知错误: {str(e)}")
            return None

    async def analyze_project(self):
        project_data = await self.build_source_context()
        prompt = f"""
        作为资深技术文档分析师，您将根据项目切片内容进行结构化分析。请严格遵循以下格式要求，确保输出数据与数据库字段完全匹配。

        # 输入数据
        项目切片内容：
        {project_data}

        # 角色与目标
        - 角色：技术文档分析师
        - 目标：提取项目关键信息并生成结构化分析报告

        # 分析指南
        1. **subject_type（申报主体类型）**：
           - 识别申报主体类型（企业/高校/研究所/个人等）

        2. **institution_age_limit（单位成立年限）**：
           - 提取单位成立年限要求
           - 示例："2010年及以上"

        3. **leader_education_limit（负责人学历）**：
           - 识别负责人学历要求

        4. **leader_age_limit（负责人年龄）**：
           - 提取负责人年龄上限
           - 示例："30岁以下"

        5. **leader_title_limit（负责人职称）**：
           - 识别负责人职称要求

        6. **research_field（研究领域）**：
           - 提取核心研究领域（单个字符串，逗号分隔）

        7. **keywords（关键词）**：
           - 提取技术关键词（JSON数组）

        8. **research_content（研究内容）**：
           - 详细描述项目研究内容和技术方案
           - 包含：技术路线、方法论、实验设计等

        9. **technical_indicators（技术指标）**：
           - 列出量化技术指标

        10. **expected_achievements（预期成果）**：
            - 描述项目预期产出成果
            - 包含：专利、论文、产品、技术突破等

        11. **project_type（项目类型）**：
            - 识别资助项目类型
            - 示例："国家自然科学基金"

        12. **funding_amount（资助金额）**：
            - 提取资助金额范围（字符串）
            - 示例："80-120万元"

        13. **application_deadline（申报截止）**：
            - 提取申报截止日期（YYYY-MM-DD格式）
            - 示例："2025-09-30"

        # 输出要求
        - 格式：严格遵循JSON结构
        - 空值处理：无信息字段输出null
        - 数据类型：
          • 数值字段：整数或null
          • 字符串字段：精炼描述
          • keywords：字符串数组
        - 专业要求：
          • 技术描述量化具体（含性能指标）
          • 研究领域不超过100字符
          • 研究内容/成果需专业详实
        -详细说明

        # 响应格式
        {{
          "subject_type": "string | null",
          "institution_age_limit": "string | null",
          "leader_education_limit": "string | null",
          "leader_age_limit": "string | null",
          "leader_title_limit": "string | null",
          "research_field": "string | null",
          "keywords": ["关键词1", "关键词2",...] | null,
          "research_content": "string | null",
          "technical_indicators": "string | null",
          "expected_achievements": "string | null",
          "project_type": "string | null",
          "funding_amount": "string | null",
          "application_deadline": "string | null"
        }}
        """
        messages = [
            {"role": "system", "content": prompt},

        ]

        return self.chatbot.generate(messages)

    async def analyze_project_dict(self):
        await self.initialize_project()
        if self.summarysource_ids:
            project_data = await self.analyze_project()
            project_dict = await self.type_change(project_data)
            self.project_analysis_dict = project_dict

