from sapperrag.llm.oai import ChatOpenAI
from backend.core.conf import settings
from backend.app.techmanage.schema.profilesource import CreateProfileSourceParam, UpdateProfileSourceParam, \
    GetProfileSourceListDetails
from backend.app.techmanage.schema.embedding_profilesource import GetEmbeddingProfileSourceDetails
from backend.app.techmanage.service.profilesource_service import profile_source_service
from backend.app.techmanage.service.paper_service import paper_service
from backend.app.techmanage.schema.paper import CreatePaperParam, GetPaperListDetails
from backend.app.techmanage.service.embedding_profilesource_service import embedding_profile_source_service
from backend.utils.serializers import select_as_dict
from langdetect import detect
from backend.common.exception import errors
import re
import json
import asyncio
import logging
from typing import Any, Optional


class PaperAnalysisService:
    def __init__(self, paper_id: int) -> None:
        self.chatbot = ChatOpenAI(settings.OPENAI_KEY, settings.OPENAI_BASE_URL)
        self.paper_id = paper_id
        self.paper = None
        self.profilesource_paper_ids = []
        self.paper_base_info_dict = {}
        self.paper_summary_info_dict = {}

    async def initialize_paper(self):
        self.paper = await paper_service.get_with_sources(pk=self.paper_id)
        profilesources = [GetProfileSourceListDetails(**select_as_dict(profilesource)) for profilesource in
                          self.paper.profilesources]
        for profilesource in profilesources:
            if profilesource.source_type == '论文':
                self.profilesource_paper_ids.append(profilesource.id)

    async def build_paper_basic_information(self):
        paper_info = ""
        for profilesource_id in self.profilesource_paper_ids:
            profilesource = await profile_source_service.get_with_embedding(pk=profilesource_id)
            embedding_profilesources = [GetEmbeddingProfileSourceDetails(**select_as_dict(embedding_profilesource)) for
                                        embedding_profilesource in profilesource.embedding_profile_sources]
            # 前10个与后5个embedding_profilesource
            # 获取前10个元素
            first_three_embedding = embedding_profilesources[:10]
            # 获取后5个元素
            last_three_embedding = embedding_profilesources[-5:]
            for embedding_profilesource in first_three_embedding:
                if embedding_profilesource.content:
                    paper_info += embedding_profilesource.content + " "
            for embedding_profilesource in last_three_embedding:
                if embedding_profilesource.content:
                    paper_info += embedding_profilesource.content + " "
        return paper_info

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

    async def analysis_base_info(self, paper_base_info):

        prompt = f"""
        ---Role---
        You are an expert academic paper analysis assistant with specialized knowledge in extracting metadata while precisely distinguishing main content from references.

        ---Goal---
        Extract the following information from the provided first two and last two pages:

        1. **Paper title** (`title`): Full title exactly as presented
        2. **Authors** (`authors`): List of ALL author names in order of appearance. Preserve special characters and name formatting
        3. **Journal name** (`journal_name`): Full publication name including subtitles
        4. **Journal category** (`journal_category`): 
           - "会议论文" if name contains: "Conference", "Proceedings", "Workshop", "Symposium", "Conf.", "Proc.", "Int'l", "SIG" prefix (e.g., SIGIR)
           - "期刊论文" otherwise
        5. **DOI** (`url`): Complete DOI link if available
        6. **Equal Contributors** (`equal_contributors`): Authors explicitly marked with equal contribution notes
        7. **Corresponding author** (`corresponding_author`): NAMES of authors with correspondence indicators (email, †, ✉, or "Corresponding Author" label)
        8. **Publication info** (`pubdatevolissue`): Combined string in "YYYY年MM月DD日, Vol XX(Iss XX)" format
        9. **Affiliations** (`affiliations`): 
           - List of dictionaries mapping each author to their FULL affiliation string
           - Handle multiple affiliations per author using semicolon separation
           - Preserve institutional hierarchy and department names
        10. **Funding** (`funding`): Complete funding statement including grant numbers

        ---Critical Requirements---
        1. **Strict Extraction Boundaries**:
           - NEVER use references section content
           - Ignore "References", "Bibliography" and subsequent sections
           - Verify context: Affiliation data must come from author footnotes/header/footer

        2. **Data Completeness**:
           - Missing data: Return `null` for non-array fields
           - Empty arrays: Return `[]` for authors/affiliations if none found
           - Partial matches: Only capture explicitly stated information

        3. **Affiliation Special Handling**:
           - Each author must have separate dictionary entry
           - Multi-affiliation format: `{{"Author Name": "Inst1; DeptA; Inst2"}}`
           - Preserve exact institutional naming (including abbreviations in context)
           - Example format: 
                [
                    {{"Zhang San": "Tsinghua University, Computer Science; Beijing AI Medical Institute"}},
                    {{"Li Si": "Peking University Health Science Center"}}
                ]

        4. **Corresponding Author**:
           - Extract NAMES only (not emails)
           - Capture all authors marked with correspondence indicators
           - Match names exactly as they appear in author list

        5. **Output Validation**:
           - Authors and affiliations arrays MUST maintain 1:1 positional correspondence
           - Journal category requires semantic analysis of publication name
           - Reject inferred data without explicit source evidence

        ---Output Format---
        Return STRICT JSON format (NO additional text):
        {{
            "title": "Full Paper Title",
            "authors": ["First Author", "Second Author", ...],
            "journal_name": "Journal/Proceedings Name",
            "journal_category": "会议论文/期刊论文",
            "url": "https://doi.org/xxx.xxx",
            "equal_contributors": ["Contributor1", "Contributor2"],
            "corresponding_author": ["Corresponding Author Name"],
            "pubdatevolissue": "2023年12月31日, Vol 15(Iss 4)",
            "funding": "National Science Foundation Grant #XXXXX",
            "affiliations": [
                {{"Author Name1": "Affiliation1; Affiliation2"}},
                {{"Author Name2": "Single Affiliation"}},
                ...
            ]
        }}

        ---Paper Excerpt---
        {paper_base_info}
        """
        messages = [
            {"role": "system", "content": prompt},

        ]
        return self.chatbot.generate(messages)

    @staticmethod
    async def convert_affiliations(json_data):
        # 提取并处理机构列表，确保唯一性
        affiliations = []
        seen_affiliations = set()

        # 遍历所有作者的机构信息，收集唯一机构
        for entry in json_data['affiliations']:
            aff_str = list(entry.values())[0]
            institutions = aff_str.split('; ')
            for inst in institutions:
                if inst not in seen_affiliations:
                    seen_affiliations.add(inst)
                    affiliations.append(inst)

        # 构建作者到机构索引的映射
        authoraffmaps = []
        for entry in json_data['affiliations']:
            author = list(entry.keys())[0]
            aff_str = entry[author]
            institutions = aff_str.split('; ')
            # 获取每个机构在affiliations中的索引
            indices = [affiliations.index(inst) + 1 for inst in institutions]
            authoraffmaps.append({"name": author, "affmap": indices})

        # 创建新字典，移除原字段并添加新字段
        converted_data = json_data.copy()
        del converted_data['affiliations']
        converted_data['affiliations'] = affiliations
        converted_data['authoraffmaps'] = authoraffmaps

        return converted_data

    async def build_paper_context(self):
        query = """
        Paper title, author information, publication time, 
        publication journal or conference name, keywords,
        abstract content, core research objectives, main methods, research results, interdisciplinary information, 
        technical or methodological innovation description
        """

        embeddings = await embedding_profile_source_service.get_embeddings_by_query(
            profilesource_ids=self.profilesource_paper_ids,
            query=query, topK=20)
        system = ""
        index = 1
        for emb in embeddings:
            system += f"这是第{index}块\n" + emb.content + "\n"
            index += 1
        return system

    async def analyze_paper(self):
        paper_data = await self.build_paper_context()
        prompt = f"""
                             ---Role---  
                             You are a professional academic assistant with exceptional academic writing and data analysis summarization skills.
                             You excel at extracting key information from paper fragments, understanding the core content of research,
                             and generating detailed, comprehensive, accurate, and academically compliant summaries in response to user needs.

                             You possess the following abilities:
                             1.  Deeply analyze the key data of academic papers and extract and expand core information.
                             2.  Supplement necessary general background knowledge to ensure the completeness of the summary.
                             3.  Clearly inform the user when information is insufficient rather than fabricating content.
                             4.  Ensure the generated content in both Chinese and English is logically coherent and natural.

                             ---Goal---  
                             Provide a comprehensive and detailed academic summary of a paper in Chinese and English.
                             Summarize all relevant information from the provided paper fragments, with a particular emphasis on elaborating
                             the abstract, core research objectives, main methods, research results, and descriptions of technical or methodological innovations.
                             Ensure logical coherence and clearly convey the essence of the research in fluent Chinese and English.

                             ---Response Format---  
                             Your response must follow this JSON format:
                            {{
                                 "title_zh": "[提供论文的中文标题，准确描述其研究主题]",
                                 "core_keywords": "[总结该研究的核心关键词，以数组格式输出，例如：[\"人工智能\", \"机器学习\", \"自然语言处理\"]]",
                                 "abstract": "[详细总结论文的摘要，突出研究背景、核心目标、主要发现和结论，确保表达流畅自然]",
                                 "research_subject": "[详细描述本研究所聚焦的核心研究对象及其特征]",
                                 "research_problem": "[详细说明该研究所聚焦的关键问题，包含其学术背景及当前挑战]",
                                 "research_objective_and_significance": "[分析该研究试图解决的问题及其对学术发展和实际应用的推动作用]",
                                 "core_research_methods": "[详细描述研究中所采用的关键技术、实验流程或数据分析模型]",
                                 "methodological": "[详细阐述研究方法的核心逻辑路径，包括关键步骤或理论依据]",
                                 "research_results_and_validation": "[详细说明实验结果，并结合数据分析验证其实际影响或科学意义]"
                            }}
                         If there is insufficient information to answer a question, respond with null instead of fabricating an answer.

                         The data fragments of the paper are as follows:  {paper_data}
                         paper_title:{self.paper.title}
                         Based on the above paper fragments, generate a detailed and comprehensive academic summary in Chinese and English following the specified response format, ensuring logical coherence and clear, natural expression.
                         输出是中文
                        """
        messages = [
            {"role": "system", "content": prompt},

        ]

        return self.chatbot.generate(messages)

    async def analyze_paper_dict(self):
        await self.initialize_paper()
        # 构建论文基本信息
        paper_base_info = await self.build_paper_basic_information()
        base_info = await self.analysis_base_info(paper_base_info)
        paper_base_info_dict_no_convert = await self.type_change(base_info)
        paper_base_info_dict = await self.convert_affiliations(paper_base_info_dict_no_convert)
        if not paper_base_info_dict:
            self.paper_base_info_dict = {}
        else:
            self.paper_base_info_dict = paper_base_info_dict
        # 构建论文总结信息
        paper_summary_info_data = await self.analyze_paper()
        paper_summary_info_dict = await self.type_change(paper_summary_info_data)
        if not paper_summary_info_dict:
            self.paper_summary_info_dict = {}
        else:
            self.paper_summary_info_dict = paper_summary_info_dict


async def main():
    analysis_service = PaperAnalysisService(13)
    await analysis_service.analyze_paper_dict()
    print(analysis_service.paper_base_info_dict)
    print(analysis_service.paper_summary_info_dict)


if __name__ == '__main__':
    asyncio.run(main())
