from typing import Dict, List
from prompt_manager import load_prompt
import asyncio
import json
import os
from openai import OpenAI
from config import Config

async def generate_tags_for_agent(data: Dict, agent_type: str) -> List[str]:
    """
    为特定类型的智能体生成标签
    
    Args:
        data: 输入的数据
        agent_type: 智能体类型（如 teacher, student, parent 等）
    
    Returns:
        List[str]: 生成的标签列表
    """
    # 加载对应智能体的提示词
    prompt = load_prompt(agent_type)
    
    # 调用实际的大模型API来生成标签
    tags = await call_qwen_api(prompt, data, agent_type)
    
    return tags

async def call_qwen_api(prompt: str, data: Dict, agent_type: str) -> List[str]:
    """
    使用OpenAI兼容接口调用Qwen大模型API生成标签
    
    Args:
        prompt: 提示词内容
        data: 输入数据
        agent_type: 智能体类型
    
    Returns:
        List[str]: 生成的标签列表
    """
    # 处理上传文件数据，避免序列化错误
    processed_data = {}
    for key, value in data.items():
        print(f"系统key----{key},value----{value}")
        if value is None:
            processed_data[key] = None
        elif isinstance(value, list):
            # 处理文件列表（如image_files）
            processed_files = []
            for f in value:
                if hasattr(f, 'filename'):
                    try:
                        # 读取文件内容
                        content = await f.read()
                        file_info = {
                            "filename": getattr(f, 'filename', 'unknown'),
                            "content_type": getattr(f, 'content_type', 'unknown'),
                            "size": len(content) if content else 0,
                            "content_preview": content[:1000].decode('utf-8', errors='ignore') if content else ""
                        }
                        processed_files.append(file_info)
                        # 重置文件指针
                        await f.seek(0)
                    except Exception as e:
                        # 如果读取失败，回退到原来的处理方式
                        file_info = {
                            "filename": getattr(f, 'filename', 'unknown'),
                            "content_type": getattr(f, 'content_type', 'unknown'),
                            "error": str(e)
                        }
                        processed_files.append(file_info)
                else:
                    processed_files.append(f)
            processed_data[key] = processed_files
        elif hasattr(value, 'filename'):
            # 处理单个文件（如video_file, subtitle_file等）
            try:
                # 读取文件内容
                content = await value.read()
                processed_data[key] = {
                    "filename": value.filename,
                    "content_type": getattr(value, 'content_type', 'unknown'),
                    "size": len(content) if content else 0,
                    "content_preview": content[:9000].decode('utf-8', errors='ignore') if content else ""
                }
                # 重置文件指针
                await value.seek(0)
            except Exception as e:
                # 如果读取失败，回退到原来的处理方式
                processed_data[key] = {
                    "filename": value.filename,
                    "content_type": getattr(value, 'content_type', 'unknown'),
                    "error": str(e)
                }
        else:
            processed_data[key] = value
    print(f"系统数据_processed_data{processed_data}")
   
    # 构造完整的提示词
    full_prompt = f"{prompt}\n\n输入数据：{json.dumps(processed_data, ensure_ascii=False, indent=2)}\n\n请根据以上信息生成适合{agent_type}的标签。"
    print(f"系统数据_full_prompt{full_prompt}")
    # 准备API请求
    api_base = os.getenv("LLM_API_BASE", "http://106.227.68.83:8000/v1")
    api_key = os.getenv("LLM_API_KEY", "dummy-key")  # Qwen API可能不需要有效的API密钥
    model_name = os.getenv("LLM_MODEL", "qwen2.5-32b")
    
    client = OpenAI(
        base_url=api_base,
        api_key=api_key
    )
    
    try:
        # 发送API请求
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=8192
            )
        )
        
        # 解析响应
        content = response.choices[0].message.content
        
        # 检查content是否为None
        if content is None:
            content = ""
        
        # 尝试解析返回的标签列表
        try:
            # 如果返回的是JSON格式的标签列表
            tags = json.loads(content)
            if isinstance(tags, list):
                return [str(tag) for tag in tags]
        except:
            # 如果返回的是普通文本，按行分割
            tags = [tag.strip() for tag in content.split('\n') if tag.strip()]
            if tags:
                return tags[:10]  # 最多返回10个标签
        
        # 默认返回
        return ["标签生成成功"]
            
    except Exception as e:
        print(f"调用Qwen API时出错: {str(e)}")
        # 出错时返回默认标签
        default_tags = {
            "teacher": ["教学目标明确", "内容组织合理", "师生互动良好", "知识点讲解清晰"],
            "student": ["内容有趣", "易于理解", "节奏适中", "重点突出"],
            "parent": ["教育意义强", "内容健康", "适合学习", "启发思考"],
            "curriculum_expert": ["课程结构完整", "符合教学标准", "目标达成度高"]
        }
        return default_tags.get(agent_type, [f"{agent_type}_tag_1", f"{agent_type}_tag_2"])

async def simulate_llm_call(prompt: str, data: Dict, agent_type: str) -> List[str]:
    """
    模拟大模型调用过程（实际项目中需要替换为真实的API调用）
    
    Args:
        prompt: 提示词内容
        data: 输入数据
        agent_type: 智能体类型
    
    Returns:
        List[str]: 生成的标签列表
    """
    # 模拟异步调用延迟
    await asyncio.sleep(0.1)
    
    # 根据智能体类型生成不同的示例标签
    tag_mapping = {
        "teacher": ["教学目标明确", "内容组织合理", "师生互动良好", "知识点讲解清晰"],
        "student": ["内容有趣", "易于理解", "节奏适中", "重点突出"],
        "parent": ["教育意义强", "内容健康", "适合学习", "启发思考"],
        "curriculum_expert": ["课程结构完整", "符合教学标准", "目标达成度高"]
    }
    
    # 实际应用中，这里会使用prompt和data通过大模型API生成真实标签
    return tag_mapping.get(agent_type, [f"{agent_type}_tag_1", f"{agent_type}_tag_2"])