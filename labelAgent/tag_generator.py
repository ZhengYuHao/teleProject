from typing import Dict, List
from prompt_manager import load_prompt
import asyncio
import json
import os
from openai import OpenAI
from config import Config

# 尝试导入tiktoken，如果没有安装则使用字符数估算
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("警告: 未安装tiktoken库，将使用字符数估算token数量")

# 缓存模型信息，避免重复请求
_model_info_cache = {}

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

async def get_model_info(client: OpenAI, model_name: str) -> dict:
    """
    获取模型信息，包括最大上下文长度等
    
    Args:
        client: OpenAI客户端
        model_name: 模型名称
        
    Returns:
        dict: 模型信息
    """
    global _model_info_cache
    
    # 如果缓存中有信息，直接返回
    if model_name in _model_info_cache:
        return _model_info_cache[model_name]
    
    try:
        # 尝试获取模型列表
        loop = asyncio.get_event_loop()
        models = await loop.run_in_executor(None, lambda: client.models.list())
        
        # 查找目标模型信息
        for model in models.data:
            if model.id == model_name:
                _model_info_cache[model_name] = model.dict()
                return _model_info_cache[model_name]
    except Exception as e:
        print(f"获取模型信息时出错: {e}")
    
    # 默认模型信息
    default_info = {
        "id": model_name,
        "object": "model",
        "owned_by": "openai",
        "permission": [],
        "root": model_name,
        "parent": None
    }
    
    # 对于qwen2.5-32b模型，设置默认的上下文长度
    if "qwen2.5-32b" in model_name.lower():
        default_info["max_context_length"] = 32768
    else:
        default_info["max_context_length"] = 8192
    
    _model_info_cache[model_name] = default_info
    return default_info

def count_tokens(text: str) -> int:
    """
    计算文本的token数量
    
    Args:
        text: 输入文本
        
    Returns:
        int: token数量
    """
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            print(f"使用tiktoken计算token时出错: {e}")
    
    # 如果没有tiktoken库，使用字符数估算
    # 粗略估算每个token约4个字符
    return len(text) // 4

def truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    """
    根据token数量截断文本
    
    Args:
        text: 输入文本
        max_tokens: 最大token数量
        
    Returns:
        str: 截断后的文本
    """
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            # 截断到指定token数量
            truncated_tokens = tokens[:max_tokens]
            return encoding.decode(truncated_tokens)
        except Exception as e:
            print(f"使用tiktoken截断文本时出错: {e}")
    
    # 如果没有tiktoken库，使用字符数估算
    # 粗略估算每个token约4个字符
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars]

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
                            "content_preview": content[:-1].decode('utf-8', errors='ignore') if content else ""
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
                    "content_preview": content[:-1].decode('utf-8', errors='ignore') if content else ""
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
    
    # 准备API请求
    api_base = os.getenv("LLM_API_BASE", "http://106.227.68.83:8000/v1")
    api_key = os.getenv("LLM_API_KEY", "dummy-key")  # Qwen API可能不需要有效的API密钥
    model_name = os.getenv("LLM_MODEL", "qwen2.5-32b")
    
    client = OpenAI(
        base_url=api_base,
        api_key=api_key
    )
    
    # 获取模型信息
    model_info = await get_model_info(client, model_name)
    max_context_length = model_info.get("max_context_length", 8192)
    print(f"模型 {model_name} 的最大上下文长度: {max_context_length}")
    
    # 计算预留的输出token数量
    reserved_output_tokens = 1000
    
    # 构造完整的提示词
    data_json = json.dumps(processed_data, ensure_ascii=False, indent=2)
    full_prompt = f"{prompt}\n\n输入数据：{data_json}\n\n请根据以上信息生成适合{agent_type}的标签。"
    
    # 计算当前提示词的token数量
    current_tokens = count_tokens(full_prompt)
    print(f"当前提示词token数量: {current_tokens}")
    
    # 检查是否超过模型上下文限制
    if current_tokens >= max_context_length:
        print("警告: 提示词长度超过模型最大上下文长度，需要进行截断处理")
        
        # 计算可用于输入的最大token数量
        max_input_tokens = max_context_length - reserved_output_tokens
        
        # 首先尝试截断数据部分
        data_tokens = count_tokens(data_json)
        prompt_tokens = count_tokens(prompt)
        other_text_tokens = count_tokens(f"\n\n输入数据：\n\n请根据以上信息生成适合{agent_type}的标签。")
        
        # 计算可用的纯数据token数量
        available_data_tokens = max_input_tokens - prompt_tokens - other_text_tokens
        
        if available_data_tokens > 0:
            # 截断数据部分
            truncated_data_json = truncate_text_by_tokens(data_json, available_data_tokens)
            full_prompt = f"{prompt}\n\n输入数据：{truncated_data_json}\n\n请根据以上信息生成适合{agent_type}的标签。"
            print(f"截断后提示词token数量: {count_tokens(full_prompt)}")
        else:
            # 如果连提示词都放不下，只能截断提示词
            full_prompt = truncate_text_by_tokens(prompt, max_input_tokens)
            print(f"由于空间不足，仅保留提示词，截断后提示词token数量: {count_tokens(full_prompt)}")
    
    print(f"系统数据_full_prompt{full_prompt}")
    
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
                max_tokens=reserved_output_tokens
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