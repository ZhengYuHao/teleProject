import os
from typing import Optional

def load_prompt(agent_type: str) -> str:
    """
    根据智能体类型加载对应的提示词文件
    
    Args:
        agent_type: 智能体类型
    
    Returns:
        str: 提示词内容
    """
    prompt_file_path = os.path.join("prompts", f"{agent_type}_prompt.txt")
    
    # 如果特定类型的提示词文件不存在，则使用默认提示词
    if not os.path.exists(prompt_file_path):
        return get_default_prompt(agent_type)
    
    # 读取提示词文件内容
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"读取提示词文件失败 {prompt_file_path}: {e}")
        return get_default_prompt(agent_type)

def get_default_prompt(agent_type: str) -> str:
    """
    获取默认提示词
    
    Args:
        agent_type: 智能体类型
    
    Returns:
        str: 默认提示词内容
    """
    default_prompts = {
        "teacher": """你是一名经验丰富的教师，请根据提供的教学内容，从专业角度分析并生成标签。
需要考虑教学目标、内容组织、教学方法、师生互动等方面。
请输出5-10个最具代表性的标签。""",
        
        "student": """你是一名学生，请根据提供的教学内容，从学习者角度分析并生成标签。
考虑内容的易懂性、趣味性、学习效果等方面。
请输出5-10个最符合你感受的标签。""",
        
        "parent": """你是一名家长，请根据提供的教学内容，从家长角度分析并生成标签。
关注内容的教育意义、安全性、适宜性等方面。
请输出5-10个你最关心的标签。""",
        
        "curriculum_expert": """你是一名课程专家，请根据提供的教学内容，从课程设计角度分析并生成标签。
关注课程结构、教学标准符合性、目标达成度等方面。
请输出5-10个专业的标签。"""
    }
    
    return default_prompts.get(agent_type, f"你是一个{agent_type}，请根据提供的教学内容生成5-10个标签。")