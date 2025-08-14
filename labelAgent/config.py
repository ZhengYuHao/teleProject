import os

class Config:
    # 项目根目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 提示词文件目录
    PROMPT_DIR = os.path.join(BASE_DIR, "prompts")
    
    # 支持的智能体类型
    SUPPORTED_AGENT_TYPES = [
        "teacher",           # 课任教师智能体
        "student",           # 学生智能体
        "parent",            # 家长智能体
        "curriculum_expert"  # 课程专家智能体
    ]
    
    # API配置
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # 文件上传配置
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    # 大模型API配置（示例）
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_API_BASE = os.getenv("LLM_API_BASE", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")