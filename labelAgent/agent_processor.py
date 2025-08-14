from typing import Dict, List
import asyncio
from tag_generator import generate_tags_for_agent

async def process_data_with_agents(data: Dict, agent_types: List[str]) -> Dict:
    """
    并发处理不同类型智能体的标签生成任务
    
    Args:
        data: 输入的数据，包含视频、字幕、图像等
        agent_types: 智能体类型列表
    
    Returns:
        dict: 每个智能体类型对应的标签列表
    """
    # 创建所有智能体任务
    tasks = []
    for agent_type in agent_types:
        task = generate_tags_for_agent(data, agent_type)
        tasks.append(task)
    
    # 并发执行所有任务
    results = await asyncio.gather(*tasks)
    
    # 组织结果
    final_result = {}
    for agent_type, tags in zip(agent_types, results):
        final_result[agent_type] = tags
    # print(f"final_result {final_result}")
    return {"tags": final_result}