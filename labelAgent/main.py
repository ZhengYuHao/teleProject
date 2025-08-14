from fastapi import FastAPI, UploadFile, File, Form
from typing import List, Dict
import os
from agent_processor import process_data_with_agents
from prompt_manager import load_prompt

app = FastAPI(title="标签智能体系统", description="基于大模型的多角色标签生成系统")

# 智能体处理接口
@app.post("/generate_tags")
async def generate_tags(
    agent_types: List[str] = Form(..., description="智能体类型列表，如: teacher, student, parent"),
    video_file: UploadFile = File(None, description="教学视频文件"),
    subtitle_file: UploadFile = File(None, description="字幕文件"),
    image_files: List[UploadFile] = File([], description="图像文件列表"),
    audio_file: UploadFile = File(None, description="音频文件"),
    ppt_file: UploadFile = File(None, description="PPT文件")
):
    """
    根据指定的智能体类型并发处理数据并生成标签
    """
    data = {
        "video": video_file,
        "subtitle": subtitle_file,
        "images": image_files,
        "audio": audio_file,
        "ppt": ppt_file
    }
    
    # 处理数据并生成标签
    result = await process_data_with_agents(data, agent_types)
    
    return result

@app.get("/health")
def health_check():
    """健康检查接口"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)