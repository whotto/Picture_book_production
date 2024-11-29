"""
儿童绘本故事生成器
此脚本实现了一个完整的儿童绘本故事生成工作流，包括：
1. 故事生成：使用OpenAI API生成结构化的儿童故事
2. 配图提示词生成：为每个场景生成Flux AI绘图提示词
3. 排版处理：将故事转换为markdown格式并添加词汇解释

工作流程：
1. 配置故事参数（语言、段落长度等）
2. 生成故事内容（包含角色描述和场景描写）
3. 为每个场景生成Flux提示词
4. 格式化故事并添加词汇解释
5. 保存所有内容到文件
"""

import os
import json
import openai
import asyncio
import aiohttp
import fal_client
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
import websockets

# 加载环境变量
load_dotenv()

@dataclass
class StoryConfig:
    """
    故事配置类，使用dataclass来管理故事的基本参数
    
    属性:
        language: 故事语言（默认中文）
        words_per_paragraph: 每段字数（默认68字）
        target_age: 目标年龄段（默认5岁）
        paragraph_count: 段落数量（默认5段）
    """
    language: str = os.getenv("STORY_LANGUAGE", "中文")
    words_per_paragraph: int = int(os.getenv("WORDS_PER_PARAGRAPH", "68"))
    target_age: str = os.getenv("TARGET_AGE", "5岁")
    paragraph_count: int = int(os.getenv("PARAGRAPH_COUNT", "5"))

class StoryGenerator:
    """
    故事生成器类
    负责与OpenAI API交互，生成结构化的儿童故事
    """
    
    def __init__(self):
        """初始化故事生成器，设置系统提示词"""
        # 从环境变量获取模型名称和API配置
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        
        self.system_prompt = """你是一位专业的儿童故事作家。
你的任务是创作适合儿童阅读的有趣故事。

要求：
1. 故事要有教育意义和趣味性
2. 语言要简单易懂，适合目标年龄段
3. 情节要生动有趣，富有想象力
4. 要传递正面的价值观
5. 要符合儿童认知水平
6. 要有清晰的故事结构和情节发展"""

    def generate_story(self, 
                      theme: str,
                      config: StoryConfig,
                      additional_requirements: Optional[str] = None) -> Dict:
        """
        生成儿童故事的核心方法
        
        参数:
            theme: 故事主题
            config: 故事配置对象
            additional_requirements: 额外的故事要求
            
        返回:
            包含完整故事内容的字典，包括标题、角色描述、段落内容等
        """
        try:
            # 构建完整的提示词，包含所有故事生成要求
            prompt = f"""请为{config.target_age}的儿童创作一个关于{theme}的绘本故事。

## 基本要求：
1. 故事语言为{config.language}
2. 每段约{config.words_per_paragraph}字
3. 共{config.paragraph_count}段
4. 适合{config.target_age}儿童阅读

## 故事结构要求：
1. 清晰的开端、发展、高潮、结局
2. 情节连贯，富有想象力
3. 角色形象鲜明
4. 结尾要留有适当的想象空间
5. 确保故事传递积极正面的价值观

## 输出格式：
请将故事内容格式化为以下JSON格式：
{{\\"title\\": \\"故事标题\\",\\"characters\\": [\\"角色1描述\\",\\"角色2描述\\"],\\"paragraphs\\": [\\"第一段内容\\",\\"第二段内容\\"]}}"""

            if additional_requirements:
                prompt += f"\n\n## 额外要求：\n{additional_requirements}"

            # 调用OpenAI API生成故事
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            
            # 获取生成的内容
            content = response.choices[0].message.content.strip()
            
            try:
                # 如果返回的内容被包裹在```json和```中，去掉这些标记
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                # 尝试解析JSON
                story_content = json.loads(content)
                if not isinstance(story_content, dict):
                    raise ValueError("Response is not a dictionary")
                
                # 验证必要的字段
                required_fields = ["title", "characters", "paragraphs"]
                for field in required_fields:
                    if field not in story_content:
                        raise ValueError(f"Missing required field: {field}")
                
                # 验证字段类型
                if not isinstance(story_content["title"], str):
                    raise ValueError("Title must be a string")
                if not isinstance(story_content["characters"], list):
                    raise ValueError("Characters must be a list")
                if not isinstance(story_content["paragraphs"], list):
                    raise ValueError("Paragraphs must be a list")
                
                # 验证内容不为空
                if not story_content["title"].strip():
                    raise ValueError("Title cannot be empty")
                if not story_content["characters"]:
                    raise ValueError("Characters list cannot be empty")
                if not story_content["paragraphs"]:
                    raise ValueError("Paragraphs list cannot be empty")
                
                print(f"成功生成故事：{story_content['title']}")
                return story_content

            except json.JSONDecodeError:
                print(f"JSON解析错误。API返回内容：\n{content}")
                return None
            except ValueError as e:
                print(f"内容格式错误: {str(e)}")
                return None
            
        except Exception as e:
            print(f"生成故事时发生错误: {str(e)}")
            if hasattr(e, 'response'):
                print(f"API响应: {e.response}")
            return None

class FluxPromptGenerator:
    """
    Flux提示词生成器类
    负责为故事场景生成AI绘图提示词
    """
    
    def __init__(self):
        """初始化提示词生成器，设置系统提示词"""
        # 从环境变量获取模型名称和API配置
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        
        # 设置系统提示词
        self.system_prompt = """你是一位专业的儿童绘本插画提示词工程师。
你的任务是为儿童故事场景生成高质量的Flux AI绘图提示词。

## 输出要求：
1. 提示词必须使用英文
2. 提示词要准确描述场景和人物
3. 提示词要符合儿童绘本的温馨可爱风格
4. 输出必须是JSON格式，包含以下字段：
   - Title: 场景标题
   - Positive Prompt: 正向提示词，包含以下内容：
     * 场景描述 (Scene Description)
     * 艺术风格 (Art Style): children's book illustration, digital art, cute, warm
     * 画面质量 (Quality): masterpiece, best quality, highly detailed
     * 光照效果 (Lighting): soft lighting, warm colors
   - Negative Prompt: 负向提示词，用于避免不需要的元素：
     * 通用负向词: nsfw, ugly, duplicate, morbid, mutilated, poorly drawn face
     * 画面控制: blurry, bad anatomy, bad proportions, extra limbs, text, watermark
     * 风格控制: photo-realistic, 3d render, cartoon, anime, sketches

## 示例输出：
{
    "Title": "Bunny in Garden",
    "Positive Prompt": "A cute white bunny sitting in a colorful garden, surrounded by blooming flowers and butterflies, children's book illustration style, digital art, masterpiece, best quality, highly detailed, soft lighting, warm colors, peaceful atmosphere",
    "Negative Prompt": "nsfw, ugly, duplicate, morbid, mutilated, poorly drawn face, blurry, bad anatomy, bad proportions, extra limbs, text, watermark, photo-realistic, 3d render, cartoon, anime, sketches"
}"""

    def generate_prompts(self, title: str, scene: str, main_character: str) -> Dict:
        """
        为场景生成图像提示词
        
        参数:
            title: 故事标题
            scene: 场景描述
            main_character: 主角英文名称
            
        返回:
            包含正向和负向提示词的字典
        """
        try:
            # 构建提示词生成的请求
            prompt = f"""请为以下儿童故事场景生成Flux绘图提示词：

故事标题：{title}
主角：{main_character}
场景描述：{scene}

要求：
1. 正向提示词必须包含场景描述、艺术风格、画面质量和光照效果
2. 负向提示词必须包含所有必要的控制词
3. 确保输出格式为规定的JSON格式
4. 所有提示词必须是英文
5. 风格必须是儿童绘本插画风格"""

            # 调用OpenAI API生成提示词
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            
            # 获取生成的内容
            content = response.choices[0].message.content.strip()
            
            try:
                # 如果返回的内容被包裹在```json和```中，去掉这些标记
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                # 尝试解析JSON
                prompt_content = json.loads(content)
                
                # 验证必要的字段
                required_fields = ["Title", "Positive Prompt", "Negative Prompt"]
                for field in required_fields:
                    if field not in prompt_content:
                        raise ValueError(f"Missing required field: {field}")
                
                # 验证字段类型和内容
                if not isinstance(prompt_content["Title"], str):
                    raise ValueError("Title must be a string")
                if not isinstance(prompt_content["Positive Prompt"], str):
                    raise ValueError("Positive Prompt must be a string")
                if not isinstance(prompt_content["Negative Prompt"], str):
                    raise ValueError("Negative Prompt must be a string")
                
                # 验证内容不为空
                if not prompt_content["Title"].strip():
                    raise ValueError("Title cannot be empty")
                if not prompt_content["Positive Prompt"].strip():
                    raise ValueError("Positive Prompt cannot be empty")
                if not prompt_content["Negative Prompt"].strip():
                    raise ValueError("Negative Prompt cannot be empty")
                
                print(f"成功生成提示词：{prompt_content['Title']}")
                return {
                    "positive_prompt": prompt_content["Positive Prompt"],
                    "negative_prompt": prompt_content["Negative Prompt"]
                }
                
            except json.JSONDecodeError:
                print(f"生成提示词时发生错误: JSON解析错误\nAPI返回内容：\n{content}")
                return None
            except ValueError as e:
                print(f"生成提示词时发生错误: {str(e)}")
                return None
            
        except Exception as e:
            print(f"生成提示词时发生错误: {str(e)}")
            if hasattr(e, 'response'):
                print(f"API响应: {e.response}")
            return None

class FluxImageGenerator:
    """
    Flux图像生成器类
    负责调用Flux API生成图像
    """
    
    def __init__(self, api_key: str):
        """初始化Flux图像生成器
        
        参数:
            api_key: Flux API密钥
        """
        self.api_key = api_key
        os.environ["FAL_KEY"] = api_key
        
        # 从环境变量读取图像生成参数
        width, height = os.getenv("IMAGE_SIZE", "1024x768").split('x')
        self.width = int(width)
        self.height = int(height)
        self.inference_steps = int(os.getenv("INFERENCE_STEPS", "30"))
        self.guidance_scale = float(os.getenv("GUIDANCE_SCALE", "7.5"))
        self.scheduler = os.getenv("SCHEDULER", "DDIM")
        
    async def _generate_image_async(self, 
                                  positive_prompt: str, 
                                  negative_prompt: str,
                                  output_path: str) -> bool:
        """
        异步调用Flux API生成图像
        
        参数:
            positive_prompt: 正向提示词
            negative_prompt: 负向提示词
            output_path: 图像保存路径
            
        返回:
            bool: 是否成功生成图像
        """
        try:
            print(f"\n开始生成图像...")
            print(f"提示词: {positive_prompt}")
            print(f"负向提示词: {negative_prompt}")

            # 设置环境变量
            fal_client.api_key = self.api_key
            
            # 准备API请求参数
            data = {
                "prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "image_size": "landscape_16_9",
                "num_inference_steps": self.inference_steps,
                "guidance_scale": self.guidance_scale,
                "scheduler": self.scheduler.lower(),
                "seed": -1
            }
            
            # 调用Flux API生成图像
            result = await fal_client.subscribe_async("fal-ai/flux/dev", data)

            print(f"\nAPI响应: {result}")

            # 检查API响应结构
            if result and isinstance(result, dict):
                if 'images' in result and isinstance(result['images'], list) and len(result['images']) > 0:
                    image_data = result['images'][0]
                    if isinstance(image_data, dict) and 'url' in image_data:
                        image_url = image_data['url']
                        print(f"获取到图像URL: {image_url}")
                        
                        # 下载图像
                        async with aiohttp.ClientSession() as session:
                            async with session.get(image_url) as img_response:
                                if img_response.status == 200:
                                    # 确保输出目录存在
                                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                                    with open(output_path, 'wb') as f:
                                        f.write(await img_response.read())
                                    print(f"图像已保存到: {output_path}")
                                    return True
                                else:
                                    print(f"下载图像失败: HTTP {img_response.status}")
                    else:
                        print(f"API响应中的图像数据格式不正确: {image_data}")
                else:
                    print(f"API响应中未找到有效的图像列表")
            else:
                print(f"API响应格式不正确: {result}")
            
            return False
            
        except Exception as e:
            print(f"生成图像时发生错误: {str(e)}")
            if hasattr(e, 'response'):
                print(f"API错误响应: {e.response.text if hasattr(e.response, 'text') else e.response}")
            return False
            
    def generate_image(self, 
                      positive_prompt: str, 
                      negative_prompt: str,
                      output_path: str) -> bool:
        """
        同步方式调用Flux API生成图像
        
        参数:
            positive_prompt: 正向提示词
            negative_prompt: 负向提示词
            output_path: 图像保存路径
            
        返回:
            bool: 是否成功生成图像
        """
        try:
            # 获取或创建事件循环
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # 运行异步函数
            result = loop.run_until_complete(
                self._generate_image_async(
                    positive_prompt,
                    negative_prompt,
                    output_path
                )
            )
            return result
            
        except Exception as e:
            print(f"生成图像时发生错误: {str(e)}")
            return False

class StoryFormatter:
    """
    故事格式化器类
    负责将生成的故事转换为markdown格式，并添加词汇解释
    """
    
    def __init__(self):
        """初始化格式化器，设置系统提示词"""
        # 从环境变量获取模型名称和API配置
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        
        self.system_prompt = """Role: 儿童故事绘本排版整理专家

## Profile
- **Author:** 翔宇工作流
- **Description:** 专注于将儿童故事绘本转化为适合儿童阅读的Markdown格式，保持原文内容和图片链接不变，优化排版以提升阅读体验和理解效果。

## Attention
1. **内容完整性：** 不修改故事内容的顺序和文本，确保原始故事的连贯性和完整性。
2. **图片保留：** 不删除任何图片链接及相关内容，确保视觉元素完整呈现。
3. **排版优化：** 仅优化排版，提升文本的可读性，并在故事结尾添加文中难点词汇的尾注，提供词汇的解释和翻译，辅助儿童理解。

## Goals
1. **格式优化：** 将儿童故事绘本的文本排版优化为符合儿童阅读习惯的Markdown格式，确保结构清晰。
2. **内容保持：** 完整保留文本内容、图片链接及所有原始元素，不做任何内容删减或修改。
3. **提升阅读体验：** 设计简洁、友好的版面布局，通过合理的排版和视觉元素，增强儿童的阅读兴趣和理解能力。

## Skills
1. **Markdown专业知识：** 深入掌握Markdown排版规范，能够灵活运用各种Markdown元素，如标题、列表、图片嵌入和代码块等，创建结构化且美观的文档。
2. **语言与词汇处理：** 擅长识别和提取文本中的难点词汇，能够准确理解其上下文含义，并在尾注中提供简明易懂的解释和翻译，帮助儿童扩展词汇量。
3. **儿童友好设计：** 具备设计简洁、直观的排版能力，能够根据儿童的阅读习惯和认知特点，优化文本布局和视觉呈现，确保内容既吸引人又易于理解。
4. **细节审查能力：** 具备高度的细致性，能够仔细检查文本和图片链接的准确性，确保最终输出的文档无误且高质量。

## Constraints
1. **内容不变：** 严格不修改故事文本的内容或顺序，确保所有原始内容完整保留，不做任何删减或调整。
2. **语种一致：** 输出文档必须与输入内容保持相同的语言，不进行任何语言转换或混用。
3. **真实呈现：** 禁止随意编造内容，所有输出内容必须基于用户提供的原始故事文本，确保真实性和一致性。
4. **标准格式：** 确保文档遵循标准的Markdown语法规范，版面设计需符合儿童阅读的视觉需求，保持整洁和易读性。

## Output Format
1. **标题：** 使用`#`标题格式，确保标题清晰且易于识别。
2. **正文：** 段落保持简短，每个段落之间用空行分隔。适当使用无序列表、加粗或斜体突出重点词汇，帮助儿童理解，同时保留所有图片。
3. **分隔符：** 文章与尾注之间使用水平分隔线`---`，清晰区分内容主体与解释部分。
4. **尾注：** 列出难解词汇，词汇后附上简短易懂的翻译或注释，帮助儿童理解故事内容。

## Workflows
1. **读取故事文本**
   - 获取并导入儿童故事文本
   - 确保所有图片链接完整无误
   - 验证文本格式，确保无乱码或缺失内容

2. **进行Markdown格式化**
   - 将章节标题转换为相应的Markdown标题格式（使用`#`）
   - 将段落内容转换为Markdown段落，保持段落简洁
   - 插入图片链接，确保语法正确并图片显示正常
   - 标记对话内容为引用或特定格式，以增强可读性
   - 识别并收集文本中的难点词汇，准备添加到尾注

3. **排版优化**
   - 调整标题层级，确保结构清晰且逻辑分明
   - 使用无序列表、加粗或斜体等Markdown元素突出重点词汇
   - 确保图片位置合理，避免破坏文本流畅性
   - 调整段落间距，提高整体可读性和视觉舒适度
   - 确保使用一致的字体样式和大小，符合儿童阅读习惯

4. **文档检查**
   - 校对文本内容，确保无拼写或语法错误
   - 检查所有图片链接的有效性，确保图片能正确显示
   - 确认尾注内容的准确性和完整性，确保词汇解释清晰易懂
   - 进行最终预览，确保整体布局适合儿童阅读，版面整洁美观"""

    def process_story(self, story_content: Dict, output_dir: Path) -> Optional[str]:
        """
        处理故事内容，生成最终的故事文件
        
        参数:
            story_content: 故事内容字典
            output_dir: 输出目录
            
        返回:
            str: 生成的故事文件路径，如果失败则返回None
        """
        try:
            # 准备输出目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            story_file = output_dir / f"{story_content['title']}_{timestamp}.md"
            
            # 创建图片目录
            images_dir = output_dir.parent / "generated_images"
            images_dir.mkdir(parents=True, exist_ok=True)
            
            # 初始化图片生成器
            image_generator = FluxImageGenerator(os.getenv("FAL_KEY"))
            
            # 初始化提示词生成器
            prompt_generator = FluxPromptGenerator()
            
            # 生成Markdown格式的故事内容
            markdown_content = [
                f"# {story_content['title']}\n",
                "**角色：**"
            ]
            
            # 添加角色描述
            for character in story_content['characters']:
                markdown_content.append(f"- {character}")
            
            markdown_content.append("\n---\n")
            
            # 处理每个段落
            for i, paragraph in enumerate(story_content['paragraphs']):
                # 添加段落内容
                markdown_content.append(paragraph)
                
                # 为段落生成图片
                scene_name = f"scene_{i+1}"
                image_path = images_dir / f"{story_content['title']}_{scene_name}.png"
                
                # 生成图片的提示词
                prompts = prompt_generator.generate_prompts(
                    title=story_content['title'],
                    scene=paragraph,
                    main_character=story_content['characters'][0]
                )
                
                if prompts:
                    # 生成图片
                    success = image_generator.generate_image(
                        positive_prompt=prompts['positive_prompt'],
                        negative_prompt=prompts['negative_prompt'],
                        output_path=str(image_path)
                    )
                    
                    if success:
                        # 使用相对路径添加图片链接
                        rel_path = os.path.relpath(image_path, output_dir)
                        markdown_content.append(f"\n![{scene_name}]({rel_path})\n")
                    else:
                        print(f"生成图片失败: {scene_name}")
                        markdown_content.append(f"\n![{scene_name}](图片生成失败)\n")
                else:
                    print(f"生成提示词失败: {scene_name}")
                    markdown_content.append(f"\n![{scene_name}](提示词生成失败)\n")
            
            # 添加尾注
            markdown_content.extend([
                "\n---\n",
                "**尾注：**\n"
            ])
            
            # 将内容写入文件
            with open(story_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(markdown_content))
            
            print(f"故事已保存到: {story_file}")
            return str(story_file)
            
        except Exception as e:
            print(f"处理故事时发生错误: {str(e)}")
            return None

    def format_story(self, story: Dict, image_links: List[str]) -> str:
        """
        将故事内容格式化为Markdown格式
        
        参数:
            story: 故事内容字典
            image_links: 图片链接列表
            
        返回:
            格式化后的Markdown文本
        """
        try:
            prompt = f"""请将以下故事内容格式化为Markdown格式的文档：

标题：{story["title"]}

角色：
{chr(10).join(story["characters"])}

段落：
{chr(10).join(story["paragraphs"])}

图片链接：
{chr(10).join(image_links)}

要求：
1. 使用Markdown语法
2. 标题使用一级标题
3. 在适当位置插入图片
4. 段落之间要有适当的空行
5. 保持整体排版美观"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )

            return response.choices[0].message.content
            
        except Exception as e:
            print(f"格式化故事时发生错误: {str(e)}")
            return None

    def save_formatted_story(self, formatted_story: str, output_dir: str, title: str):
        """
        保存格式化后的故事到文件
        
        参数:
            formatted_story: markdown格式的故事文本
            output_dir: 输出目录
            title: 故事标题
        """
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 使用时间戳创建唯一的文件名
        filename = f"{title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(output_dir, filename)
        
        # 保存故事到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(formatted_story)

def main():
    """
    主函数：批量读取test.md文件并生成故事
    """
    import os
    from dotenv import load_dotenv
    import time
    from pathlib import Path
    import sys
    
    # 加载环境变量
    load_dotenv()
    
    # 获取API密钥和基础URL
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_base = os.getenv("OPENAI_API_BASE")
    fal_key = os.getenv("FAL_KEY")
    
    # 获取输入文件路径
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "test.md"
    
    # 创建输出目录
    stories_dir = Path("generated_stories")
    images_dir = Path("generated_images")
    stories_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    # 创建故事配置
    config = StoryConfig(
        language="中文",
        target_age="5-8岁",
        words_per_paragraph=100,
        paragraph_count=3
    )
    
    # 初始化各个组件
    story_generator = StoryGenerator()
    prompt_generator = FluxPromptGenerator()
    image_generator = FluxImageGenerator(api_key=fal_key)
    story_formatter = StoryFormatter()
    
    # 读取输入文件
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            themes = [line.strip() for line in f if line.strip()]
        
        print(f"找到 {len(themes)} 个故事主题")
        
        # 为每个主题生成故事
        for i, theme in enumerate(themes, 1):
            print(f"\n正在生成第 {i} 个故事: {theme}")
            
            # 生成故事
            story = story_generator.generate_story(
                theme=theme,
                config=config,
                additional_requirements="故事要富有教育意义，适合儿童阅读"
            )
            
            if story:
                image_links = []
                # 为每个段落生成配图
                if isinstance(story.get('paragraphs', []), list):
                    for j, paragraph in enumerate(story['paragraphs']):
                        # 获取段落内容和场景描述
                        if isinstance(paragraph, dict):
                            content = paragraph.get('paragraph', '')
                            scene = paragraph.get('scene', '')
                        else:
                            content = paragraph
                            scene = content  # 如果没有场景描述，使用段落内容
                        
                        # 生成图片提示词
                        prompts = prompt_generator.generate_prompts(
                            title=story['title'],
                            scene=scene,
                            main_character=story.get('main_character', '')
                        )
                        
                        if prompts:
                            # 生成图片
                            image_path = images_dir / f"{theme}_scene_{j+1}.png"
                            success = image_generator.generate_image(
                                positive_prompt=prompts['positive_prompt'],
                                negative_prompt=prompts['negative_prompt'],
                                output_path=str(image_path)
                            )
                            if success:
                                # 使用相对路径保存图片链接
                                image_links.append(f"../generated_images/{image_path.name}")
                
                # 格式化故事
                formatted_story = story_formatter.format_story(story, image_links)
                if formatted_story:
                    # 保存格式化后的故事
                    story_formatter.save_formatted_story(
                        formatted_story=formatted_story,
                        output_dir=str(stories_dir),
                        title=theme
                    )
                    print(f"故事已保存到: {stories_dir}/{theme}.md")
                else:
                    print(f"格式化故事 '{theme}' 失败")
            else:
                print(f"生成故事 '{theme}' 失败")
            
            # 添加延时以避免API限制
            time.sleep(2)
        
        print("\n所有故事生成完成！")
        print(f"故事文件保存在 {stories_dir} 目录下")
        print(f"图片文件保存在 {images_dir} 目录下")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()
