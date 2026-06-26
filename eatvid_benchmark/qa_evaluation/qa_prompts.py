"""
问答Prompt模板模块

提供多种题型的Prompt模板，支持中英文。
"""

from typing import Optional, List
from dataclasses import dataclass


@dataclass
class QAPromptTemplate:
    """问答Prompt模板"""
    name: str
    template: str
    language: str = "zh"


class QAPromptTemplates:
    """
    问答Prompt模板集合
    
    为不同题型提供标准化的Prompt模板。
    """
    
    # ==================== 单选题 ====================
    SINGLE_CHOICE = QAPromptTemplate(
        name="single_choice",
        template="""Please watch the video and answer the following question.

Question: {question}

Options:
{options}

Please directly output the letter of the correct option (e.g., A, B, C, D) without explanation.""",
        language="en"
    )
    
    # ==================== 多选题 ====================
    MULTI_CHOICE = QAPromptTemplate(
        name="multi_choice",
        template="""Please watch the video and answer the following question.

Question: {question}

Options:
{options}

Please select all correct options, separated by commas (e.g., A,B,C) without explanation.""",
        language="en"
    )
    
    # ==================== 判断题 ====================
    TRUE_FALSE = QAPromptTemplate(
        name="true_false",
        template="""Please watch the video and determine whether the following statement is correct.

Statement: {statement}

Please directly answer "True" or "False" without explanation.""",
        language="en"
    )
    
    # ==================== 填空题 ====================
    FILL_BLANK = QAPromptTemplate(
        name="fill_blank",
        template="""Please watch the video and answer the following question.

Question: {question}

Please directly fill in the answer without explanation. If the answer contains numbers, please write the numbers directly.""",
        language="en"
    )
    
    # ==================== 简答题 ====================
    SHORT_ANSWER = QAPromptTemplate(
        name="short_answer",
        template="""Please watch the video and answer the following question.

Question: {question}

Please answer in detail, including key information points. Your answer should:
1. Directly address the question
2. Include specific details and observations
3. Provide reasonable analysis and suggestions if needed""",
        language="en"
    )
    
    # ==================== 带上下文的模板 ====================
    SINGLE_CHOICE_WITH_CONTEXT = QAPromptTemplate(
        name="single_choice_with_context",
        template="""Please watch the video and answer the following question.

{context}

Question: {question}

Options:
{options}

Please directly output the letter of the correct option (e.g., A, B, C, D) without explanation.""",
        language="en"
    )
    
    SHORT_ANSWER_WITH_HINTS = QAPromptTemplate(
        name="short_answer_with_hints",
        template="""Please watch the video and answer the following question.

Question: {question}

Answer hints:
{hints}

Please answer in detail according to the hints, ensuring all key points are covered.""",
        language="en"
    )
    
    @classmethod
    def get_template(cls, question_type: str) -> QAPromptTemplate:
        """根据题型获取模板"""
        templates = {
            'single_choice': cls.SINGLE_CHOICE,
            'multi_choice': cls.MULTI_CHOICE,
            'true_false': cls.TRUE_FALSE,
            'fill_blank': cls.FILL_BLANK,
            'short_answer': cls.SHORT_ANSWER,
        }
        if question_type not in templates:
            raise ValueError(f"Unknown question type: {question_type}")
        return templates[question_type]
    
    @classmethod
    def format_prompt(
        cls,
        question_type: str,
        question: str,
        options: Optional[List[str]] = None,
        context: Optional[str] = None,
        hints: Optional[List[str]] = None
    ) -> str:
        """
        格式化Prompt
        
        Args:
            question_type: 题型
            question: 问题内容
            options: 选项列表（选择题需要）
            context: 上下文信息
            hints: 回答提示（简答题）
        
        Returns:
            格式化后的Prompt
        """
        template = cls.get_template(question_type)
        
        if question_type in ['single_choice', 'multi_choice']:
            if not options:
                raise ValueError(f"Options required for {question_type}")
            options_text = '\n'.join(options)
            return template.template.format(
                question=question,
                options=options_text
            )
        
        elif question_type == 'true_false':
            return template.template.format(statement=question)
        
        elif question_type == 'fill_blank':
            return template.template.format(question=question)
        
        elif question_type == 'short_answer':
            if hints:
                hints_text = '\n'.join(f"- {h}" for h in hints)
                return cls.SHORT_ANSWER_WITH_HINTS.template.format(
                    question=question,
                    hints=hints_text
                )
            return template.template.format(question=question)
        
        return template.template.format(question=question)
    
    @classmethod
    def format_prompt_with_context(
        cls,
        question_type: str,
        question: str,
        context: str,
        options: Optional[List[str]] = None
    ) -> str:
        """带上下文的Prompt格式化"""
        if question_type == 'single_choice' and options:
            options_text = '\n'.join(options)
            return cls.SINGLE_CHOICE_WITH_CONTEXT.template.format(
                context=context,
                question=question,
                options=options_text
            )
        
        # 其他类型使用普通模板
        return cls.format_prompt(question_type, question, options)


class QAPromptBuilder:
    """
    问答Prompt构建器
    
    提供更灵活的Prompt构建方式。
    """
    
    def __init__(self, language: str = "zh"):
        self.language = language
        self.templates = QAPromptTemplates()
    
    def build(
        self,
        question_type: str,
        question: str,
        **kwargs
    ) -> str:
        """构建Prompt"""
        return self.templates.format_prompt(
            question_type=question_type,
            question=question,
            **kwargs
        )
    
    def build_with_video_context(
        self,
        question_type: str,
        question: str,
        video_summary: Optional[str] = None,
        **kwargs
    ) -> str:
        """带视频上下文摘要的Prompt构建"""
        if video_summary:
            context = f"视频摘要：{video_summary}"
            return self.templates.format_prompt_with_context(
                question_type=question_type,
                question=question,
                context=context,
                **kwargs
            )
        return self.build(question_type, question, **kwargs)
    
    def build_batch(
        self,
        questions: list,
        video_summary: Optional[str] = None
    ) -> list:
        """批量构建Prompt"""
        prompts = []
        for q in questions:
            prompt = self.build_with_video_context(
                question_type=q.question_type,
                question=q.question,
                video_summary=video_summary,
                options=q.options,
                hints=q.key_points if q.question_type == 'short_answer' else None
            )
            prompts.append({
                'question_id': q.question_id,
                'prompt': prompt
            })
        return prompts