"""
Advanced QA Prompt Module - Supports Batch Question Input and Enhanced Input with Annotations

New Features:
1. Batch Question Mode: Combine multiple questions into one Prompt
2. Enhanced Input Mode: Provide annotations as context

All prompts are in English.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass


class QABatchPromptTemplates:
    """
    Batch QA Prompt Templates
    
    Combine multiple questions into one Prompt, one model call to get all answers.
    """
    
    # Batch by difficulty
    BATCH_BY_DIFFICULTY = """Watch the video and answer the following {num} questions.

You must output answers in strict JSON format, no other content:
```json
[
  {{"question_id": "ACTUAL_QUESTION_ID_1", "answer": "YOUR_ANSWER"}},
  {{"question_id": "ACTUAL_QUESTION_ID_2", "answer": "YOUR_ANSWER"}}
]
```

IMPORTANT: Replace ACTUAL_QUESTION_ID_X with the actual question IDs from the questions below (e.g., task136_action_001).

Questions:
{questions}

Requirements:
1. Use correct format for each answer
2. For multiple choice, output only the letter (e.g., A, B)
3. For true/false, output "True" or "False"
4. For fill-in-blank, output the answer directly
5. For short answer, summarize in one sentence"""
    
    # Batch by category
    BATCH_BY_CATEGORY = """Watch the video and answer the following {num} questions about {category}.

You must output answers in strict JSON format, no other content:
```json
[
  {{"question_id": "ACTUAL_QUESTION_ID_1", "answer": "YOUR_ANSWER"}},
  {{"question_id": "ACTUAL_QUESTION_ID_2", "answer": "YOUR_ANSWER"}}
]
```

IMPORTANT: Replace ACTUAL_QUESTION_ID_X with the actual question IDs from the questions below (e.g., task136_action_001).

Questions:
{questions}

Ensure correct answer format."""
    
    # All questions in one prompt
    BATCH_ALL = """Watch the video and answer all questions at once.

You must output answers in strict JSON format, no other content:
```json
[
  {{"question_id": "ACTUAL_QUESTION_ID_1", "answer": "YOUR_ANSWER"}},
  {{"question_id": "ACTUAL_QUESTION_ID_2", "answer": "YOUR_ANSWER"}}
]
```

CRITICAL: Use the EXACT question_id from the questions below (e.g., task136_action_001, task136_bite_001). Do NOT use placeholder text.

Questions:
{questions}

Notes:
- Multiple choice: output only letter (e.g., A, B, C)
- Multiple select: comma separated (e.g., A,B,C)
- True/False: output "True" or "False"
- Fill in blank: output number or text directly
- Short answer: answer concisely"""

    @classmethod
    def build_batch_prompt(
        cls,
        questions: List[Dict[str, Any]],
        mode: str = "all"
    ) -> str:
        """
        Build batch question Prompt
        
        Args:
            questions: List of questions, each containing id, type, question, options, etc.
            mode: Grouping mode, can be "all", "difficulty", or "category"
        
        Returns:
            Formatted Prompt
        """
        if mode == "difficulty":
            # Group by difficulty
            questions_by_diff = {}
            for q in questions:
                diff = q.get('difficulty', 'L1')
                if diff not in questions_by_diff:
                    questions_by_diff[diff] = []
                questions_by_diff[diff].append(q)
            
            # Concatenate all questions
            all_questions_text = ""
            for diff in ['L1', 'L2', 'L3']:
                if diff in questions_by_diff:
                    qs = questions_by_diff[diff]
                    all_questions_text += f"\n### Difficulty {diff} ({len(qs)} questions)\n"
                    for i, q in enumerate(qs, 1):
                        all_questions_text += f"{i}. [{q['id']}] {q['question']}\n"
                        if q.get('options'):
                            for opt in q['options']:
                                all_questions_text += f"   {opt}\n"
            
            return cls.BATCH_BY_DIFFICULTY.format(
                num=len(questions),
                questions=all_questions_text
            )
        
        elif mode == "category":
            # Group by category
            questions_by_cat = {}
            for q in questions:
                cat = q.get('category', 'other')
                if cat not in questions_by_cat:
                    questions_by_cat[cat] = []
                questions_by_cat[cat].append(q)
            
            all_questions_text = ""
            for cat in questions_by_cat:
                qs = questions_by_cat[cat]
                all_questions_text += f"\n### {cat} ({len(qs)} questions)\n"
                for i, q in enumerate(qs, 1):
                    all_questions_text += f"{i}. [{q['id']}] {q['question']}\n"
                    if q.get('options'):
                        for opt in q['options']:
                            all_questions_text += f"   {opt}\n"
            
            return cls.BATCH_BY_CATEGORY.format(
                num=len(questions),
                category=list(questions_by_cat.keys())[0] if questions_by_cat else "video",
                questions=all_questions_text
            )
        
        else:
            # All questions
            questions_text = ""
            for i, q in enumerate(questions, 1):
                questions_text += f"{i}. [{q['id']}] {q['question']}\n"
                if q.get('options'):
                    for opt in q['options']:
                        questions_text += f"   {opt}\n"
            
            return cls.BATCH_ALL.format(
                questions=questions_text
            )


class QAEnhancedPromptTemplates:
    """
    Enhanced Input Prompt Templates - Provide Annotations as Context
    
    Provide pre-computed annotation results in the Prompt to help the model answer better.
    """
    
    # Basic annotation context template
    BASE_CONTEXT = """Reference Annotations:
- Video Duration: {duration}
- Total Bites: {num_bites} times
- Bite Frequency: {bites_per_minute} times/minute
- Dominant Hand: {dominant_hand}
- Food Composition: {composition}
- Eating Pace: {pace}"""
    
    # Full annotation context template
    FULL_CONTEXT = """Video Annotations:

1. Basic Information:
{duration_str}

2. Bite Events (total {num_bites}):
{bites_summary}

3. Food Detection:
{composition_summary}

4. Eating Pace:
{pace_summary}

5. Facial Expression Sequence:
{expression_summary}"""
    
    @classmethod
    def build_context_from_annotations(
        cls,
        annotation_data: Dict[str, Any],
        level: str = "basic"
    ) -> str:
        """
        Build context from annotation data
        
        Args:
            annotation_data: Annotation data dictionary
            level: Context detail level, can be "basic" or "full"
        
        Returns:
            Formatted context string
        """
        if level == "basic":
            # Basic context
            duration = annotation_data.get('metadata', {}).get('duration_str', 'Unknown')
            bites = annotation_data.get('bite', {})
            pace = annotation_data.get('pace', {})
            composition = annotation_data.get('composition', {})
            
            # Bite statistics
            num_bites = bites.get('num_bites', 0)
            stats = bites.get('statistics', {})
            bites_per_min = stats.get('bites_per_minute', 0)
            
            # Dominant hand
            left = stats.get('left_hand_bites', 0)
            right = stats.get('right_hand_bites', 0)
            if left > right:
                dominant_hand = "Left hand"
            elif right > left:
                dominant_hand = "Right hand"
            else:
                dominant_hand = "Alternating hands"
            
            # Food composition
            comp_by_area = composition.get('composition_by_area', {})
            comp_str = ", ".join([
                f"{k}: {v.get('percentage', 0):.1f}%"
                for k, v in comp_by_area.items()
                if v.get('percentage', 0) > 0
            ])
            
            # Eating pace
            pace_level = pace.get('overall_rate', {}).get('bites_per_minute', 0)
            if pace_level < 6:
                pace_str = "Slow"
            elif pace_level < 10:
                pace_str = "Moderate"
            else:
                pace_str = "Fast"
            
            return cls.BASE_CONTEXT.format(
                duration=duration,
                num_bites=num_bites,
                bites_per_minute=round(bites_per_min, 1),
                dominant_hand=dominant_hand,
                composition=comp_str,
                pace=pace_str
            )
        
        else:
            # Full context (simplified, extract key information)
            duration_str = annotation_data.get('metadata', {}).get('duration_str', 'Unknown')
            
            # Bite summary
            bites = annotation_data.get('bite', {})
            num_bites = bites.get('num_bites', 0)
            bite_events = bites.get('bite_events', [])[:5]  # Only first 5
            bites_summary = f"{num_bites} bite events in total"
            if bite_events:
                times = [f"{b['timestamp']:.1f}s" for b in bite_events[:3]]
                bites_summary += f", first few at {', '.join(times)}..."
            
            # Food composition summary
            comp = annotation_data.get('composition', {})
            comp_by_area = comp.get('composition_by_area', {})
            comp_summary = ", ".join([
                f"{k}: {v.get('percentage', 0):.1f}%"
                for k, v in sorted(comp_by_area.items(), key=lambda x: x[1].get('percentage', 0), reverse=True)[:3]
            ])
            
            # Eating pace summary
            pace = annotation_data.get('pace', {})
            overall = pace.get('overall_rate', {})
            pace_summary = f"Average {overall.get('bites_per_minute', 0):.1f} bites/minute"
            
            # Expression summary
            expr = annotation_data.get('facial_expression', {})
            expr_seq = expr.get('facial_expression_timeline', [])
            if expr_seq:
                first_expr = expr_seq[0].get('description', 'Unknown')
                expr_summary = f"Main expression: {first_expr}"
            else:
                expr_summary = "No expression data"
            
            return cls.FULL_CONTEXT.format(
                duration_str=duration_str,
                num_bites=num_bites,
                bites_summary=bites_summary,
                composition_summary=comp_summary,
                pace_summary=pace_summary,
                expression_summary=expr_summary
            )
    
    @classmethod
    def build_enhanced_prompt(
        cls,
        question: str,
        annotation_context: str,
        question_type: str = "single_choice",
        options: Optional[List[str]] = None
    ) -> str:
        """
        Build Enhanced Prompt - Question + Annotation Context
        
        Args:
            question: Question content
            annotation_context: Annotation context
            question_type: Question type
            options: Option list
        
        Returns:
            Formatted Prompt
        """
        if question_type == "single_choice" and options:
            options_text = "\n".join(options)
            return f"""Watch the video and answer the question using the reference information below.

{annotation_context}

Question: {question}

Options:
{options_text}

Output only the correct letter (e.g., A, B, C, D), no explanation."""
        
        elif question_type == "true_false":
            return f"""Watch the video and determine if the following statement is true or false using the reference information.

{annotation_context}

Statement: {question}

Output only "True" or "False"."""
        
        elif question_type == "fill_blank":
            return f"""Watch the video and fill in the blank using the reference information.

{annotation_context}

Question: {question}

Output the answer directly."""
        
        elif question_type == "short_answer":
            return f"""Watch the video and answer the question using the reference information.

{annotation_context}

Question: {question}

Provide a detailed answer."""
        
        else:
            return f"""Watch the video and answer the question using the reference information.

{annotation_context}

Question: {question}"""


# Export
__all__ = [
    'QABatchPromptTemplates',
    'QAEnhancedPromptTemplates'
]
