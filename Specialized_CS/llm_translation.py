from langchain_core.language_models import BaseChatModel
from typing import List, Dict, Type, Optional, Union



def translate_to_english(text: str, llm: BaseChatModel) -> str:
    return llm.invoke(
        "Translate the following text to English without adding any notes:\n\n" + text
    ).content


def translate_to_persian(text: Optional[Union[list, str, dict]], llm: BaseChatModel) -> str:
    return llm.invoke(
        "Translate the following text to Persian without adding any notes.\n" +
        "Do not convert dates to Jalali calander and keep times in GMT.\n" +
        "Do not say anything before or after the translated text.\n" +
        f"Here is the text to translate: {text}"
    ).content
