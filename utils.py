from pydantic import BaseModel
from typing import List, Dict, Any, Union, Optional, Literal
from llama_index.multi_modal_llms.openai import OpenAIMultiModal


class IdentityCard(BaseModel):
    """Data model for a user's identity card."""

    full_name: Optional[str]
    sex: Literal["Nam", "Nữ", "Khác"]
    address: Optional[str]
    date_of_birth: Optional[str]


prompt_template_str = """\
Use the attached IdentityCard image to extract data from it and store into the
provided data class. Always answer in the same language as the general one in document. Try all the best to extract all the information from the image.
"""
gpt_4o = OpenAIMultiModal(
    model="gpt-4o",
    temperature=0.0,
)
