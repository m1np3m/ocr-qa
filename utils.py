from pydantic import BaseModel
from typing import List, Dict, Any, Union, Optional
from llama_index.multi_modal_llms.openai import OpenAIMultiModal


class IdentityCard(BaseModel):
    """Data model for a user's identity card."""

    full_name: Optional[str]
    sex: Optional[str]
    address: Optional[str]
    date_of_birth: Optional[str]


prompt_template_str = """\
Use the attached IdentityCard image to extract data from it and store into the
provided data class.
"""
gpt_4o = OpenAIMultiModal(
    model="gpt-4o",
    temperature=0.0,
)


