from pydantic import BaseModel,Field,field_validator
class TextRequest(BaseModel):
    text:str=Field(...,min_length=1,max_length=1500,description="Text for emotion prediction")

    @field_validator("text")
    @classmethod
    def validate_text(cls,value):

        value=value.strip()

        if not value:
            raise ValueError("Text cannot be empty")
        
        if value.isdigit():
            raise ValueError("Only numbers are not allowed")
        
        return value