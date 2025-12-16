import re
import numpy as np
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

# -------------------------------------------
# Create a common base model class
# -------------------------------------------
class BaseModelWithConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra="ignore"
    )

LABEL_PATTERN = re.compile(r"[^\w]+", flags=re.UNICODE)  # Keep Unicode word chars (incl. Chinese)
NAME_PATTERN  = re.compile(r'[_"\-]+')       # For cleaning name underscores, quotes, dashes

class EntityProperties(BaseModelWithConfig):
    embeddings: Optional[np.ndarray] = None

# -------------------------------------------
# Entity model
# -------------------------------------------
class Entity(BaseModelWithConfig):
    label: str = ""
    name: str  = ""
    properties: EntityProperties = Field(default_factory=EntityProperties)

    def process(self) -> "Entity":
        """
        Normalize `label` and `name` in-place and return self.
        """
        cleaned_label = LABEL_PATTERN.sub("_", self.label).replace("&", "and")
        cleaned_label = re.sub(r"_+", "_", cleaned_label).strip("_")
        self.label = cleaned_label.lower()
        n = self.name.lower()
        n = NAME_PATTERN.sub(" ", n)
        self.name = n.strip()
        return self

    def __eq__(self, other) -> bool:
        if isinstance(other, Entity):
            return self.name == other.name and self.label == other.label
        return False

    def __hash__(self) -> int:
        return hash((self.name, self.label))

    def __repr__(self) -> str:
        return f"Entity(name={self.name!r}, label={self.label!r}, properties={self.properties!r})"
