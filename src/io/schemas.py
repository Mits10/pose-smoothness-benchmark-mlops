from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator

class SequenceMetadata(BaseModel):
    subject_id: str = Field(..., min_length=1)
    task: str = Field(..., min_length=1)
    trial: int = Field(..., ge=1)

class PoseSequence(BaseModel):
    sequence_id: str = Field(..., min_length=1)
    source: Literal["vicon", "xsens", "gopro"]
    fps: float = Field(..., gt=0)
    joints: dict[str, list[list[float]]]
    metadata: SequenceMetadata | None = None

    @model_validator(mode="after")
    def validate_joint_shapes(self) -> "PoseSequence":
        frame_counts = set ()
        dims = set()

        for name, coords in self.joints.items():
            if not coords:
                raise ValueError(f"Joint '{name} has no coordinates.")
            frame_counts.add(len(coords))
            for point in coords:
                dims.add(len(point))
                if len(point) not in (2,3):
                    raise ValueError(f"Joint '{name}' points must have 2 or 3 values.")
                
            if len(frame_counts) > 1:
                raise ValueError("All joints must have the same number of frames.")
            if len(dims) > 1:
                raise ValueError("All joints must use the same functionality.")
            return self