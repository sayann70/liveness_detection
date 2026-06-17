import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # MySQL Database Settings
    MYSQL_HOST: str = Field(default="localhost")
    MYSQL_PORT: int = Field(default=3306)
    MYSQL_USER: str = Field(default="root")
    MYSQL_PASSWORD: str = Field(default="")
    MYSQL_DB: str = Field(default="employee_activity_db")

    # Camera Settings
    CAMERA_INDEX: int = Field(default=0)
    MOCK_CAMERA: bool = Field(default=False)  # Force Simulated Feed (Mock Camera)
    
    # Computer Vision Settings
    MOVEMENT_THRESHOLD: float = Field(default=0.015)  # Normalized Euclidean threshold
    ABSENT_TIMEOUT: float = Field(default=10.0)      # Seconds before ABSENT
    IDLE_TIMEOUT: float = Field(default=60.0)        # Seconds before IDLE

    # YOLO Settings
    YOLO_MODEL_NAME: str = Field(default="yolov8n-pose.pt")
    YOLO_INFERENCE_SIZE: int = Field(default=320)
    YOLO_CONFIDENCE_THRESHOLD: float = Field(default=0.25)
    YOLO_DEVICE: str = Field(default="cpu")
    YOLO_SMOOTHING_FACTOR: float = Field(default=0.35)
    YOLO_EPSILON_FILTER: float = Field(default=0.0015)
    YOLO_ACTIVITY_THRESHOLD: float = Field(default=0.50)
    YOLO_TRACKER: str = Field(default="bytetrack.yaml")
    YOLO_DEBUG_MODE: bool = Field(default=False)
    
    # Active employee to monitor (for single camera PoC)
    DEFAULT_EMPLOYEE_ID: int = Field(default=1)

    @property
    def database_url(self) -> str:
        # Construct pymysql connection string
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DB}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
