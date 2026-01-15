from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices
from pathlib import Path


class Parameters(BaseSettings):
    """CLI parameters for the overview tool"""
    input_dir: Path = Field(..., description="Directory containing LAZ/LAS files to process", alias=AliasChoices("input-dir", "input_dir"))
    max_total_points: int = Field(50_000_000, alias=AliasChoices("max-total-points", "max_total_points"), description="Maximum total points when aggregating. Points are sampled proportionally from each file.")
    section_width: int = Field(10, alias=AliasChoices("section-width", "section_width"))
    image_width: int = Field(1024, alias=AliasChoices("image-width", "image_width"))
    image_height: int = Field(768, alias=AliasChoices("image-height", "image_height"))
    top_views_deg: int = Field(180, alias=AliasChoices("top-views-deg", "top_views_deg"), description="The angular width of each top view rotation around the LAZ center in degrees.")
    cmap: str = Field("viridis", description="The colormap to use for the point cloud. All 'cmap' options are available in matplotlib.pyplot.get_cmap()")
    camera_distance: float = Field(35, alias=AliasChoices("camera-distance", "camera_distance"), description="The distance of the camera from the center of the point cloud.")
    output_dir: Path = Field("/out", alias=AliasChoices("output-dir", "output_dir"), description="The directory to save the output images.")

    model_config = SettingsConfigDict(
        case_sensitive=False,
        cli_parse_args=True,
        cli_ignore_unknown_args=True
    )
