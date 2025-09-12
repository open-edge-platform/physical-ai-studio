from typing import List
import os
from pathlib import Path
from json.decoder import JSONDecodeError

from lerobot.datasets.utils import write_jsonlines, load_jsonlines
from schemas import ProjectConfig

default_home = os.path.join(os.path.expanduser("~"), ".cache")
GETI_ACTION_HOME = Path(os.path.expandvars(
    os.path.expanduser(
        os.getenv(
            "GETI_ACTION_HOME",
            os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "geti_action"),
        )
    )
))

#ROBOTS_PATH = "robots.json"
PROJECTS_PATH = "projects.jsonl"

def write_projects(data: List[ProjectConfig]):
   write_jsonlines([row.model_dump() for row in data], GETI_ACTION_HOME / PROJECTS_PATH)

def write_project(data: ProjectConfig):
   existing_projects = load_projects()
   already_exists = len([project for project in existing_projects if project.id == data.id]) > 0
   if already_exists:
      projects = [data if project.id == data.id else project for project in existing_projects]
      write_projects(projects)
   else:
      projects = existing_projects + [data]
      write_projects(projects)

def load_projects() -> List[ProjectConfig]:
   fpath = (GETI_ACTION_HOME / PROJECTS_PATH)
   try:
      return [ProjectConfig(**data) for data in load_jsonlines(fpath)]
   except (FileNotFoundError, JSONDecodeError):
      write_jsonlines([], fpath)
      return []

def load_project(project_id: str) -> ProjectConfig | None:
    projects = load_projects()
    project = next((project for project in projects if project.id == project_id), None)
    return project
