from uuid import UUID

from db import get_db_session
from repositories import ProjectRepository
from schemas import Project
from services.base import GenericPersistenceService, ResourceNotFoundError, ResourceType, ServiceConfig
from services.mappers.project_mapper import ProjectMapper
from services.parent_process_guard import parent_process_only

MSG_ERR_DELETE_ACTIVE_PROJECT = "Cannot delete a project with a running pipeline."


class ProjectService:
    def __init__(self) -> None:
        self._persistence: GenericPersistenceService[Project, ProjectRepository] = GenericPersistenceService(
            ServiceConfig(ProjectRepository, ProjectMapper, ResourceType.PROJECT)
        )

    @parent_process_only
    def create_project(self, project: Project) -> Project:
        return self._persistence.create(project)

    def list_projects(self) -> list[Project]:
        return self._persistence.list_all()

    @parent_process_only
    def update_project(self, project: Project, partial_config: dict) -> Project:
        return self._persistence.update(project, partial_config)

    #@parent_process_only
    #def import_dataset(self, project_id: UUID, dataset: Dataset, config: ProjectConfig) -> Project:
    #    with get_db_session() as db:
    #        project = self.get_project_by_id(project_id)
    #        project.config = config
    #        project.datasets.append(dataset)
    #        print(project)
    #        db.commit()
    #        return project

    def get_project_by_id(self, project_id: UUID) -> Project:
        project = self._persistence.get_by_id(project_id)
        if not project:
            raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
        return project

    @parent_process_only
    def delete_project_by_id(self, project_id: UUID) -> None:
        with get_db_session() as db:
            self._persistence.delete_by_id(project_id, db)
            db.commit()
