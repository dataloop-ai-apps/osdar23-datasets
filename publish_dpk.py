import dtlpy as dl


if __name__ == '__main__':
    project_id = "76ca8599-86e0-4535-886a-622bbbfc3102"
    project = dl.projects.get(project_id=project_id)
    project.dpks.publish()
