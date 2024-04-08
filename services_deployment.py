import dtlpy as dl


def deploy_lidar_parser(project_name: str, upload_package: bool):
    project = dl.projects.get(project_name=project_name)
    package_name = 'lidar-osdar-parser-node'

    modules = [dl.PackageModule(
        name=package_name,
        init_inputs=[
            dl.FunctionIO(type=dl.PackageInputType.STRING, name="enable_ir_cameras"),
            dl.FunctionIO(type=dl.PackageInputType.STRING, name="enable_rgb_cameras"),
            dl.FunctionIO(type=dl.PackageInputType.STRING, name="enable_rgb_highres_cameras"),
        ],
        class_name='ServiceRunner',
        entry_point='lidar_parser_node.py',
        functions=[
            dl.PackageFunction(
                name='parse_lidar_data',
                inputs=[dl.FunctionIO(type=dl.PackageInputType.ITEM, name='item')],
                outputs=[dl.FunctionIO(type=dl.PackageInputType.ITEM, name='frames_item')],
            )
        ]
    )]

    if upload_package:
        package = project.packages.push(
            package_name=package_name,
            modules=modules,
            service_config={
                'runtime': dl.KubernetesRuntime(
                    runner_image="ofirgiladdataloop/dtlpy-raillabel-lidar-docker:1.0.0",
                    concurrency=10,
                    autoscaler=dl.KubernetesRabbitmqAutoscaler(
                        min_replicas=0,
                        max_replicas=1,
                        queue_length=100
                    )
                ).to_json()
            },
            src_path='.'
        )
        print("package has been deployed: ", package.name)
    else:
        package = project.packages.get(package_name=package_name)
        print("package has been gotten: ", package.name)

    #################
    # create service #
    #################

    try:
        service = package.services.get(service_name=package.name)
        print("service has been gotten: ", service.name)
    except dl.exceptions.NotFound:
        service = package.services.deploy(
            init_input=[
                dl.FunctionIO(
                    type=dl.PackageInputType.STRING,
                    value="true",
                    name="enable_ir_cameras"
                ),
                dl.FunctionIO(
                    type=dl.PackageInputType.STRING,
                    value="true",
                    name="enable_rgb_cameras"
                ),
                dl.FunctionIO(
                    type=dl.PackageInputType.STRING,
                    value="true",
                    name="enable_rgb_highres_cameras"
                ),
            ],
            service_name=package.name,
            module_name=package_name
        )
        print("service has been deployed: ", service.name)

    print("package.version: ", package.version)
    print("service.package_revision: ", service.package_revision)
    print("service.runtime.concurrency: ", service.runtime.concurrency)
    service.runtime.autoscaler.print()

    if package.version != service.package_revision:
        service.package_revision = package.version
        service.update(force=True)
        print("service.package_revision has been updated: ", service.package_revision)


def deploy_ground_detection(project_name: str, upload_package: bool):
    project = dl.projects.get(project_name=project_name)
    package_name = 'ground-detection-node'

    modules = [dl.PackageModule(
        name=package_name,
        class_name='ServiceRunner',
        entry_point='ground_detection_node.py',
        functions=[
            dl.PackageFunction(
                name='ground_detection',
                inputs=[
                    dl.FunctionIO(type=dl.PackageInputType.ITEM, name='item'),
                    dl.FunctionIO(type=dl.PackageInputType.STRING, name='model_id')
                ],
                outputs=[dl.FunctionIO(type=dl.PackageInputType.ITEM, name='item')]
            )
        ]
    )]

    if upload_package:
        package = project.packages.push(
            package_name=package_name,
            modules=modules,
            service_config={
                'runtime': dl.KubernetesRuntime(
                    concurrency=10,
                    autoscaler=dl.KubernetesRabbitmqAutoscaler(
                        min_replicas=0,
                        max_replicas=1,
                        queue_length=100
                    )
                ).to_json()
            },
            src_path='.'
        )
        print("package has been deployed: ", package.name)
    else:
        package = project.packages.get(package_name=package_name)
        print("package has been gotten: ", package.name)

    #################
    # create service #
    #################
    try:
        service = package.services.get(service_name=package.name)
        print("service has been gotten: ", service.name)
    except dl.exceptions.NotFound:
        service = package.services.deploy(service_name=package.name, module_name=package_name)
        print("service has been deployed: ", service.name)

    print("package.version: ", package.version)
    print("service.package_revision: ", service.package_revision)
    print("service.runtime.concurrency: ", service.runtime.concurrency)
    service.runtime.autoscaler.print()

    if package.version != service.package_revision:
        service.package_revision = package.version
        service.update(force=True)
        print("service.package_revision has been updated: ", service.package_revision)


if __name__ == "__main__":
    # TODO: Fill project name
    project_name = '<project-name>'

    deploy_lidar_parser(project_name=project_name, upload_package=True)
    deploy_ground_detection(project_name=project_name, upload_package=True)
