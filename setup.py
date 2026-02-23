from setuptools import setup

package_name = 'scheduler'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/scheduler_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Multi-robot task scheduler node with embedded optimization logic.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scheduler_node_1stage_compare = scheduler.scheduler_node_1stage_compare:main',
            'scheduler_node_1stage_RIME = scheduler.scheduler_node_1stage_RIME:main',
            'test = scheduler.test:main',
            'scheduler_node_2stage_simple = scheduler.scheduler_node_2stage_simple:main',
            'scheduler_node_2stage_complex = scheduler.scheduler_node_2stage_complex:main',
            'task_timeline_visualizer = scheduler.task_timeline_visualizer:main',
            'scheduler_node_GA = scheduler.scheduler_node_GA:main',
            'old_scheduler_node = scheduler.old_scheduler_node:main',
            'sequence_node = scheduler.sequence_node:main',
            'sequence_node_consist_vel = scheduler.sequence_node_consist_vel:main',
            'gazebo_test = scheduler.gazebo_test:main',
        ],
    },
)
