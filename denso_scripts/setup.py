from setuptools import find_packages, setup

package_name = 'denso_scripts'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'move_joints = denso_scripts.move_joints:main',
        'move_xyz = denso_scripts.move_xyz:main',
        'main = denso_scripts.main:main',
        'keyboard_teleop = denso_scripts.keyboard_teleop:main',
        'PBVS = denso_scripts.PBVS:main',
    ],
},
)
