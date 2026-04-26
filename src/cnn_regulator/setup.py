from setuptools import find_packages, setup

package_name = 'cnn_regulator'

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
    maintainer='zidane',
    maintainer_email='zidane.khadri@enetcom.u-sfax.tn',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ros_controller = cnn_regulator.ros_controller:main',
            'data_collector = cnn_regulator.data_collector:main',
            'target_ui = cnn_regulator.target_ui:main',
            'target_gui = cnn_regulator.gui_target_ui:main',
            'classic_controller = cnn_regulator.classic_controller:main'
        ],
    },
)
