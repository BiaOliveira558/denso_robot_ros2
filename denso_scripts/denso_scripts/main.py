from denso_scripts.move_joints import move_joints, move_joints_sequence
from denso_scripts.move_xyz import move_xyz



def main():
    #move_joints(positions=[-114, -36, 86, -1, 0, -269], duration=10, degrees=True,)
    move_joints_sequence(
        targets=[
            ([-114, -36,  86, -1, 0, -269], 6),   # posição 1
            ([ -89,  29,  66,  0, 0,    0], 6),   # posição 2
            ([   0,   0,   0,  0, 0,    0], 5),   # home
        ],
        degrees=True,
    )
    #move_xyz(x=-0.039, y=-0.087, z=0.764,qx=0.413, qy=-0.088, qz=-0.188, qw=0.887,go_home_first=True,)


if __name__ == '__main__':
    main()