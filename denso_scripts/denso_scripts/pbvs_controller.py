#!/usr/bin/env python3

import math
import time
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

# ── Configurações ─────────────────────────────────────────────────────────
PRINT_HZ        = 5       # frequência do print no terminal
LOOP_HZ         = 50      # frequência do laço de checagem 
CAMERA_FRAME    = 'basic_camera_link'   # frame em que /aruco/pose é publicado
ARUCO_TOPIC     = '/aruco/pose'
POSE_TIMEOUT_S  = 0.5     # se não receber /aruco/pose por esse tempo, considera perdido
# ─────────────────────────────────────────────────────────────────────────


class PBVSPerception(Node):

    def __init__(self):
        super().__init__('pbvs_perception')

        # ── Subscriber da pose do marcador (ArUco) ──────────────────────────
        self._marker_pos     = None   # posição atual do marcador, frame câmera
        self._marker_rot     = None   # rotação atual do marcador, frame câmera
        self._last_marker_ts = None   # timestamp da última leitura recebida
        self.create_subscription(PoseStamped, ARUCO_TOPIC, self._marker_pose_cb, 10)

        # ── Estado interno ───────────────────────────────────────────────────
        self._goal_pos     = None   # pose DESEJADA do marcador, frame câmera
        self._goal_rot     = None
        self._active       = False
        self._iter         = 0
        self._pose_warned  = False

        # ── Timer do laço de percepção ───────────────────────────────────────
        self.create_timer(1.0 / LOOP_HZ, self._loop)

        print('[PERCEPTION] Nó iniciado')

    # ─────────────────────────────────────────── Callbacks ────────────────────

    def _marker_pose_cb(self, msg: PoseStamped) -> None:
        """Atualiza a pose ATUAL do marcador, sempre que uma nova leitura chega."""
        p = msg.pose.position
        q = msg.pose.orientation
        self._marker_pos = np.array([p.x, p.y, p.z])
        self._marker_rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        self._last_marker_ts = self.get_clock().now()

    # ─────────────────────────────────────────── API pública ──────────────────


    def set_goal_from_quat(
        self,
        x: float, y: float, z: float,
        qx: float, qy: float, qz: float, qw: float,
    ) -> None:
    
        self._goal_pos = np.array([float(x), float(y), float(z)])
        self._goal_rot = Rotation.from_quat(
            [float(qx), float(qy), float(qz), float(qw)]
        ).as_matrix()
        self._active      = True
        self._iter         = 0
        self._pose_warned  = False

        print(
            f'\n[PERCEPTION] Goal (marcador relativo à câmera, via quaternion bruto) → '
            f'pos=({x:.4f}, {y:.4f}, {z:.4f}) m  '
            f'quat=({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})'
        )

        header = f'{"ciclo":>6} | {"et(m)":>7} | {"eang(°)":>7} | {"marker":>7}'
        print(header)
        print('─' * 40)

    def stop(self) -> None:
        """Para o laço de percepção (não desliga o nó)."""
        self._active = True is False  # explícito: seta False
        self._active = False
        print('[PERCEPTION] Laço de percepção pausado.')

    # ─────────────────────────────────────────── Cálculo de erro ──────────────

    def compute_error(self):
        """
        Calcula o erro atual (posição + orientação) entre a pose estimada
        do marcador e a pose goal, ambos no frame da câmera ATUAL.

        Retorna:
            t_err  (np.ndarray 3,)  — erro de translação [m], no frame câmera atual
            t_norm (float)          — norma do erro de translação [m]
            r_err  (np.ndarray 3,)  — erro de rotação como vetor eixo-ângulo [rad]
            r_norm (float)          — norma do erro de rotação [rad]

        Retorna None se ainda não há leitura de marcador ou goal definido.
        """
        if self._marker_pos is None or self._goal_pos is None:
            return None

        # Rotação relativa entre a câmera atual (C) e a câmera no goal (C*)
        # R_goal_cam leva do frame câmera atual para o frame desejado
        R_goal_cam = self._goal_rot @ self._marker_rot.T
        R_cam_goal = R_goal_cam.T

        # Erro de orientação como vetor eixo-ângulo (evita gimbal lock/wrap
        # que a subtração direta de ângulos de Euler sofre)
        r_err  = Rotation.from_matrix(R_goal_cam).as_rotvec()
        r_norm = float(np.linalg.norm(r_err))

        # Erro de translação: projeta o goal no frame da câmera ATUAL antes
        # de subtrair, pra comparar as duas posições no mesmo referencial
        goal_pos_in_current_cam = R_cam_goal @ self._goal_pos
        t_err  = goal_pos_in_current_cam - self._marker_pos
        t_norm = float(np.linalg.norm(t_err))

        return t_err, t_norm, r_err, r_norm

    # ─────────────────────────────────────────── Laço principal ───────────────

    def _loop(self) -> None:
        if not self._active:
            return

        # ── Verifica se a leitura do marcador está disponível e recente ───────
        if self._marker_pos is None:
            if not self._pose_warned:
                print('[PERCEPTION] Aguardando primeira leitura de /aruco/pose...')
                self._pose_warned = True
            return

        age_s = (self.get_clock().now() - self._last_marker_ts).nanoseconds / 1e9
        marker_ok = age_s < POSE_TIMEOUT_S
        marker_str = 'OK' if marker_ok else 'PERDIDO'

        if not marker_ok:
            if not self._pose_warned:
                print(f'[PERCEPTION] ⚠ Marcador PERDIDO há {age_s:.2f}s')
                self._pose_warned = True
            return
        self._pose_warned = False

        result = self.compute_error()
        if result is None:
            return
        t_err, t_norm, r_err, r_norm = result

        # ── Print periódico ──────────────────────────────────────────────────
        self._iter += 1
        if self._iter % max(1, (LOOP_HZ // PRINT_HZ)) == 0:
            print(
                f'{self._iter:>6} | '
                f'{t_norm:>7.3f} | '
                f'{math.degrees(r_norm):>7.2f} | '
                f'{marker_str:>7}'
            )


# ── Entry point ─────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = PBVSPerception()

    time.sleep(1.5)   # aguarda primeiras leituras popularem

    # pose goal desejada.
    node.set_goal_from_quat(
        x=-0.027442032293871793, y=0.036176424210395096, z=0.11842066207463271,
        qx=0.8466827248468187, qy=-0.4033038838473596, qz=0.3157677182743702, qw=0.144100967433948
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n[PERCEPTION] Interrompido pelo usuário.')
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()