#!/usr/bin/env python3
"""
pbvs_controller.py  —  PBVS para o braço Denso

Uso:
    ros2 run denso_scripts pbvs_controller

Funcionalidades:
    • Controle proporcional de posição e orientação via MoveIt Servo
    • Legenda explicativa no terminal
    • Status do Servo Node em tempo real
    • Parada automática se erro < 2 cm OU servo reportar singularidade/limite
    • Log completo salvo em /tmp/pbvs_log_<timestamp>.txt
"""

import math
import time
import datetime
import os
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Int8
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# ── Configurações ─────────────────────────────────────────────────────────────
KP_LIN          = 0.5     # ganho proporcional translação [1/s]
KP_ANG          = 0.5     # ganho proporcional rotação    [1/s]
MAX_LIN_VEL     = 0.10    # saturação linear              [m/s]
MAX_ANG_VEL     = 0.30    # saturação angular             [rad/s]
STOP_LIN_THRESH = 0.02    # para o script se et < 2 cm   [m]
STOP_ANG_THRESH = 0.035   # para o script se eang < 2°   [rad]
CONTROL_HZ      = 50      # frequência do laço de controle
PRINT_HZ        = 5       # frequência do print no terminal
BASE_FRAME      = 'base_link'
EE_FRAME        = 'J6'
SERVO_TOPIC     = '/servo_node/delta_twist_cmds'
STATUS_TOPIC    = '/servo_node/status'
LOG_DIR        = os.getcwd()
# ─────────────────────────────────────────────────────────────────────────────

# Status codes do MoveIt Servo (ROS 2 Humble)
#   Fonte: moveit_servo/status_codes.h
SERVO_STATUS_MAP = {
    -1: 'INVALIDO',
     0: 'OK',
     1: 'DECEL_SINGULARIDADE',   # desacelerando — singularidade próxima
     2: 'PARADO_SINGULARIDADE',  # movimento bloqueado por singularidade
     3: 'DECEL_COLISAO',         # desacelerando — colisão próxima
     4: 'PARADO_COLISAO',        # movimento bloqueado por colisão
     5: 'LIMITE_JUNTA',          # movimento bloqueado por limite de junta
}
# Status que causam parada automática do script
HALT_STATUSES = {1, 2, 4, 5}
 
 
class PBVSController(Node):
 
    def __init__(self):
        super().__init__('pbvs_controller')
 
        # ── TF ───────────────────────────────────────────────────────────────
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
 
        # ── Publisher de velocidade ───────────────────────────────────────────
        self._pub = self.create_publisher(TwistStamped, SERVO_TOPIC, 10)
 
        # ── Subscriber do status do Servo ─────────────────────────────────────
        self._servo_status     = 0
        self._servo_status_str = 'OK'
        self.create_subscription(Int8, STATUS_TOPIC, self._status_cb, 10)
 
        # ── Estado interno ────────────────────────────────────────────────────
        self._goal_pos   = None
        self._goal_rot   = None
        self._active     = False
        self._done       = False   # sinaliza ao main() para encerrar o spin
        self._iter       = 0
        self._tf_warned  = False   # evita spam de "Aguardando TF"
 
        # ── Arquivo de log ────────────────────────────────────────────────────
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self._log_path = os.path.join(LOG_DIR, f'pbvs_log_{ts}.txt')
        self._log_file = open(self._log_path, 'w', buffering=1)
 
        # ── Timer de controle ─────────────────────────────────────────────────
        self.create_timer(1.0 / CONTROL_HZ, self._loop)
 
        self._log(f'[PBVS] Nó iniciado  |  log: {self._log_path}')
        self._print_legend()
 
    # ─────────────────────────────────────────── Callback status Servo ────────
 
    def _status_cb(self, msg: Int8) -> None:
        self._servo_status     = msg.data
        self._servo_status_str = SERVO_STATUS_MAP.get(msg.data, f'DESCONHECIDO({msg.data})')
 
    # ─────────────────────────────────────────── Logging ─────────────────────
 
    def _log(self, text: str) -> None:
        """Imprime no terminal e salva no arquivo simultaneamente."""
        print(text)
        self._log_file.write(text + '\n')
 
    def _print_legend(self) -> None:
        legend = (
            '\n'
            '╔══════════════════════════════════════════════════════════════════╗\n'
            '║               PBVS Controller — Legenda das colunas            ║\n'
            '╠══════════════════════════════════════════════════════════════════╣\n'
            '║  ERRO DE POSIÇÃO (metros)                                       ║\n'
            '║    ex, ey, ez  → erro por eixo: goal_xyz − atual_xyz           ║\n'
            '║    et          → norma total: sqrt(ex²+ey²+ez²)                ║\n'
            '║                                                                  ║\n'
            '║  ERRO DE ORIENTAÇÃO (graus)                                     ║\n'
            '║    er, ep, ey_ → erro em roll, pitch, yaw                      ║\n'
            '║    eang        → norma total do vetor de rotação residual       ║\n'
            '║                                                                  ║\n'
            '║  VELOCIDADE ENVIADA AO ROBÔ                                     ║\n'
            '║    vx, vy, vz  → vel. linear  [m/s]   — move o EEF no espaço  ║\n'
            '║    wx, wy, wz  → vel. angular [rad/s] — gira o EEF             ║\n'
            '║    calculadas como: v = clip(KP × erro, ±MAX_VEL)              ║\n'
            '║                                                                  ║\n'
            '║  STATUS DO SERVO NODE                                           ║\n'
            '║    OK                    → operação normal                      ║\n'
            '║    DECEL_SINGULARIDADE   → desacelerando (próx. singularidade)  ║\n'
            '║    PARADO_SINGULARIDADE  → ✗ bloqueado por singularidade        ║\n'
            '║    DECEL_COLISAO         → desacelerando (próx. colisão)        ║\n'
            '║    PARADO_COLISAO        → ✗ bloqueado por colisão              ║\n'
            '║    LIMITE_JUNTA          → ✗ bloqueado por limite de junta      ║\n'
            '║                                                                  ║\n'
            '║  PARADA AUTOMÁTICA ocorre quando:                               ║\n'
            '║    • et < 0.02 m  e  eang < 2°  → goal atingido               ║\n'
            '║    • status = PARADO_* ou LIMITE_JUNTA                          ║\n'
            '╚══════════════════════════════════════════════════════════════════╝'
        )
        self._log(legend)
 
    # ─────────────────────────────────────────── API pública ─────────────────
 
    def go_to_pose(
        self,
        x: float, y: float, z: float,
        roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
    ) -> None:
        """
        Define o goal e inicia o controlador.
        Posição em metros, orientação em graus.
 
        Exemplos:
            node.go_to_pose(0.3, 0.0, 0.5)
            node.go_to_pose(0.3, 0.0, 0.5, pitch=90)
            node.go_to_pose(0.3, 0.0, 0.5, roll=-168, pitch=43, yaw=-143)
        """
        self._goal_pos = np.array([float(x), float(y), float(z)])
        self._goal_rot = Rotation.from_euler(
            'xyz', [math.radians(roll), math.radians(pitch), math.radians(yaw)]
        ).as_matrix()
        self._active   = True
        self._iter     = 0
        self._tf_warned = False
 
        self._log(
            f'\n[PBVS] Goal → pos=({x:.4f}, {y:.4f}, {z:.4f}) m  '
            f'rpy=({roll:.2f}°, {pitch:.2f}°, {yaw:.2f}°)'
        )
        self._log(
            f'[PBVS] Parâmetros: KP_lin={KP_LIN}  KP_ang={KP_ANG}  '
            f'max_lin={MAX_LIN_VEL} m/s  max_ang={MAX_ANG_VEL} rad/s  '
            f'stop_thresh={STOP_LIN_THRESH*100:.0f} cm'
        )
 
        # Cabeçalho da tabela
        header = (
            f'\n{"ciclo":>6} | '
            f'{"ex(m)":>7} {"ey(m)":>7} {"ez(m)":>7} {"et(m)":>7} | '
            f'{"er(°)":>7} {"ep(°)":>7} {"ey_(°)":>7} {"eang(°)":>7} | '
            f'{"vx(m/s)":>8} {"vy(m/s)":>8} {"vz(m/s)":>8} | '
            f'{"wx(r/s)":>8} {"wy(r/s)":>8} {"wz(r/s)":>8} | '
            f'{"servo_status"}'
        )
        self._log(header)
        self._log('─' * 125)
 
    # ─────────────────────────────────────────── Laço principal ──────────────
 
    def _loop(self) -> None:
        if not self._active:
            return
 
        tf_data = self._get_ee_tf()
        if tf_data is None:
            if not self._tf_warned:
                self._log('[PBVS] Aguardando TF...')
                self._tf_warned = True
            return
        self._tf_warned = False
 
        pos_c, rot_c = tf_data
 
        # ── Erros ─────────────────────────────────────────────────────────────
        t_err  = self._goal_pos - pos_c
        t_norm = float(np.linalg.norm(t_err))
 
        R_e    = self._goal_rot @ rot_c.T
        r_err  = Rotation.from_matrix(R_e).as_rotvec()
        r_norm = float(np.linalg.norm(r_err))
 
        # ── Verificar status do Servo (singularidade / limite de junta) ────────
        if self._servo_status in HALT_STATUSES:
            self._send_zero()
            self._active = False
            self._done   = True
            self._log('─' * 125)
            self._log(
                f'[PBVS] ✗ PARADO pelo Servo: {self._servo_status_str}\n'
                f'         Erro no momento: Δpos={t_norm*1000:.2f} mm  '
                f'Δang={math.degrees(r_norm):.2f}°'
            )
            self._close_log()
            return
 
        # ── Parada automática por convergência (et < 2 cm) ────────────────────
        if t_norm < STOP_LIN_THRESH and r_norm < STOP_ANG_THRESH:
            self._send_zero()
            self._active = False
            self._done   = True
            self._log('─' * 125)
            self._log(
                f'[PBVS] ✓ Convergido!  '
                f'Δpos={t_norm*1000:.2f} mm  Δang={math.degrees(r_norm):.2f}°'
            )
            self._close_log()
            return
 
        # ── Lei de controle proporcional com saturação ────────────────────────
        vx = float(np.clip(KP_LIN * t_err[0], -MAX_LIN_VEL, MAX_LIN_VEL))
        vy = float(np.clip(KP_LIN * t_err[1], -MAX_LIN_VEL, MAX_LIN_VEL))
        vz = float(np.clip(KP_LIN * t_err[2], -MAX_LIN_VEL, MAX_LIN_VEL))
        wx = float(np.clip(KP_ANG * r_err[0], -MAX_ANG_VEL, MAX_ANG_VEL))
        wy = float(np.clip(KP_ANG * r_err[1], -MAX_ANG_VEL, MAX_ANG_VEL))
        wz = float(np.clip(KP_ANG * r_err[2], -MAX_ANG_VEL, MAX_ANG_VEL))
 
        self._send_vel(vx, vy, vz, wx, wy, wz)
 
        # ── Print a cada (CONTROL_HZ / PRINT_HZ) ciclos ───────────────────────
        self._iter += 1
        if self._iter % (CONTROL_HZ // PRINT_HZ) == 0:
            rpy_err = np.degrees(Rotation.from_rotvec(r_err).as_euler('xyz'))
            self._log(
                f'{self._iter:>6} | '
                f'{t_err[0]:>7.3f} {t_err[1]:>7.3f} {t_err[2]:>7.3f} {t_norm:>7.3f} | '
                f'{rpy_err[0]:>7.2f} {rpy_err[1]:>7.2f} {rpy_err[2]:>7.2f} '
                f'{math.degrees(r_norm):>7.2f} | '
                f'{vx:>8.4f} {vy:>8.4f} {vz:>8.4f} | '
                f'{wx:>8.4f} {wy:>8.4f} {wz:>8.4f} | '
                f'{self._servo_status_str}'
            )
 
    # ─────────────────────────────────────────── Helpers ─────────────────────
 
    def _get_ee_tf(self) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            tf = self._tf_buffer.lookup_transform(
                BASE_FRAME, EE_FRAME, rclpy.time.Time()
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            pos = np.array([t.x, t.y, t.z])
            rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            return pos, rot
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
 
    def _send_vel(self, vx, vy, vz, wx=0.0, wy=0.0, wz=0.0) -> None:
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_FRAME
        msg.twist.linear.x  = vx
        msg.twist.linear.y  = vy
        msg.twist.linear.z  = vz
        msg.twist.angular.x = wx
        msg.twist.angular.y = wy
        msg.twist.angular.z = wz
        self._pub.publish(msg)
 
    def _send_zero(self) -> None:
        self._send_vel(0.0, 0.0, 0.0)
 
    def _close_log(self) -> None:
        if self._log_file.closed:
            return
        self._log(f'\n[PBVS] Log salvo em: {self._log_path}')
        self._log_file.close()
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
def main(args=None):
    rclpy.init(args=args)
    node = PBVSController()
 
    time.sleep(1.5)   # aguarda TF popular
 
    node.go_to_pose(
        x=0.35, y=0.10, z=0.40,
        roll=-168.349, pitch=42.951, yaw=-142.996
    )
 
    try:
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        print('\n[PBVS] Interrompido pelo usuário.')
    finally:
        try:
            node._send_zero()
        except Exception:
            pass
        try:
            node._close_log()
        except Exception:
            pass
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