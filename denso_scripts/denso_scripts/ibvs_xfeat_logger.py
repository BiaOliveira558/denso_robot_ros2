#!/usr/bin/env python3
"""
Logger de experimentos IBVS ponderado com XFeat + KLT (robô real).

Modo 'goal' (uma vez, com o robô na pose goal):
  salva em ~/xfeat_ibvs_logs/goal/: goal_image.png e goal_pose.yaml (pose da câmera + juntas)

Modo 'run' (cada experimento), em ~/xfeat_ibvs_logs/run_<data>_<tag>/:
  data.csv      1 linha por mensagem do controlador (/ibvs/debug): mesmas métricas do IBVS
                clássico (status, erro, comando, juntas, pose da câmera, convergência)
                + rms, N, soma_w, cond(L), saturação, idade da medida, tempo do controlador
                + pontos ativos por slot (id, ex, ey, w, Z)
  motion.csv    1 linha por comando enviado ao Servo (~50 Hz): comando, juntas e pose
  frames.csv    1 linha por frame do extractor: tempos, XFeat, tracking, sobrevivência dos pontos
  tracks.csv    1 linha por ID por frame: status, idade, posição, peso, erro em px
  summary.yaml  métricas escalares do run
  initial_image.png, final_image.png, goal_image.png, goal_pose.yaml, comparison_side_by_side.png
Após convergir, grava por settle_time (status SETTLING): a última linha é a pose final parada.
"""
import os
import csv
import math
import shutil
from datetime import datetime

import yaml
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener

CMD_VEL_TOPIC = '/servo_node/delta_twist_cmds'
STATUS = {0: 'OK', 1: 'LOST', 2: 'STALE', 3: 'OUTLIER', 4: 'CONVERGED',
          5: 'BAD_MSG', 6: 'LOW_W', 7: 'BAD_Z', 8: 'WAIT'}
NOME_TRK = {-1: 'descartado', 0: 'perdido', 1: 'reserva', 2: 'entrando', 3: 'ativo', 4: 'saindo'}
H_CTRL = 22      # cabeçalho do /ibvs/debug
H_EXT = 25       # cabeçalho do /ibvs/xfeat_debug
COLS_PT = 6      # por ponto no /ibvs/debug: id, ex, ey, w, Z, Z*
COLS_ID = 6      # por ID no /ibvs/xfeat_debug: status, idade, u, v, w, e_px
N_SLOTS = 9      # 8 ativos + 1 entrando durante uma troca


def fmt(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ''
    return '' if math.isnan(v) else f'{v:.6f}'


def quat_angle_deg(q1, q2):
    d = abs(float(np.dot(q1, q2)))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


def estat(v):
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return {'media': float(a.mean()), 'p95': float(np.percentile(a, 95)),
            'max': float(a.max()), 'min': float(a.min()), 'n': int(a.size)}


class IBVSXFeatLogger(Node):
    def __init__(self):
        super().__init__('ibvs_xfeat_logger')

        self.declare_parameter('mode', 'run')          # run | goal
        self.declare_parameter('log_dir', os.path.expanduser('~/xfeat_ibvs_logs'))
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'basic_camera_link')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('joint_names',
                               ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        self.declare_parameter('settle_time', 1.0)
        self.declare_parameter('settle_rate', 10.0)
        self.declare_parameter('timeout', 90.0)
        self.declare_parameter('tag', '')
        self.declare_parameter('debug_topic', '/ibvs/debug')
        self.declare_parameter('xfeat_debug_topic', '/ibvs/xfeat_debug')

        gp = lambda n: self.get_parameter(n).value
        self.mode = gp('mode')
        self.log_dir = gp('log_dir')
        self.goal_dir = os.path.join(self.log_dir, 'goal')
        self.base_frame = gp('base_frame')
        self.camera_frame = gp('camera_frame')
        self.joint_names = list(gp('joint_names'))
        self.settle_time = gp('settle_time')
        self.settle_rate = gp('settle_rate')
        self.timeout = gp('timeout')

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._last_img = None
        self._joints = np.full(len(self.joint_names), np.nan)
        self._finished = False
        self.done = False

        self.create_subscription(Image, gp('image_topic'), self.image_cb, qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)

        # ======================= modo GOAL =======================
        if self.mode == 'goal':
            os.makedirs(self.goal_dir, exist_ok=True)
            self.create_timer(0.5, self.try_save_goal)
            self.get_logger().info(f'Modo GOAL: aguardando imagem + TF + juntas para salvar em {self.goal_dir}')
            return

        # ======================= modo RUN ========================
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tag = str(gp('tag'))
        self.run_dir = os.path.join(self.log_dir, f'run_{stamp}' + (f'_{tag}' if tag else ''))
        os.makedirs(self.run_dir, exist_ok=True)

        # goal: copia para o run (cada run fica autocontido) e carrega a pose
        self.goal_pose = None
        gp_file = os.path.join(self.goal_dir, 'goal_pose.yaml')
        if os.path.exists(gp_file):
            with open(gp_file) as f:
                self.goal_pose = np.array(yaml.safe_load(f)['pose'], dtype=float)
        else:
            self.get_logger().warn('goal_pose.yaml nao encontrado — rode antes com mode:=goal')
        for fn in ('goal_pose.yaml', 'goal_image.png', 'goal_ids.csv'):
            src = os.path.join(self.goal_dir, fn)
            if os.path.exists(src):
                shutil.copy(src, self.run_dir)

        juntas = [f'q_{n}' for n in self.joint_names]
        pose_cols = ['px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw']

        # ---- data.csv ----
        self.f_data = open(os.path.join(self.run_dir, 'data.csv'), 'w', newline='')
        self.w_data = csv.writer(self.f_data)
        slots = []
        for k in range(1, N_SLOTS + 1):
            slots += [f'id{k}', f'ex{k}', f'ey{k}', f'w{k}', f'Z{k}']
        self.w_data.writerow(
            ['t_ros', 't_rel', 'status', 'rms', 'err_norm', 'n_pontos', 'soma_w', 'cond_L', 'sat_k',
             'idade_ms', 't_ctrl_ms', 'below',
             'vx', 'vy', 'vz', 'wx', 'wy', 'wz',
             'cmd_vx', 'cmd_vy', 'cmd_vz', 'cmd_wx', 'cmd_wy', 'cmd_wz']
            + slots + juntas + pose_cols + ['ext_frame', 'converged'])

        # ---- motion.csv (~50 Hz) ----
        self.f_motion = open(os.path.join(self.run_dir, 'motion.csv'), 'w', newline='')
        self.w_motion = csv.writer(self.f_motion)
        self.w_motion.writerow(['t_ros', 't_rel', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz']
                               + juntas + pose_cols)
        self._n_motion = 0

        # ---- frames.csv ----
        self.f_frames = open(os.path.join(self.run_dir, 'frames.csv'), 'w', newline='')
        self.w_frames = csv.writer(self.f_frames)
        self.w_frames.writerow(
            ['t_ros', 't_rel', 'frame', 'intervalo_ms', 'idade_frame_ms', 't_total_ms', 't_klt_ms',
             't_xfeat_ms', 'xfeat_rodou', 'n_matches', 'n_inliers', 'n_repostos', 'n_rastreados',
             'n_elegiveis', 'n_ativos', 'soma_w', 'dist_min_px', 'rms', 'perdas_ativo', 'entradas',
             'n_keypoints_xfeat', 'n_rastreados_antes', 'n_mantidos', 'n_perdidos_klt',
             'ativos_mudaram'])

        # ---- tracks.csv ----
        self.f_tracks = open(os.path.join(self.run_dir, 'tracks.csv'), 'w', newline='')
        self.w_tracks = csv.writer(self.f_tracks)
        self.w_tracks.writerow(['t_ros', 't_rel', 'frame', 'id', 'status', 'idade',
                                'u', 'v', 'w', 'e_px'])

        # ---- estado do run ----
        self._t0 = None
        self._init_saved = False
        self._ext_frame = None
        self._converged = False
        self._t_conv = None
        self._settle_start = None
        self._settle_timer = None

        # ---- acumuladores para o summary ----
        self.c_t, self.c_age, self.c_tctrl = [], [], []
        self.rms_ini = self.err_ini = self.rms_fim = self.err_fim = None
        self.ids_ini = self.ids_fim = None
        self.n_ctrl = 0
        self.e_int, self.e_idade, self.e_total, self.e_klt, self.e_xfeat = [], [], [], [], []
        self.e_kp, self.e_rastreados = [], []
        self.n_frames = self.n_xfeat = self.n_perdas = self.n_entradas = 0
        self.soma_antes = self.soma_mantidos = self.n_perdidos_klt = self.n_mudancas = 0

        self.create_subscription(Float64MultiArray, gp('debug_topic'), self.debug_cb, 50)
        self.create_subscription(Float64MultiArray, gp('xfeat_debug_topic'), self.ext_cb, 50)
        self.create_subscription(TwistStamped, CMD_VEL_TOPIC, self.twist_cb, 50)
        self.create_timer(0.5, self.check_timeout)

        self.get_logger().info(f'Modo RUN: gravando em {self.run_dir}')

    # ------------------------------------------------------------------
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def lookup_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, Time())
        except Exception as ex:
            self.get_logger().warn(f'TF indisponível: {ex}', throttle_duration_sec=2.0)
            return np.full(7, np.nan)
        t, q = tf.transform.translation, tf.transform.rotation
        return np.array([t.x, t.y, t.z, q.x, q.y, q.z, q.w])

    def image_cb(self, msg):
        self._last_img = msg

    def joint_cb(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}   # lê pelo NOME (ordem do tópico pode variar)
        for k, n in enumerate(self.joint_names):
            if n in idx:
                self._joints[k] = msg.position[idx[n]]

    def save_img(self, pasta, nome):
        if self._last_img is None:
            return None
        img = self.bridge.imgmsg_to_cv2(self._last_img, desired_encoding='bgr8')
        cv2.imwrite(os.path.join(pasta, nome), img)
        return img

    # ======================= modo GOAL =======================
    def try_save_goal(self):
        if self.done:
            return
        pose = self.lookup_pose()
        if (self._last_img is None or not np.all(np.isfinite(pose))
                or not np.all(np.isfinite(self._joints))):
            return
        self.save_img(self.goal_dir, 'goal_image.png')
        with open(os.path.join(self.goal_dir, 'goal_pose.yaml'), 'w') as f:
            yaml.safe_dump({
                'base_frame': self.base_frame,
                'camera_frame': self.camera_frame,
                'stamp': datetime.now().isoformat(),
                'pose': [float(v) for v in pose],                      # x y z qx qy qz qw
                'joints': {n: float(v) for n, v in zip(self.joint_names, self._joints)},
            }, f, sort_keys=False)
        self.get_logger().info(f'Goal salva em {self.goal_dir}: pose {np.round(pose, 4).tolist()}')
        self.done = True

    # ======================= modo RUN ========================
    def write_data(self, t, status, vals, pts, converged):
        rms, en, n, sw, cond, k, age, tctrl, below = vals[:9]
        alvo, cmd = vals[9:15], vals[15:21]
        slots = []
        for j in range(N_SLOTS):
            if j < len(pts):
                p = pts[j]
                slots += [str(int(p[0])), fmt(p[1]), fmt(p[2]), fmt(p[3]), fmt(p[4])]
            else:
                slots += ['', '', '', '', '']
        pose = self.lookup_pose()
        self.w_data.writerow(
            [f'{t:.6f}', f'{t - self._t0:.6f}', status, fmt(rms), fmt(en), fmt(n), fmt(sw),
             fmt(cond), fmt(k), fmt(age * 1000.0), fmt(tctrl), fmt(below)]
            + [fmt(v) for v in alvo] + [fmt(v) for v in cmd]
            + slots
            + [fmt(v) for v in self._joints]
            + [fmt(v) for v in pose]
            + ['' if self._ext_frame is None else self._ext_frame, int(converged)])
        self.f_data.flush()

    def debug_cb(self, msg):
        if self._converged or self._finished:
            return
        d = np.asarray(msg.data, dtype=float)
        if d.size < H_CTRL:
            return
        code = int(d[0])
        status = STATUS.get(code, 'UNKNOWN')
        t = self.now_s()
        if self._t0 is None:
            self._t0 = t
        if not self._init_saved and self._last_img is not None:
            self.save_img(self.run_dir, 'initial_image.png')   # POV na pose inicial (start_delay)
            self._init_saved = True

        resto = d[H_CTRL:]
        if resto.size and resto.size % COLS_PT == 0:
            pts = resto.reshape(-1, COLS_PT)
        else:
            pts = np.zeros((0, COLS_PT))
        converged = code == 4
        self.write_data(t, status, d[1:H_CTRL], pts, converged)
        self.n_ctrl += 1

        if code in (0, 4):
            rms, en = d[1], d[2]
            if np.isfinite(rms):
                if self.rms_ini is None:
                    self.rms_ini, self.err_ini = float(rms), float(en)
                self.rms_fim, self.err_fim = float(rms), float(en)
            self.c_t.append(t)
            self.c_age.append(d[7] * 1000.0)
            self.c_tctrl.append(d[8])
            ids = [int(x) for x in pts[:, 0]]
            if ids:
                if self.ids_ini is None:
                    self.ids_ini = ids
                self.ids_fim = ids

        if converged:
            self._converged = True
            self._t_conv = t - self._t0
            self.get_logger().info(
                f'Convergiu em {self._t_conv:.2f}s — gravando settling por {self.settle_time}s')
            self._settle_start = t
            self._settle_timer = self.create_timer(1.0 / self.settle_rate, self.settle_tick)

    def twist_cb(self, msg):
        if self._t0 is None or self._finished:
            return
        t = self.now_s()
        l, a = msg.twist.linear, msg.twist.angular
        pose = self.lookup_pose()
        self.w_motion.writerow([f'{t:.6f}', f'{t - self._t0:.6f}']
                               + [fmt(v) for v in (l.x, l.y, l.z, a.x, a.y, a.z)]
                               + [fmt(v) for v in self._joints]
                               + [fmt(v) for v in pose])
        self._n_motion += 1
        if self._n_motion % 25 == 0:
            self.f_motion.flush()

    def ext_cb(self, msg):
        d = np.asarray(msg.data, dtype=float)
        if d.size < H_EXT:
            return
        self._ext_frame = int(d[0])
        if self._t0 is None or self._finished:
            return
        t = self.now_s()
        tr = t - self._t0
        self.w_frames.writerow([f'{t:.6f}', f'{tr:.6f}', int(d[0])] + [fmt(v) for v in d[3:H_EXT]])

        n_ids = (d.size - H_EXT) // COLS_ID
        for i in range(n_ids):
            b = d[H_EXT + COLS_ID * i: H_EXT + COLS_ID * (i + 1)]
            self.w_tracks.writerow([f'{t:.6f}', f'{tr:.6f}', int(d[0]), i,
                                    NOME_TRK.get(int(b[0]), '?'), int(b[1]),
                                    fmt(b[2]), fmt(b[3]), fmt(b[4]), fmt(b[5])])
        self.f_frames.flush()
        self.f_tracks.flush()

        # acumuladores
        self.n_frames += 1
        self.e_int.append(d[3])
        self.e_idade.append(d[4])
        self.e_total.append(d[5])
        self.e_klt.append(d[6])
        self.e_rastreados.append(d[12])
        if d[8] > 0.5:
            self.n_xfeat += 1
            self.e_xfeat.append(d[7])
            self.e_kp.append(d[20])
        self.n_perdas += int(d[18])
        self.n_entradas += int(d[19])
        self.soma_antes += int(d[21])
        self.soma_mantidos += int(d[22])
        self.n_perdidos_klt += int(d[23])
        self.n_mudancas += int(d[24])

    def settle_tick(self):
        if self._finished:
            return
        self.write_data(self.now_s(), 'SETTLING', np.full(21, np.nan), np.zeros((0, COLS_PT)), True)
        if self.now_s() - self._settle_start >= self.settle_time:
            self.finalize()

    def check_timeout(self):
        if self._finished or self._converged or self._t0 is None:
            return
        if self.now_s() - self._t0 > self.timeout:
            self.get_logger().warn(f'Timeout de {self.timeout}s sem convergir — FALHA')
            self.finalize()

    # ------------------------------------------------------------------
    def finalize(self):
        if self._finished or self.mode != 'run':
            return
        self._finished = True
        if self._settle_timer is not None:
            self._settle_timer.cancel()

        final_img = self.save_img(self.run_dir, 'final_image.png')
        goal_img_path = os.path.join(self.run_dir, 'goal_image.png')
        if final_img is not None and os.path.exists(goal_img_path):
            goal_img = cv2.imread(goal_img_path)
            final_rs = cv2.resize(final_img, (goal_img.shape[1], goal_img.shape[0]))
            side = cv2.hconcat([goal_img, final_rs])
            cv2.putText(side, 'GOAL', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(side, 'FINAL', (goal_img.shape[1] + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(self.run_dir, 'comparison_side_by_side.png'), side)

        taxa_ctrl = None
        if len(self.c_t) > 1 and self.c_t[-1] > self.c_t[0]:
            taxa_ctrl = (len(self.c_t) - 1) / (self.c_t[-1] - self.c_t[0])
        int_stats = estat(self.e_int)

        final_pose = self.lookup_pose()
        summary = {
            'converged': bool(self._converged),
            'convergence_time_s': self._t_conv,
            'timeout_s': float(self.timeout),
            'n_amostras_controle': self.n_ctrl,
            'taxa_controle_hz': taxa_ctrl,
            'rms_inicial': self.rms_ini, 'rms_final': self.rms_fim,
            'err_norm_inicial': self.err_ini, 'err_norm_final': self.err_fim,
            'latencia_camera_controlador_ms': estat(self.c_age),
            't_controlador_ms': estat(self.c_tctrl),
            'extractor': {
                'n_frames': self.n_frames,
                'taxa_hz': (1000.0 / int_stats['media']) if int_stats else None,
                'intervalo_ms': int_stats,
                'idade_frame_ms': estat(self.e_idade),
                't_total_ms': estat(self.e_total),
                't_klt_ms': estat(self.e_klt),
                't_xfeat_ms': estat(self.e_xfeat),
                'n_execucoes_xfeat': self.n_xfeat,
                'keypoints_xfeat': estat(self.e_kp),
            },
            'tracking': {
                'ids_rastreados_por_frame': estat(self.e_rastreados),
                'taxa_sobrevivencia_klt': (self.soma_mantidos / self.soma_antes)
                    if self.soma_antes > 0 else None,
                'perdidos_klt_total': self.n_perdidos_klt,
                'perdas_de_ativos': self.n_perdas,
                'entradas_de_substitutos': self.n_entradas,
                'mudancas_conjunto_ativos': self.n_mudancas,
                'ativos_iniciais': self.ids_ini,
                'ativos_finais': self.ids_fim,
                'ativos_iguais_inicio_fim': (self.ids_ini == self.ids_fim)
                    if self.ids_ini is not None else None,
            },
            'final_joints': [float(v) for v in self._joints],
            'final_pose': [float(v) for v in final_pose],
        }
        if self.goal_pose is not None and np.all(np.isfinite(final_pose)):
            dt = final_pose[:3] - self.goal_pose[:3]
            summary['goal_pose'] = self.goal_pose.tolist()
            summary['pos_error_mm'] = {'dx': float(dt[0] * 1e3), 'dy': float(dt[1] * 1e3),
                                       'dz': float(dt[2] * 1e3),
                                       'norm': float(np.linalg.norm(dt) * 1e3)}
            summary['rot_error_deg'] = quat_angle_deg(self.goal_pose[3:], final_pose[3:])

        with open(os.path.join(self.run_dir, 'summary.yaml'), 'w') as f:
            yaml.safe_dump(summary, f, sort_keys=False)

        for f in (self.f_data, self.f_motion, self.f_frames, self.f_tracks):
            f.close()
        self.get_logger().info(
            f'Run finalizado ({"CONVERGIU" if self._converged else "NAO convergiu"}) — {self.run_dir}')
        self.done = True


def main(args=None):
    rclpy.init(args=args)
    node = IBVSXFeatLogger()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if node.mode == 'run' and not node._finished:
            node.finalize()          # Ctrl+C sem convergir: salva tudo com converged=False
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()