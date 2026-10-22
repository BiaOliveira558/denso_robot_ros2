#!/usr/bin/env python3
"""
Extrator XFeat + KLT para IBVS (simulação Gazebo OU câmera real RealSense D405),
com banco multi-escala / multi-rotação da goal.

O parâmetro 'sim' (vindo do launch) escolhe os defaults:
  sim = True  -> tópicos /basic_camera, goal em ~/xfeat_goal, z 0.05-2.0 m, callbacks separados
  sim = False -> tópicos /camera/camera/..., goal em ~/xfeat_goal_real, z 0.07-0.6 m, sync RGB-D

Publica em /ibvs/xfeat_features (Float64MultiArray):
  [N, detectado, stamp_sec, stamp_nanosec,
   x1, y1, z1, x1*, y1*, z1*,  ...  (N blocos, ordenados por ID)]

Pipeline:
  Goal  : os n_pontos de maior score viram IDs fixos 0..n-1 (raio_nms evita IDs amontoados).
          Banco: para cada escala x ângulo, a goal é reduzida/girada e guardamos
            - descritores lidos EXATAMENTE na posição de cada ID (cópia do ID naquela escala)
            - keypoints detectados naquela versão (não são IDs; reforçam o RANSAC)
          Toda linha do banco guarda a posição correspondente na goal ORIGINAL.
  Frame : KLT rastreia cada ID (checagem ida-e-volta). ID perdido sai do conjunto.
  Reposição: XFeat do frame atual x banco inteiro (vizinho mútuo + min_cossim) + RANSAC
          com as posições na goal original. IDs perdidos voltam com o mesmo número.
          O erro do IBVS continua sendo medido contra a posição do ID na goal ORIGINAL.

Teclas na janela: g = definir goal | q = sair
"""
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

DEFAULTS = {
    True: {   # simulação
        'rgb_topic': '/basic_camera',
        'depth_topic': '/basic_camera/depth/image_raw',
        'info_topic': '/basic_camera/camera_info',
        'goal_dir': os.path.expanduser('~/xfeat_goal'),
        'z_min': 0.05,
        'z_max': 2.0,
        'janela': 'XFeat + KLT feature extractor (sim)',
    },
    False: {  # robô real (RealSense D405)
        'rgb_topic': '/camera/camera/color/image_raw',
        'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
        'info_topic': '/camera/camera/color/camera_info',
        'goal_dir': os.path.expanduser('~/xfeat_goal_real'),
        'z_min': 0.07,
        'z_max': 0.6,
        'janela': 'XFeat + KLT feature extractor (real)',
    },
}


def top_n_pontos(pts, scores, n, raio):
    """Índices dos n pontos de maior score. Se raio > 0, ignora pontos a menos de
    'raio' px de um já escolhido (evita IDs amontoados)."""
    ordem = np.argsort(-scores)
    if raio <= 0:
        return ordem[:n]
    escolhidos = []
    for i in ordem:
        if escolhidos and np.min(np.linalg.norm(pts[escolhidos] - pts[i], axis=1)) < raio:
            continue
        escolhidos.append(i)
        if len(escolhidos) == n:
            break
    return np.array(escolhidos, dtype=int)


class XFeatKLTExtractor(Node):
    def __init__(self):
        super().__init__('xfeat_feature_extractor')

        # ---- Modo (precisa ser lido antes dos outros parâmetros) ----
        self.declare_parameter('sim', False)
        self.sim = bool(self.get_parameter('sim').value)
        d = DEFAULTS[self.sim]
        self.janela = d['janela']

        self.declare_parameter('rgb_topic', d['rgb_topic'])
        self.declare_parameter('depth_topic', d['depth_topic'])
        self.declare_parameter('info_topic', d['info_topic'])
        self.declare_parameter('features_topic', '/ibvs/xfeat_features')
        self.declare_parameter('xfeat_path', os.path.expanduser('~/accelerated_features'))
        self.declare_parameter('goal_dir', d['goal_dir'])
        self.declare_parameter('top_k', 2048)          # keypoints do XFeat no frame atual e na goal original
        self.declare_parameter('n_pontos', 20)         # IDs fixos escolhidos na goal
        self.declare_parameter('raio_nms', 20.0)       # px entre IDs na goal (0 = só score)
        # ---- Banco multi-escala / multi-rotação ----
        self.declare_parameter('escalas', [1.0, 0.7, 0.5, 0.35, 0.25])
        self.declare_parameter('angulos', [0.0, -25.0, 25.0])   # graus
        self.declare_parameter('top_k_nivel', 1024)    # keypoints detectados por versão (reforço do RANSAC)
        self.declare_parameter('min_lado', 64)         # px: ignora versões menores que isso
        self.declare_parameter('dup_px', 3.0)          # px: remove keypoint nativo colado num ID
        # ---------------------------------------------
        self.declare_parameter('min_pontos', 6)
        self.declare_parameter('redetect_every', 10)
        self.declare_parameter('rate', 15.0)
        self.declare_parameter('max_dim', 640)
        self.declare_parameter('min_cossim', 0.82)
        self.declare_parameter('ransac_px', 2.0)
        self.declare_parameter('klt_win', 21)
        self.declare_parameter('klt_niveis', 3)
        self.declare_parameter('klt_fb_px', 1.0)
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('sync_slop', 0.03)
        self.declare_parameter('z_min', d['z_min'])
        self.declare_parameter('z_max', d['z_max'])
        self.declare_parameter('mostrar_janela', True)
        self.declare_parameter('log_terminal', True)
        gp = lambda n: self.get_parameter(n).value

        self.top_k = gp('top_k')
        self.n_pontos = gp('n_pontos')
        self.raio_nms = float(gp('raio_nms'))
        self.escalas = [float(s) for s in gp('escalas')]
        self.angulos = [float(a) for a in gp('angulos')]
        self.top_k_nivel = int(gp('top_k_nivel'))
        self.min_lado = int(gp('min_lado'))
        self.dup_px = float(gp('dup_px'))
        self.min_pontos = max(gp('min_pontos'), 4)
        self.redetect_every = max(gp('redetect_every'), 1)
        self.max_dim = gp('max_dim')
        self.min_cossim = gp('min_cossim')
        self.ransac_px = gp('ransac_px')
        w = gp('klt_win')
        self.klt_params = dict(winSize=(w, w), maxLevel=gp('klt_niveis'),
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.klt_fb_px = gp('klt_fb_px')
        self.depth_scale = gp('depth_scale')
        self.z_min, self.z_max = float(gp('z_min')), float(gp('z_max'))
        self.goal_dir = gp('goal_dir')
        self.mostrar = gp('mostrar_janela')
        self.log_terminal = gp('log_terminal')
        os.makedirs(self.goal_dir, exist_ok=True)

        # ---- XFeat ----
        sys.path.insert(0, gp('xfeat_path'))
        from modules.xfeat import XFeat
        torch.set_grad_enabled(False)
        self.xfeat = XFeat()

        # ---- Estado ----
        self.bridge = CvBridge()
        self.frame = None
        self.frame_stamp = None
        self.depth = None
        self.novo = False
        self.K = None
        self.goal = None
        self.banco = None
        self.versoes = []
        self.encoding_logado = False
        self.n_frame = 0
        self.ultima_versao = '-'
        self.resetar_rastreio()

        # ---- Goal salva em disco ----
        rgb_path = os.path.join(self.goal_dir, 'rgb.png')
        depth_path = os.path.join(self.goal_dir, 'depth.npy')
        if os.path.exists(rgb_path):
            depth = np.load(depth_path) if os.path.exists(depth_path) else None
            self.definir_goal(cv2.imread(rgb_path), depth, salvar=False)
        else:
            self.get_logger().warn('Sem goal salva: posicione o robo e aperte g na janela')

        # ---- ROS ----
        if self.sim:
            self.create_subscription(Image, gp('rgb_topic'), self.rgb_cb, qos_profile_sensor_data)
            self.create_subscription(Image, gp('depth_topic'), self.depth_cb, qos_profile_sensor_data)
        else:
            self.sub_rgb = Subscriber(self, Image, gp('rgb_topic'), qos_profile=qos_profile_sensor_data)
            self.sub_depth = Subscriber(self, Image, gp('depth_topic'), qos_profile=qos_profile_sensor_data)
            self.sync = ApproximateTimeSynchronizer(
                [self.sub_rgb, self.sub_depth], queue_size=10, slop=gp('sync_slop'))
            self.sync.registerCallback(self.rgbd_cb)
        self.create_subscription(CameraInfo, gp('info_topic'), self.info_cb, 10)
        self.pub = self.create_publisher(Float64MultiArray, gp('features_topic'), 10)
        self.create_timer(1.0 / gp('rate'), self.processar,
                          clock=Clock(clock_type=ClockType.STEADY_TIME))

        if self.mostrar:
            cv2.namedWindow(self.janela, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(self.janela, 1600, 640)

        self.get_logger().info(
            f'XFeat+KLT iniciado [{"SIM" if self.sim else "REAL"}] ({self.xfeat.dev}) | '
            f'rgb={gp("rgb_topic")} | goal_dir={self.goal_dir} | n_pontos={self.n_pontos} | '
            f'escalas={self.escalas} | angulos={self.angulos} | teclas: g = goal, q = sair')

    # ================================================================== callbacks
    def converter_depth(self, msg):
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.float32)
        if msg.encoding == '16UC1':
            d *= self.depth_scale
        d[~np.isfinite(d)] = 0.0
        return d

    def rgbd_cb(self, rgb_msg, depth_msg):
        self.frame = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        self.frame_stamp = rgb_msg.header.stamp
        self.depth = self.converter_depth(depth_msg)
        self.novo = True
        if not self.encoding_logado:
            self.encoding_logado = True
            self.get_logger().info(
                f'RGB {rgb_msg.width}x{rgb_msg.height} ({rgb_msg.encoding}) | '
                f'depth {depth_msg.width}x{depth_msg.height} ({depth_msg.encoding})')
            if (rgb_msg.width, rgb_msg.height) != (depth_msg.width, depth_msg.height):
                self.get_logger().warn(
                    'RGB e depth com resolucoes diferentes: confira align_depth.enable:=true')

    def rgb_cb(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.frame_stamp = msg.header.stamp
        self.novo = True

    def depth_cb(self, msg):
        self.depth = self.converter_depth(msg)
        if not self.encoding_logado:
            self.encoding_logado = True
            self.get_logger().info(f'depth {msg.width}x{msg.height} ({msg.encoding})')
            if self.frame is not None and self.frame.shape[:2] != self.depth.shape[:2]:
                self.get_logger().warn('RGB e depth com resolucoes diferentes: Z sera reescalado')

    def info_cb(self, msg):
        if self.K is None:
            self.get_logger().info('camera_info recebido')
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    # ================================================================== XFeat
    def extrair(self, img):
        """Keypoints (pixels da imagem original), descritores e score de detecção."""
        h, w = img.shape[:2]
        escala = 1.0
        img_in = img
        if self.max_dim > 0 and max(h, w) > self.max_dim:
            escala = self.max_dim / max(h, w)
            img_in = cv2.resize(img, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_AREA)
        out = self.xfeat.detectAndCompute(img_in, top_k=self.top_k)[0]
        kp = out['keypoints'].cpu().numpy() / escala
        scores = out['scores'].cpu().numpy()
        return kp, out['descriptors'], scores

    def descrever_em(self, img, pts):
        """Descritores XFeat lidos EXATAMENTE nas posições pts (px de img), sem detector."""
        if len(pts) == 0:
            return torch.zeros((0, 64), device=self.xfeat.dev)
        x, rh, rw = self.xfeat.preprocess_tensor(img)
        M1, _, _ = self.xfeat.net(x)
        M1 = F.normalize(M1, dim=1)
        _, _, H, W = x.shape
        p = torch.tensor(pts / np.array([rw, rh]), dtype=torch.float32, device=x.device)[None]
        d = self.xfeat.interpolator(M1, p, H=H, W=W)[0]
        return F.normalize(d, dim=-1)

    def construir_banco(self, img, pts_id):
        """Banco de descritores da goal em várias escalas e ângulos.
        Cada linha: posição na goal ORIGINAL, ID (-1 = não é ID), descritor e versão."""
        H, W = img.shape[:2]
        base = self.max_dim / max(H, W) if (self.max_dim > 0 and max(H, W) > self.max_dim) else 1.0
        pos_l, id_l, desc_l, ver_l = [], [], [], []
        self.versoes = []
        margem = 6.0

        for s in self.escalas:
            fator = base * s
            w_s, h_s = int(round(W * fator)), int(round(H * fator))
            if min(w_s, h_s) < self.min_lado:
                continue
            img_s = cv2.resize(img, (w_s, h_s), interpolation=cv2.INTER_AREA)

            for a in self.angulos:
                R = cv2.getRotationMatrix2D((w_s / 2.0, h_s / 2.0), a, 1.0)
                img_v = img_s if a == 0.0 else cv2.warpAffine(
                    img_s, R, (w_s, h_s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                A = np.vstack([R, [0.0, 0.0, 1.0]])         # reduzida -> versão
                Ainv = np.linalg.inv(A)                     # versão -> reduzida
                v = len(self.versoes)
                self.versoes.append((s, a))

                # (a) IDs: descritor lido na posição exata de cada ID nesta versão
                pid_v = (A[:2, :2] @ (pts_id * fator).T).T + A[:2, 2]
                dentro = ((pid_v[:, 0] >= margem) & (pid_v[:, 0] < w_s - margem) &
                          (pid_v[:, 1] >= margem) & (pid_v[:, 1] < h_s - margem))
                ids_ok = np.where(dentro)[0]
                if len(ids_ok):
                    pos_l.append(pts_id[ids_ok])
                    id_l.append(ids_ok)
                    desc_l.append(self.descrever_em(img_v, pid_v[ids_ok]))
                    ver_l.append(np.full(len(ids_ok), v))

                # (b) keypoints detectados nesta versão (reforçam o RANSAC; não são IDs)
                out = self.xfeat.detectAndCompute(img_v, top_k=self.top_k_nivel)[0]
                kp_v = out['keypoints'].cpu().numpy()
                if len(kp_v) == 0:
                    continue
                kp_s = (Ainv[:2, :2] @ kp_v.T).T + Ainv[:2, 2]       # coordenadas na reduzida
                ok = ((kp_s[:, 0] >= margem) & (kp_s[:, 0] < w_s - margem) &
                      (kp_s[:, 1] >= margem) & (kp_s[:, 1] < h_s - margem))   # fora das bordas pretas da rotação
                if len(ids_ok):
                    dmin = np.min(np.linalg.norm(kp_v[:, None, :] - pid_v[ids_ok][None, :, :], axis=2), axis=1)
                    ok &= dmin > self.dup_px                          # não competir com o próprio ID
                idx = np.where(ok)[0]
                if len(idx):
                    pos_l.append((kp_s[idx] / fator).astype(np.float32))
                    id_l.append(np.full(len(idx), -1))
                    desc_l.append(out['descriptors'][torch.as_tensor(idx, device=out['descriptors'].device)])
                    ver_l.append(np.full(len(idx), v))

        if not desc_l:
            return None
        return {'pos': np.vstack(pos_l).astype(np.float32),
                'id': np.concatenate(id_l).astype(int),
                'desc': torch.cat([dd.to(self.xfeat.dev) for dd in desc_l], 0),
                'ver': np.concatenate(ver_l).astype(int)}

    def definir_goal(self, img, depth, salvar=True):
        t0 = time.time()
        kp, desc, scores = self.extrair(img)
        sel = top_n_pontos(kp, scores, self.n_pontos, self.raio_nms)
        pts_id = kp[sel].astype(np.float32)

        self.goal = {'img': img.copy(), 'pts_id': pts_id,
                     'depth': None if depth is None else depth.copy()}
        self.n_ids = len(sel)
        self.banco = self.construir_banco(img, pts_id)
        self.resetar_rastreio()

        if salvar:
            cv2.imwrite(os.path.join(self.goal_dir, 'rgb.png'), img)
            if depth is not None:
                np.save(os.path.join(self.goal_dir, 'depth.npy'), depth)

        n_banco = 0 if self.banco is None else len(self.banco['id'])
        n_id_rows = 0 if self.banco is None else int((self.banco['id'] >= 0).sum())
        self.get_logger().info(
            f'Goal definida: IDs fixos {self.n_ids} | versoes {len(self.versoes)} '
            f'(escalas x angulos) | banco {n_banco} descritores ({n_id_rows} de IDs) | '
            f'{"COM" if depth is not None else "SEM"} profundidade | {time.time() - t0:.1f} s')

    def casar(self, desc_cur):
        """Vizinho mais próximo mútuo entre banco e frame atual, com min_cossim.
        Devolve (linhas do banco, índices no frame, similaridade)."""
        B = self.banco['desc']
        if B.shape[0] == 0 or desc_cur.shape[0] == 0:
            return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
        sim = B @ desc_cur.T                         # (M, N)
        melhor_c = sim.argmax(dim=1)                 # para cada linha do banco
        melhor_b = sim.argmax(dim=0)                 # para cada ponto do frame
        ib = torch.arange(B.shape[0], device=B.device)
        mutuo = melhor_b[melhor_c] == ib
        val = sim[ib, melhor_c]
        ok = mutuo & (val > self.min_cossim)
        return ib[ok].cpu().numpy(), melhor_c[ok].cpu().numpy(), val[ok].cpu().numpy()

    def buscar_xfeat(self, img):
        """Frame atual x banco + RANSAC (posições na goal original).
        Devolve {id: posição atual} dos IDs inliers, n_matches, n_inliers."""
        achados = {}
        if self.banco is None:
            return achados, 0, 0
        kp, desc, _ = self.extrair(img)
        ib, ic, sv = self.casar(desc)
        if len(ib) < 8:
            return achados, len(ib), 0
        pg = self.banco['pos'][ib]
        pc = kp[ic].astype(np.float32)
        _, mask = cv2.findFundamentalMat(pg, pc, cv2.FM_RANSAC, self.ransac_px, 0.999)
        if mask is None:
            return achados, len(ib), 0
        inl = mask.ravel().astype(bool)

        # Versão (escala, ângulo) que mais contribuiu: indica a "distância aparente" atual
        vers = self.banco['ver'][ib[inl]]
        if len(vers):
            s, a = self.versoes[int(np.bincount(vers).argmax())]
            self.ultima_versao = f'escala {s:g} ang {a:+.0f}'

        melhor = {}
        for b, c, sim_v in zip(ib[inl], ic[inl], sv[inl]):
            id_ = int(self.banco['id'][b])
            if id_ >= 0 and (id_ not in melhor or sim_v > melhor[id_][1]):
                melhor[id_] = (kp[c], sim_v)
        achados = {id_: p for id_, (p, _) in melhor.items()}
        return achados, len(ib), int(inl.sum())

    # ================================================================== KLT
    def resetar_rastreio(self):
        n = getattr(self, 'n_ids', self.n_pontos)
        self.trk_pts = np.zeros((n, 2), dtype=np.float32)
        self.trk_ok = np.zeros(n, dtype=bool)
        self.trk_novo = np.zeros(n, dtype=bool)
        self.prev_gray = None
        self.frames_sem_xfeat = 0

    def rastrear_klt(self, gray):
        idx = np.where(self.trk_ok)[0]
        if self.prev_gray is None or len(idx) == 0:
            return
        p0 = self.trk_pts[idx].reshape(-1, 1, 2)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.klt_params)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None, **self.klt_params)
        fb = np.linalg.norm(p0 - p0r, axis=2).ravel()
        h, w = gray.shape
        p1 = p1.reshape(-1, 2)
        dentro = (p1[:, 0] >= 0) & (p1[:, 0] < w) & (p1[:, 1] >= 0) & (p1[:, 1] < h)
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.klt_fb_px) & dentro
        self.trk_pts[idx[ok]] = p1[ok]
        self.trk_ok[idx[~ok]] = False

    # ================================================================== auxiliares
    def amostrar_z(self, depth, pts, img_shape, r=2):
        z = np.zeros(len(pts))
        if depth is None:
            return z
        sy = depth.shape[0] / img_shape[0]
        sx = depth.shape[1] / img_shape[1]
        h, w = depth.shape[:2]
        for i, (x, y) in enumerate(pts):
            u, v = int(round(x * sx)), int(round(y * sy))
            jan = depth[max(v - r, 0):min(v + r + 1, h), max(u - r, 0):min(u + r + 1, w)]
            val = jan[(jan > self.z_min) & (jan < self.z_max)]
            if val.size > 0:
                z[i] = np.median(val)
        return z

    def publicar(self, stamp, pares=None):
        n = 0 if pares is None else len(pares)
        detectado = 1.0 if n >= self.min_pontos else 0.0
        dados = [float(n), detectado, float(stamp.sec), float(stamp.nanosec)]
        if pares is not None and n > 0:
            dados += pares.flatten().tolist()
        msg = Float64MultiArray()
        msg.data = dados
        self.pub.publish(msg)

    # ================================================================== laço principal
    def processar(self):
        if self.frame is None or not self.novo:
            self.teclado()
            return
        self.novo = False
        img = self.frame.copy()
        stamp = self.frame_stamp
        depth_atual = None if self.depth is None else self.depth.copy()

        if self.goal is None or self.K is None:
            self.publicar(stamp)
            self.mostrar_imagem(img, None, 'SEM GOAL (aperte g)' if self.goal is None else 'SEM camera_info', None)
            return

        self.n_frame += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.trk_novo[:] = False

        # 1. KLT nos IDs ativos
        self.rastrear_klt(gray)
        n_ok = int(self.trk_ok.sum())

        # 2. XFeat: busca inicial, rastreio fraco, ou reposição periódica de IDs perdidos
        modo = 'KLT'
        precisa = (n_ok < self.min_pontos) or \
                  (n_ok < self.n_ids and self.frames_sem_xfeat >= self.redetect_every)
        if precisa:
            achados, n_m, n_in = self.buscar_xfeat(img)
            for id_, p in achados.items():
                if not self.trk_ok[id_]:
                    self.trk_pts[id_] = p
                    self.trk_ok[id_] = True
                    self.trk_novo[id_] = True
            self.frames_sem_xfeat = 0
            modo = (f'KLT+XFeat (matches {n_m}, inliers {n_in}, repostos {int(self.trk_novo.sum())}, '
                    f'{self.ultima_versao})')
        else:
            self.frames_sem_xfeat += 1
        self.prev_gray = gray

        # 3. IDs ativos, ordenados por ID
        ids = np.where(self.trk_ok)[0]
        perdidos = np.where(~self.trk_ok)[0]
        if len(ids) < self.min_pontos:
            if self.log_terminal:
                self.get_logger().info(f'frame {self.n_frame} | {modo} | ativos {len(ids)}/{self.n_ids} | POUCOS PONTOS')
            self.publicar(stamp)
            self.mostrar_imagem(img, None, f'POUCOS PONTOS ({len(ids)}/{self.n_ids})', None)
            return

        pa = self.trk_pts[ids].astype(np.float64)
        pg = self.goal['pts_id'][ids].astype(np.float64)   # alvo = posição na goal ORIGINAL

        # 4. Erro IBVS em pixels
        e_px = np.linalg.norm(pa - pg, axis=1)
        erro = (float(e_px.mean()), float(np.sqrt((e_px ** 2).sum())), float(e_px.max()))

        # 5. Normalização e profundidade
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        x, y = (pa[:, 0] - cx) / fx, (pa[:, 1] - cy) / fy
        xs, ys = (pg[:, 0] - cx) / fx, (pg[:, 1] - cy) / fy
        z = self.amostrar_z(depth_atual, pa, img.shape)
        zs = self.amostrar_z(self.goal['depth'], pg, img.shape)

        # 6. Publicação
        self.publicar(stamp, np.column_stack([x, y, z, xs, ys, zs]))

        if self.log_terminal:
            perd_txt = ' '.join(str(i) for i in perdidos) if len(perdidos) else '-'
            self.get_logger().info(
                f'frame {self.n_frame} | {modo} | ativos {len(ids)}/{self.n_ids} | perdidos: {perd_txt} | '
                f'erro medio {erro[0]:.1f} px | max {erro[2]:.1f} px')

        z_val = z[z > 0]
        z_txt = f'{np.median(z_val):.3f} m' if z_val.size else '-'
        info = (f'{modo} | ativos: {len(ids)}/{self.n_ids} | '
                f'Z validos: {int((z > 0).sum())}/{len(ids)} | Z med: {z_txt}')
        self.mostrar_imagem(img, ids, info, erro)

    # ================================================================== visualização
    def mostrar_imagem(self, img, ids, texto, erro):
        if not self.mostrar:
            return
        if self.goal is not None:
            g = self.goal['img'].copy()
            atual = img if img.shape[0] == g.shape[0] else cv2.resize(
                img, (int(img.shape[1] * g.shape[0] / img.shape[0]), g.shape[0]))
            for id_, p in enumerate(self.goal['pts_id']):
                cor = (0, 255, 0) if (ids is not None and self.trk_ok[id_]) else (128, 128, 128)
                cv2.circle(g, (int(p[0]), int(p[1])), 5, cor, 2)
                cv2.putText(g, str(id_), (int(p[0]) + 6, int(p[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)
            vis = np.hstack([g, atual])
            off = g.shape[1]
            cv2.putText(vis, 'GOAL', (10, g.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(vis, 'ATUAL', (off + 10, g.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        else:
            vis = img.copy()
            off = 0

        if ids is not None:
            for id_ in ids:
                a = self.goal['pts_id'][id_]
                b = self.trk_pts[id_]
                pa_ = (int(b[0]) + off, int(b[1]))
                pg_ = (int(a[0]), int(a[1]))
                alvo = (int(a[0]) + off, int(a[1]))
                cor = (255, 255, 0) if self.trk_novo[id_] else (0, 255, 0)
                cv2.line(vis, pg_, pa_, cor, 1)
                cv2.arrowedLine(vis, pa_, alvo, (0, 255, 255), 1, tipLength=0.2)
                cv2.circle(vis, pa_, 5, cor, 2)
                cv2.putText(vis, str(id_), (pa_[0] + 6, pa_[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)

        cv2.rectangle(vis, (0, 0), (vis.shape[1], 66), (0, 0, 0), -1)
        cv2.putText(vis, texto, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        erro_txt = (f'ERRO medio: {erro[0]:.1f} px | max: {erro[2]:.1f} px'
                    if erro is not None else 'ERRO: -')
        cv2.putText(vis, erro_txt, (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.imshow(self.janela, vis)
        self.teclado()

    def teclado(self):
        if not self.mostrar:
            return
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt
        if key == ord('g') and self.frame is not None:
            self.definir_goal(self.frame.copy(), None if self.depth is None else self.depth.copy())


def main():
    rclpy.init()
    node = XFeatKLTExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()