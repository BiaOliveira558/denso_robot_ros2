#!/usr/bin/env python3
"""
Extrator XFeat + SAM2 — ETAPA 1: teste da goal segmentada (simulação Gazebo OU RealSense D405).

Fluxo:
  sem goal : janela mostra só o frame atual (keypoints XFeat em azul)
  g        : congela o frame -> desenha a caixa no objeto (Enter confirma, c cancela)
             -> SAM2 segmenta o objeto -> XFeat na goal, só keypoints dentro da máscara
             -> salva rgb.png, depth.npy, mask.png, bbox.txt, goal_keypoints.csv
  com goal : GOAL (máscara + keypoints em vermelho) | ATUAL (keypoints em azul)
             + linhas amarelas de match (vizinho mais próximo mútuo, SEM RANSAC)

Ainda NÃO há: RANSAC, escolha dos 20 IDs, KLT, ativos nem erro.
Publica em /ibvs/xfeat_features sempre [0, 0, stamp_sec, stamp_nanosec] (controle parado).

O parâmetro 'sim' (vindo do launch) escolhe os defaults:
  sim = True  -> tópicos /basic_camera, goal em ~/xfeat_goal, callbacks separados
  sim = False -> tópicos /camera/camera/..., goal em ~/xfeat_ibvs_logs/goal, sync RGB-D

Teclas na janela: g = definir goal | q = sair
"""
import os
import sys
import csv
import time
import cv2
import numpy as np
import torch
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
        'janela': 'XFeat + SAM2 (sim)',
    },
    False: {  # robô real (RealSense D405)
        'rgb_topic': '/camera/camera/color/image_raw',
        'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
        'info_topic': '/camera/camera/color/camera_info',
        'goal_dir': os.path.expanduser('~/xfeat_ibvs_logs/goal'),
        'janela': 'XFeat + SAM2 (real)',
    },
}

COR_KP_GOAL = (0, 0, 255)      # vermelho
COR_KP_ATUAL = (255, 0, 0)     # azul
COR_MATCH = (0, 255, 255)      # amarelo
COR_MASCARA = (0, 255, 0)      # verde


class XFeatSAM2Extractor(Node):
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
        self.declare_parameter('goal_dir', d['goal_dir'])
        # ---- XFeat ----
        self.declare_parameter('xfeat_path', os.path.expanduser('~/accelerated_features'))
        self.declare_parameter('top_k', 2048)          # keypoints do XFeat por imagem
        self.declare_parameter('max_dim', 640)
        self.declare_parameter('min_cossim', 0.82)     # similaridade mínima no match
        # ---- SAM2 ----
        self.declare_parameter('sam2_path', os.path.expanduser('~/sam2'))
        self.declare_parameter('sam2_cfg', 'configs/sam2.1/sam2.1_hiera_s.yaml')
        self.declare_parameter('sam2_ckpt', os.path.expanduser('~/sam2/checkpoints/sam2.1_hiera_small.pt'))
        self.declare_parameter('erosao_mascara', 5)     # px de erosão (tira a borda objeto/fundo)
        # ---- geral ----
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('sync_slop', 0.03)
        self.declare_parameter('rate', 30.0)
        self.declare_parameter('mostrar_janela', True)
        self.declare_parameter('desenhar_matches', True)
        self.declare_parameter('log_terminal', True)
        gp = lambda n: self.get_parameter(n).value

        self.goal_dir = gp('goal_dir')
        self.top_k = int(gp('top_k'))
        self.max_dim = int(gp('max_dim'))
        self.min_cossim = float(gp('min_cossim'))
        self.sam2_path = gp('sam2_path')
        self.sam2_cfg = gp('sam2_cfg')
        self.sam2_ckpt = gp('sam2_ckpt')
        self.erosao = int(gp('erosao_mascara'))
        self.depth_scale = gp('depth_scale')
        self.mostrar = gp('mostrar_janela')
        self.desenhar_matches = gp('desenhar_matches')
        self.log_terminal = gp('log_terminal')
        os.makedirs(self.goal_dir, exist_ok=True)

        # ---- XFeat ----
        sys.path.insert(0, gp('xfeat_path'))
        from modules.xfeat import XFeat
        torch.set_grad_enabled(False)
        self.xfeat = XFeat()

        # ---- SAM2: carregado só no primeiro g ----
        self.sam = None

        # ---- Estado ----
        self.bridge = CvBridge()
        self.frame = None
        self.frame_stamp = None
        self.depth = None
        self.novo = False
        self.K = None
        self.goal = None
        self.encoding_logado = False
        self.n_frame = 0
        self.erro = None             # ainda não calculado nesta etapa

        # ---- Goal salva em disco (só se tiver máscara) ----
        self.carregar_goal()

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
            f'XFeat+SAM2 iniciado [{"SIM" if self.sim else "REAL"}] ({self.xfeat.dev}) | '
            f'rgb={gp("rgb_topic")} | goal_dir={self.goal_dir} | top_k={self.top_k} | '
            f'erosao={self.erosao} px | teclas: g = goal, q = sair')

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

    def rgb_cb(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.frame_stamp = msg.header.stamp
        self.novo = True

    def depth_cb(self, msg):
        self.depth = self.converter_depth(msg)

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

    def casar(self, desc_goal, desc_cur):
        """Vizinho mais próximo mútuo goal x atual, com min_cossim.
        Devolve (índices na goal, índices no atual)."""
        if desc_goal.shape[0] == 0 or desc_cur.shape[0] == 0:
            return np.zeros(0, int), np.zeros(0, int)
        sim = desc_goal @ desc_cur.T                  # (M, N)
        melhor_c = sim.argmax(dim=1)
        melhor_g = sim.argmax(dim=0)
        ig = torch.arange(desc_goal.shape[0], device=desc_goal.device)
        ok = (melhor_g[melhor_c] == ig) & (sim[ig, melhor_c] > self.min_cossim)
        return ig[ok].cpu().numpy(), melhor_c[ok].cpu().numpy()

    # ================================================================== SAM2
    def carregar_sam(self):
        if self.sam is None:
            t0 = time.time()
            sys.path.insert(0, self.sam2_path)
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.sam = SAM2ImagePredictor(build_sam2(self.sam2_cfg, self.sam2_ckpt, device=dev))
            self.get_logger().info(f'SAM2 carregado ({dev}) em {time.time() - t0:.1f} s')
        return self.sam

    def segmentar(self, img, caixa):
        """Máscara (uint8 0/255) do objeto dentro da caixa [x0, y0, x1, y1] e score do SAM2."""
        pred = self.carregar_sam()
        with torch.inference_mode():
            pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            masks, scores, _ = pred.predict(box=np.array(caixa, dtype=np.float32), multimask_output=False)
        return (masks[0] > 0).astype(np.uint8) * 255, float(scores[0])

    # ================================================================== goal
    def capturar_goal(self):
        """g: congela o frame, desenha a caixa, segmenta com SAM2 e define/salva a goal."""
        img = self.frame.copy()
        depth = None if self.depth is None else self.depth.copy()

        jan = 'Goal: desenhe a caixa no objeto (Enter confirma, c cancela)'
        x, y, w, h = cv2.selectROI(jan, img, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(jan)
        cv2.waitKey(1)
        if w == 0 or h == 0:
            self.get_logger().warn('Caixa cancelada: goal nao alterada')
            return
        caixa = [int(x), int(y), int(x + w), int(y + h)]

        t0 = time.time()
        mask, score = self.segmentar(img, caixa)
        n_px = int((mask > 0).sum())
        self.get_logger().info(f'SAM2: score {score:.3f} | {n_px} px no objeto | {time.time() - t0:.1f} s')
        if n_px == 0:
            self.get_logger().warn('SAM2 nao segmentou nada: goal nao alterada')
            return
        self.definir_goal(img, depth, mask, caixa, salvar=True)

    def carregar_goal(self):
        """Lê a goal de goal_dir (precisa de rgb.png e mask.png)."""
        rgb_path = os.path.join(self.goal_dir, 'rgb.png')
        mask_path = os.path.join(self.goal_dir, 'mask.png')
        depth_path = os.path.join(self.goal_dir, 'depth.npy')
        bbox_path = os.path.join(self.goal_dir, 'bbox.txt')
        if not (os.path.exists(rgb_path) and os.path.exists(mask_path)):
            self.get_logger().warn(f'Sem goal segmentada em {self.goal_dir}: posicione o robo e aperte g')
            return
        depth = np.load(depth_path) if os.path.exists(depth_path) else None
        caixa = np.loadtxt(bbox_path).astype(int).tolist() if os.path.exists(bbox_path) else None
        self.definir_goal(cv2.imread(rgb_path), depth, cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE),
                          caixa, salvar=False)

    def definir_goal(self, img, depth, mask, caixa, salvar=True):
        t0 = time.time()
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = (mask > 0).astype(np.uint8)
        contorno, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if self.erosao > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.erosao + 1, 2 * self.erosao + 1))
            m = cv2.erode(m, k)

        # XFeat na goal inteira; ficam só os keypoints dentro da máscara erodida
        kp, desc, scores = self.extrair(img)
        n_total = len(kp)
        u = np.clip(np.round(kp[:, 0]).astype(int), 0, img.shape[1] - 1)
        v = np.clip(np.round(kp[:, 1]).astype(int), 0, img.shape[0] - 1)
        dentro = np.where(m[v, u] > 0)[0]
        kp, scores = kp[dentro], scores[dentro]
        desc = desc[torch.as_tensor(dentro, device=desc.device)]

        self.goal = {'img': img.copy(), 'mask': mask, 'contorno': contorno, 'caixa': caixa,
                     'kp': kp.astype(np.float32), 'desc': desc, 'score': scores,
                     'depth': None if depth is None else depth.copy()}

        if salvar:
            cv2.imwrite(os.path.join(self.goal_dir, 'rgb.png'), img)
            cv2.imwrite(os.path.join(self.goal_dir, 'mask.png'), mask)
            if depth is not None:
                np.save(os.path.join(self.goal_dir, 'depth.npy'), depth)
            if caixa is not None:
                np.savetxt(os.path.join(self.goal_dir, 'bbox.txt'), np.array(caixa)[None], fmt='%d',
                           header='x0 y0 x1 y1')
        with open(os.path.join(self.goal_dir, 'goal_keypoints.csv'), 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['u', 'v', 'score'])
            for (pu, pv), s in zip(kp, scores):
                wr.writerow([f'{pu:.3f}', f'{pv:.3f}', f'{s:.4f}'])

        self.get_logger().info(
            f'Goal {"salva e " if salvar else ""}definida: keypoints no objeto {len(kp)}/{n_total} | '
            f'caixa {caixa} | {time.time() - t0:.1f} s')
        if len(kp) == 0:
            self.get_logger().warn('Nenhum keypoint XFeat dentro da mascara')

    # ================================================================== laço principal
    def publicar_vazio(self, stamp):
        msg = Float64MultiArray()
        msg.data = [0.0, 0.0, float(stamp.sec), float(stamp.nanosec)]
        self.pub.publish(msg)

    def processar(self):
        if self.frame is None or not self.novo:
            self.teclado()
            return
        self.novo = False
        img = self.frame.copy()
        self.publicar_vazio(self.frame_stamp)
        self.n_frame += 1

        t0 = time.perf_counter()
        kp_cur, desc_cur, _ = self.extrair(img)
        if self.goal is not None:
            ig, ic = self.casar(self.goal['desc'], desc_cur)
        else:
            ig, ic = np.zeros(0, int), np.zeros(0, int)
        t_ms = (time.perf_counter() - t0) * 1000.0

        if self.log_terminal:
            n_goal = 0 if self.goal is None else len(self.goal['kp'])
            self.get_logger().info(
                f'frame {self.n_frame} | keypoints goal {n_goal} | atual {len(kp_cur)} | '
                f'matches {len(ig)} | xfeat+match {t_ms:.0f} ms')

        self.mostrar_imagem(img, kp_cur, ig, ic)

    # ================================================================== visualização
    def mostrar_imagem(self, img, kp_cur, ig, ic):
        if not self.mostrar:
            return

        if self.goal is None:
            vis = img.copy()
            for p in kp_cur:
                cv2.circle(vis, (int(p[0]), int(p[1])), 2, COR_KP_ATUAL, -1)
        else:
            # GOAL: máscara translúcida + contorno + keypoints do objeto
            g = self.goal['img'].copy()
            cor = g.copy()
            cor[self.goal['mask'] > 0] = COR_MASCARA
            g = cv2.addWeighted(g, 0.7, cor, 0.3, 0)
            cv2.drawContours(g, self.goal['contorno'], -1, COR_MASCARA, 2)
            for p in self.goal['kp']:
                cv2.circle(g, (int(p[0]), int(p[1])), 2, COR_KP_GOAL, -1)

            # ATUAL: mesma altura da goal + keypoints
            s = g.shape[0] / img.shape[0]
            atual = img if s == 1.0 else cv2.resize(img, (int(img.shape[1] * s), g.shape[0]))
            kc = kp_cur * s
            for p in kc:
                cv2.circle(atual, (int(p[0]), int(p[1])), 2, COR_KP_ATUAL, -1)

            vis = np.hstack([g, atual])
            off = g.shape[1]

            # matches: goal -> atual
            if self.desenhar_matches:
                for a, b in zip(self.goal['kp'][ig], kc[ic]):
                    cv2.line(vis, (int(a[0]), int(a[1])), (int(b[0]) + off, int(b[1])), COR_MATCH, 1)

        # faixa superior: só o erro
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (0, 0, 0), -1)
        erro_txt = 'ERRO: -' if self.erro is None else f'ERRO: {self.erro:.4f}'
        cv2.putText(vis, erro_txt, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(self.janela, vis)
        self.teclado()

    def teclado(self):
        if not self.mostrar:
            return
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt
        if key == ord('g') and self.frame is not None:
            self.capturar_goal()


def main():
    rclpy.init()
    node = XFeatSAM2Extractor()
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