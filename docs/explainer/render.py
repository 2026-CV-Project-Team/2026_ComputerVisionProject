"""층별 양자화 실험 설명 영상 렌더러.

실행 (PIL · numpy · scipy, ffmpeg 필요):
  python render.py                  전체 영상 → 양자화_설명영상_v2.mp4
  python render.py --frames 12,40   해당 시각(초) 프레임만 PNG로 저장 (확인용)
  python render.py --sheet          장면마다 몇 장씩 모아 한 장으로 (확인용)

장면은 아래 SCENES 목록. 각 장면은 (장 이름, 길이, 그리기 함수, 자막, 효과음).
자막은 내레이션 대본 역할도 한다 (대본.txt 로도 저장).
수치 출처: 2026_ComputerVisionProject/results (6conv_fold/summary, imagenet/summary).
"""
import os, sys, math, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "자료")
# 외부 도구·소리 파일 위치는 환경 변수로 지정 (레포에는 소리·그림 원본을 넣지 않음, README 참고)
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
SFX_DIR = os.environ.get("SFX_DIR", os.path.join(HERE, "자료", "효과음"))
BGM = os.environ.get("BGM", os.path.join(HERE, "자료", "Wallpaper.mp3"))
FONT_DIR = os.path.expanduser("~/Library/Fonts")
OUT = os.path.join(HERE, "양자화_설명영상_v2.mp4")

W, H, FPS = 1920, 1080, 30

# 색: 교과서 느낌의 차분한 원색 몇 개만
BG = (246, 247, 249)
GRID = (230, 233, 238)
INK = (30, 34, 40)
SUB = (110, 118, 128)
LINE = (200, 206, 214)
BLUE = (31, 95, 191)
LBLUE = (214, 228, 248)
ORANGE = (236, 128, 30)
RED = (214, 64, 58)
GREEN = (36, 150, 86)
GREY = (150, 156, 164)
WHITE = (255, 255, 255)

# ───────────────────────── 수치 (실험 결과) ─────────────────────────
# 6-Conv, W4A4, 모델 5개 평균 (results/6conv_fold/summary/layers.csv, conditions.csv, mixed_all.csv)
ACC_W4 = 58.3
NMSE_ALL = [0.046, 0.123, 0.224, 0.235, 0.365, 1.302]   # 누적 오차
NMSE_ONLY = [0.046, 0.044, 0.056, 0.038, 0.038, 0.082]  # 그 층만 양자화했을 때 오차
GAIN = [19.2, 10.1, 11.7, 12.7, 4.9, -0.5]              # 한 층만 복원 시 정확도 변화(%p)
ACC_L1_8BIT = 74.0                                       # 1번 층만 8비트 (평균 비트 4.024 → +0.6%)
# ImageNet test 1만 장 (results/imagenet/summary/mixed_topk.csv), 상위 3개 단위 8비트 승격
RES = [("4비트 그대로", 17.1), ("오차가 큰 층 3개를 8비트로", 19.3), ("효과가 큰 층 3개를 8비트로", 54.8)]
DEIT = [("5비트 그대로", 6.8), ("오차가 큰 층 3개를 8비트로", 24.6), ("효과가 큰 층 3개를 8비트로", 32.9)]


# ───────────────────────── 기본 도구 ─────────────────────────
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, a, b):
    return clamp((t - a) / (b - a)) if b > a else float(t >= a)


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def out_cubic(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def out_back(x, s=1.9):
    x = clamp(x)
    x -= 1
    return x * x * ((s + 1) * x + s) + 1


def lerp(a, b, x):
    return a + (b - a) * x


def mix(c1, c2, x):
    return tuple(int(round(lerp(a, b, x))) for a, b in zip(c1, c2))


_fonts = {}


def font(wt, size):
    k = (wt, size)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(os.path.join(FONT_DIR, f"Pretendard-{wt}.otf"), size)
    return _fonts[k]


_tc = {}


def T(s, size, wt="Bold", col=INK, align="left", lh=1.35):
    """글자를 투명 배경 이미지로 (캐시)."""
    k = (s, size, wt, col, align, lh)
    if k in _tc:
        return _tc[k]
    f = font(wt, size)
    lines = s.split("\n")
    ws = [int(f.getlength(l)) for l in lines]
    asc, desc = f.getmetrics()
    step = int(size * lh)
    w = max(ws) + 6
    h = step * (len(lines) - 1) + asc + desc + 6
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, l in enumerate(lines):
        x = 3 if align == "left" else (w - ws[i]) // 2 if align == "center" else w - ws[i] - 3
        d.text((x, 3 + i * step), l, font=f, fill=col)
    _tc[k] = im
    return im


def with_alpha(im, a):
    if a >= 0.999:
        return im
    im = im.copy()
    al = im.getchannel("A").point(lambda v: int(v * a))
    im.putalpha(al)
    return im


def put(c, im, x, y, anchor="mm", s=1.0, a=1.0, rot=0):
    """이미지를 (x, y) 기준점에 붙인다. anchor: 가로 l/m/r + 세로 t/m/b."""
    if a <= 0.003 or s <= 0.01:
        return
    if abs(s - 1) > 1e-3:
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BILINEAR)
    if rot:
        im = im.rotate(rot, Image.BILINEAR, expand=True)
    im = with_alpha(im, a)
    ax = {"l": 0, "m": 0.5, "r": 1}[anchor[0]]
    ay = {"t": 0, "m": 0.5, "b": 1}[anchor[1]]
    c.paste(im, (int(round(x - im.width * ax)), int(round(y - im.height * ay))), im)


def text(c, s, x, y, size=40, wt="Bold", col=INK, anchor="mm", a=1.0, s_=1.0, align="left"):
    put(c, T(s, size, wt, col, align), x, y, anchor, s_, a)


def rect(c, x0, y0, x1, y1, fill=None, outline=None, width=3, r=10, a=1.0):
    if a <= 0.003:
        return
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    if x1 <= x0 or y1 <= y0:
        return
    if a >= 0.999:
        ImageDraw.Draw(c).rounded_rectangle((x0, y0, x1, y1), r, fill=fill, outline=outline, width=width)
        return
    pad = width + 2
    lay = Image.new("RGBA", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), (0, 0, 0, 0))
    fa = lambda col: None if col is None else tuple(col[:3]) + (int(255 * a),)
    ImageDraw.Draw(lay).rounded_rectangle((pad, pad, pad + x1 - x0, pad + y1 - y0), r, fill=fa(fill), outline=fa(outline), width=width)
    c.paste(lay, (x0 - pad, y0 - pad), lay)


def line(c, pts, col=INK, width=4, a=1.0):
    if a <= 0.003:
        return
    if a >= 0.999:
        ImageDraw.Draw(c).line(pts, fill=col, width=width, joint="curve")
        return
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(lay).line(pts, fill=tuple(col) + (int(255 * a),), width=width, joint="curve")
    c.paste(lay, (0, 0), lay)


def arrow(c, x0, y0, x1, y1, col=INK, width=6, head=22, p=1.0, a=1.0):
    if p <= 0:
        return
    x1, y1 = lerp(x0, x1, p), lerp(y0, y1, p)
    ang = math.atan2(y1 - y0, x1 - x0)
    bx, by = x1 - head * 0.8 * math.cos(ang), y1 - head * 0.8 * math.sin(ang)
    line(c, [(x0, y0), (bx, by)], col, width, a)
    pts = [(x1, y1),
           (x1 - head * math.cos(ang - 0.45), y1 - head * math.sin(ang - 0.45)),
           (x1 - head * math.cos(ang + 0.45), y1 - head * math.sin(ang + 0.45))]
    if a >= 0.999:
        ImageDraw.Draw(c).polygon(pts, fill=col)
    else:
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(lay).polygon(pts, fill=tuple(col) + (int(255 * a),))
        c.paste(lay, (0, 0), lay)


def circle(c, x, y, r, fill=None, outline=None, width=4, a=1.0):
    if a <= 0.003 or r <= 0:
        return
    if a >= 0.999:
        ImageDraw.Draw(c).ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=width)
        return
    s = int(r * 2 + width * 2 + 4)
    lay = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    fa = lambda col: None if col is None else tuple(col[:3]) + (int(255 * a),)
    o = width + 2
    ImageDraw.Draw(lay).ellipse((o, o, o + 2 * r, o + 2 * r), fill=fa(fill), outline=fa(outline), width=width)
    c.paste(lay, (int(x - r - o), int(y - r - o)), lay)


def pop(t, t0, d=0.4):
    """튀어나오기: (크기 배율, 투명도)."""
    x = prog(t, t0, t0 + d)
    return out_back(x) if x < 1 else 1.0, smooth(prog(t, t0, t0 + d * 0.5))


def fade(t, t0, t1=None, d=0.3):
    a = smooth(prog(t, t0, t0 + d))
    if t1 is not None:
        a *= 1 - smooth(prog(t, t1 - d, t1))
    return a


def count(t, t0, t1, v0, v1):
    return lerp(v0, v1, out_cubic(prog(t, t0, t1)))


# ───────────────────────── 그림 재료 ─────────────────────────
CROPS = {  # 여러 그림이 묶인 이라스토야 파일에서 하나만 (x0, y0, x1, y1)
    "cat": (38, 11, 216, 205), "dog": (135, 497, 522, 879), "car": (24, 391, 327, 602),
    "phone_happy": (636, 531, 777, 760), "phone_sad": (431, 41, 574, 270), "phone_ok": (25, 41, 166, 270),
    "clap_girl": (15, 68, 268, 526), "maru": (12, 247, 467, 693), "batsu": (471, 253, 926, 693),
    "quiz_o": (12, 98, 248, 406), "switch_on": (22, 83, 238, 408), "switch_off": (264, 84, 480, 409),
}
CROP_SRC = {"phone_happy": "phone_face", "phone_sad": "phone_face", "phone_ok": "phone_face",
            "maru": "maru_batsu", "batsu": "maru_batsu", "quiz_o": "quiz_man",
            "switch_on": "switch", "switch_off": "switch"}
_img = {}


def ira(name, h):
    """이라스토야 그림 (흰 배경 제거본), 높이 h 로."""
    k = ("ira", name, h)
    if k not in _img:
        src = CROP_SRC.get(name, name)
        im = Image.open(os.path.join(R, "cut", src + ".png")).convert("RGBA")
        if name in CROPS:
            im = im.crop(CROPS[name])
        bb = im.getchannel("A").getbbox()
        im = im.crop(bb)
        _img[k] = im.resize((int(im.width * h / im.height), h), Image.LANCZOS)
    return _img[k]


def pic(path, size, smooth_=True, border=0, bcol=WHITE):
    """정사각 이미지 (3D-Shapes, 특징 맵)."""
    k = ("pic", path, size, smooth_, border)
    if k not in _img:
        im = Image.open(os.path.join(R, path)).convert("RGBA")
        im = im.resize((size, size), Image.BICUBIC if smooth_ else Image.NEAREST)
        if border:
            b = Image.new("RGBA", (size + 2 * border, size + 2 * border), bcol + (255,))
            b.paste(im, (border, border))
            d = ImageDraw.Draw(b)
            d.rectangle((0, 0, b.width - 1, b.height - 1), outline=LINE, width=2)
            im = b
        _img[k] = im
    return _img[k]


def shadowed(im, off=8, blur=14, alpha=60):
    k = ("sh", id(im))
    if k not in _img:
        a = im.getchannel("A")
        s = Image.new("RGBA", (im.width + 4 * blur, im.height + 4 * blur), (0, 0, 0, 0))
        sh = Image.new("RGBA", im.size, (20, 30, 50, alpha))
        sh.putalpha(a.point(lambda v: v * alpha // 255))
        s.paste(sh, (2 * blur, 2 * blur + off), sh)
        s = s.filter(ImageFilter.GaussianBlur(blur))
        s.paste(im, (2 * blur, 2 * blur), im)
        _img[k] = s
    return _img[k]


MAIN = np.array(Image.open(os.path.join(R, "shapes", "main.png")).convert("RGB"))  # 64x64 예시 (빨간 원기둥)
SHAPES = [f"shapes/s{i:02d}.png" for i in range(60)]


def background():
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    for x in range(0, W, 48):
        d.line([(x, 0), (x, H)], fill=GRID, width=1)
    for y in range(0, H, 48):
        d.line([(0, y), (W, y)], fill=GRID, width=1)
    return im


BGIM = background()


def chapter(c, num, title, t):
    """왼쪽 위 장 표시."""
    a = fade(t, 0.0, d=0.35)
    dx = (1 - out_cubic(prog(t, 0, 0.45))) * -40
    rect(c, 60 + dx, 44, 124 + dx, 108, fill=BLUE, r=8, a=a)
    text(c, str(num), 92 + dx, 76, 38, "ExtraBold", WHITE, a=a)
    text(c, title, 146 + dx, 76, 40, "Bold", INK, "lm", a=a)


def card(c, x, y, w, h, label, value, t, t0, icon=None, vcol=INK):
    s, a = pop(t, t0)
    lay = Image.new("RGBA", (w + 8, h + 8), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    d.rounded_rectangle((4, 4, w + 3, h + 3), 10, fill=WHITE, outline=LINE, width=2)
    lab = T(label, 30, "Medium", SUB)
    lay.paste(lab, (34, (h + 8 - lab.height) // 2), lab)
    val = T(value, 44, "Bold", vcol)
    lay.paste(val, (160, (h + 8 - val.height) // 2), val)
    if icon is not None:
        icon(lay, w - 70, (h + 8) // 2)
    put(c, lay, x, y, "lm", s, a)


# ───────────────────────── 장면 ─────────────────────────
def s_title(c, t):
    s, a = pop(t, 0.3, 0.5)
    put(c, ira("robot_pc", 430), 1430, 520, "mm", s, a)
    for i, (ln, col) in enumerate([("숫자를 짧게 줄여도", INK), ("AI는 정답을 맞힐까?", BLUE)]):
        p = out_cubic(prog(t, 0.4 + 0.3 * i, 1.0 + 0.3 * i))
        text(c, ln, 150 - 60 * (1 - p), 400 + 128 * i, 104, "ExtraBold", col, "lm", a=p)
    p = out_cubic(prog(t, 1.2, 1.9))
    rect(c, 156, 612, 156 + 520 * p, 624, fill=ORANGE, r=2)
    text(c, "층별 양자화 실험, 쉽게 보기", 156, 690, 48, "SemiBold", SUB, "lm", a=fade(t, 1.5))
    text(c, "컴퓨터비전 기말 프로젝트   김진서 · 이하형 · 정채호준", 156, 940, 32, "Medium", SUB, "lm", a=fade(t, 2.0))


def size_icon(lay, x, y):
    d = ImageDraw.Draw(lay)
    for i in range(8):
        hh = 10 + i * 4
        d.rectangle((x - 50 + i * 13, y + 22 - hh, x - 41 + i * 13, y + 22), fill=BLUE if i < 6 else LINE)


def color_icon(lay, x, y):
    ImageDraw.Draw(lay).rounded_rectangle((x - 26, y - 26, x + 26, y + 26), 6, fill=(220, 30, 30))


def shape_icon(lay, x, y):
    d = ImageDraw.Draw(lay)
    d.rectangle((x - 22, y - 18, x + 22, y + 22), fill=(120, 130, 145))
    d.ellipse((x - 22, y - 30, x + 22, y - 8), fill=(170, 178, 190))


def s_quiz(c, t):
    chapter(c, 1, "실험 과제", t)
    out = smooth(prog(t, 9.0, 9.6))
    s, a = pop(t, 0.3, 0.5)
    put(c, shadowed(pic("shapes/main.png", 500, border=8)), 560 - 300 * out, 520, "mm", s * (1 - 0.4 * out), a * (1 - out))
    if t < 9.6:
        rows = [("모양", "원기둥", shape_icon), ("색", "빨강", color_icon), ("크기", "8단계 중 6", size_icon)]
        for i, (l, v, ic) in enumerate(rows):
            x = 1000 + 400 * out
            card(c, x, 330 + 170 * i, 640, 130, l, v, t, 4.6 + 0.8 * i, ic)
            if 0.5 < out:
                pass
        # 물음표 → 화살표
        arrow(c, 860, 520, 960, 520, BLUE, 8, 26, out_cubic(prog(t, 4.2, 4.6)), a=1 - out)
    # 데이터셋 그림들이 물결처럼
    if t >= 9.2:
        cols, rows_ = 10, 4
        for k in range(cols * rows_):
            r_, q = divmod(k, cols)
            s2, a2 = pop(t, 9.4 + (q + r_) * 0.06, 0.35)
            put(c, pic(SHAPES[k], 150, border=3), 200 + q * 169, 300 + r_ * 168, "mm", s2, a2)
        text(c, "3D Shapes 데이터: 그림 48만 장", 960, 990 - 40, 34, "SemiBold", SUB, a=fade(t, 11.5)) if False else None


def s_learn(c, t):
    chapter(c, 1, "실험 과제", t)
    s, a = pop(t, 0.2, 0.5)
    put(c, ira("ai_learn", 520), 480, 540, "mm", s, a)
    # 그림이 날아와 AI 로 들어감 (0.8 ~ 5.6초)
    for k in range(18):
        t0 = 0.8 + k * 0.27
        p = prog(t, t0, t0 + 1.1)
        if 0 < p < 1:
            e = smooth(p)
            x = lerp(1750, 560, e)
            y = lerp(520 + (k % 4) * 110, 420, e) - math.sin(e * math.pi) * 90
            put(c, pic(SHAPES[(k * 7) % 60], 130, border=3), x, y, "mm", 1 - 0.6 * e, 1 - smooth(prog(p, 0.8, 1)))
    n = count(t, 0.8, 5.8, 0, 60000)
    a1 = fade(t, 0.6, 6.4)
    text(c, "공부한 그림", 1380, 210, 40, "SemiBold", SUB, "mm", a1)
    text(c, f"{int(n):,}장", 1380, 300, 96, "ExtraBold", BLUE, "mm", a1)
    # 시험
    if t > 6.2:
        a2 = fade(t, 6.4)
        text(c, "처음 보는 그림", 1340, 290, 40, "SemiBold", SUB, "mm", a2)
        n2 = count(t, 6.6, 9.0, 0, 10000)
        text(c, f"{int(n2):,}장 중", 1340, 375, 72, "ExtraBold", INK, "mm", a2)
        n3 = count(t, 7.2, 9.6, 0, 10000)
        text(c, f"{int(n3):,}장 정답", 1340, 470, 72, "ExtraBold", GREEN, "mm", a2)
        s3, a3 = pop(t, 10.0, 0.45)
        # 도장
        lay = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        d.ellipse((10, 10, 290, 290), outline=RED, width=14)
        tt = T("100%", 92, "ExtraBold", RED)
        lay.paste(tt, ((300 - tt.width) // 2, (300 - tt.height) // 2 - 8), tt)
        put(c, lay, 1340, 720, "mm", s3 * 1.0, a3 * 0.95, rot=-12)
        s4, a4 = pop(t, 10.5, 0.4)
        put(c, ira("clap_girl", 300), 1720, 760, "mm", s4, a4)


def s_pixels(c, t):
    chapter(c, 2, "컴퓨터는 그림을 숫자로 봐요", t)
    # 확대: 전체 64x64 → 6x6 영역 (가로 22~28, 세로 24~30: 원기둥 왼쪽 테두리)
    X0, Y0, N = 20, 26, 6
    z = smooth(prog(t, 0.8, 3.2))
    fx0, fy0 = lerp(0, X0, z), lerp(0, Y0, z)
    fw = lerp(64, N, z)
    S = 600
    big = Image.fromarray(MAIN).resize((64 * 40, 64 * 40), Image.NEAREST if z > 0.3 else Image.BICUBIC)
    box = (int(fx0 * 40), int(fy0 * 40), int((fx0 + fw) * 40), int((fy0 + fw) * 40))
    view = big.crop(box).resize((S, S), Image.NEAREST).convert("RGBA")
    a0 = fade(t, 0.1)
    cx, cy = 520, 560
    put(c, view, cx, cy, "mm", 1, a0)
    rect(c, cx - S / 2 - 3, cy - S / 2 - 3, cx + S / 2 + 3, cy + S / 2 + 3, outline=INK, width=3, r=2, a=a0)
    # 칸과 숫자
    cell = S / N
    ga = fade(t, 3.3)
    if ga > 0:
        for i in range(1, N):
            line(c, [(cx - S / 2 + i * cell, cy - S / 2), (cx - S / 2 + i * cell, cy + S / 2)], WHITE, 3, ga)
            line(c, [(cx - S / 2, cy - S / 2 + i * cell), (cx + S / 2, cy - S / 2 + i * cell)], WHITE, 3, ga)
    for k in range(N * N):
        j, i = divmod(k, N)
        tk = 4.6 + k * 0.05
        if t >= tk:
            r_, g_, b_ = MAIN[Y0 + j, X0 + i]
            lum = 0.3 * r_ + 0.59 * g_ + 0.11 * b_
            col = INK if lum > 140 else WHITE
            x, y = cx - S / 2 + (i + 0.5) * cell, cy - S / 2 + (j + 0.5) * cell
            put(c, T(f"{r_}\n{g_}\n{b_}", 22, "SemiBold", col, "center", 1.15), x, y, "mm", 1, fade(t, tk, d=0.2))
    # 오른쪽: 한 칸 크게
    if t > 7.0:
        j, i = 2, 2
        r_, g_, b_ = [int(v) for v in MAIN[Y0 + j, X0 + i]]
        sx, sy = cx - S / 2 + (i + 0.5) * cell, cy - S / 2 + (j + 0.5) * cell
        p = out_cubic(prog(t, 7.0, 7.8))
        bx, by = lerp(sx, 1150, p), lerp(sy, 420, p)
        bs = lerp(cell, 220, p)
        rect(c, bx - bs / 2, by - bs / 2, bx + bs / 2, by + bs / 2, fill=(r_, g_, b_), outline=INK, width=3, r=4)
        a3 = fade(t, 7.8)
        for q, (nm, v, colr) in enumerate([("빨강", r_, (220, 40, 40)), ("초록", g_, (40, 160, 70)), ("파랑", b_, (40, 90, 220))]):
            y = 340 + q * 80
            text(c, nm, 1300, y, 36, "SemiBold", INK, "lm", a3)
            pr = out_cubic(prog(t, 8.0 + q * 0.3, 8.8 + q * 0.3))
            rect(c, 1400, y - 16, 1400 + 300, y + 16, fill=(232, 235, 240), r=4, a=a3)
            rect(c, 1400, y - 16, 1400 + 300 * v / 255 * pr, y + 16, fill=colr, r=4, a=a3)
            text(c, f"{int(v * pr)}", 1730, y, 36, "Bold", INK, "lm", a3)
        text(c, "숫자는 0부터 255까지", 1300, 590, 30, "Medium", SUB, "lm", fade(t, 9.5))
    if t > 11.2:
        s5, a5 = pop(t, 11.4)
        text(c, "64 × 64 × 3", 1420, 740, 64, "ExtraBold", INK, "mm", a5, s5)
        s6, a6 = pop(t, 12.2)
        text(c, "= 숫자 12,288개", 1420, 830, 56, "Bold", BLUE, "mm", a6, s6)


# 합성곱(convolution) 장면: 밝기를 0~9로 줄인 8x8 판 위를 3x3 필터가 이동
_gray = (0.3 * MAIN[..., 0] + 0.59 * MAIN[..., 1] + 0.11 * MAIN[..., 2])
CONV_IN = np.round(_gray[30:38, 39:47] / 255 * 9).astype(int)
FILT = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
CONV_OUT = np.array([[(CONV_IN[y:y + 3, x:x + 3] * FILT).sum() for x in range(6)] for y in range(6)])


def s_conv(c, t):
    chapter(c, 2, "AI 안에는 층이 있어요", t)
    # 0~5초: 층 6개 소개
    if t < 5.4:
        a = 1 - smooth(prog(t, 4.8, 5.4))
        for i in range(6):
            s, a1 = pop(t, 0.6 + 0.25 * i)
            x = 360 + i * 240
            lay = Image.new("RGBA", (200, 260), (0, 0, 0, 0))
            d = ImageDraw.Draw(lay)
            d.rounded_rectangle((4, 4, 196, 256), 12, fill=LBLUE, outline=BLUE, width=4)
            tt = T(f"{i + 1}번 층", 38, "Bold", BLUE)
            lay.paste(tt, ((200 - tt.width) // 2, 110), tt)
            put(c, lay, x, 520, "mm", s, a1 * a)
        s, a2 = pop(t, 0.2)
        put(c, ira("ai", 170), 960, 260, "mm", s, a2 * a)
        return
    cell = 74
    ix, iy = 200, 300   # 입력 판 왼쪽 위
    ox, oy = 1210, 374  # 출력 판 왼쪽 위
    a = fade(t, 5.4)
    text(c, "그림의 밝기 (0~9)", ix + 4 * cell, iy - 50, 32, "SemiBold", SUB, "mm", a)
    for y in range(8):
        for x in range(8):
            v = CONV_IN[y, x]
            g = int(40 + v * 23)
            rect(c, ix + x * cell, iy + y * cell, ix + (x + 1) * cell - 3, iy + (y + 1) * cell - 3, fill=(g, g, g), r=4, a=a)
            text(c, str(v), ix + (x + 0.5) * cell, iy + (y + 0.5) * cell - 2, 30, "Bold", INK if v > 5 else WHITE, "mm", a)
    # 필터 소개 (5.4~9초): 오른쪽 위에 필터 판
    fa = fade(t, 6.0)
    fx, fy = 860, 330
    text(c, "필터", fx + 1.5 * 60, fy - 50, 34, "Bold", ORANGE, "mm", fa)
    for y in range(3):
        for x in range(3):
            v = FILT[y, x]
            rect(c, fx + x * 60, fy + y * 60, fx + (x + 1) * 60 - 4, fy + (y + 1) * 60 - 4, fill=(255, 236, 214), outline=ORANGE, width=3, r=4, a=fa)
            text(c, f"{v:+d}" if v else "0", fx + (x + 0.5) * 60, fy + (y + 0.5) * 60 - 2, 28, "Bold", INK, "mm", fa)
    text(c, "새 그림", ox + 3 * cell, oy - 124, 32, "SemiBold", SUB, "mm", fade(t, 9.0))
    # 이동 (9~20초, 36칸)
    k_now = int(clamp((t - 9.0) / 11.0) * 36) if t >= 9.0 else -1
    for k in range(max(0, min(k_now + 1, 36))):
        y, x = divmod(k, 6)
        v = CONV_OUT[y, x]
        col = mix((250, 250, 250), (30, 30, 30), clamp(abs(v) / 20)) if v != 0 else (240, 240, 240)
        col = (255, 245, 235) if v == 0 else mix((255, 255, 255), ORANGE, clamp(abs(v) / 18))
        rect(c, ox + x * cell, oy - 74 + y * cell, ox + (x + 1) * cell - 3, oy - 74 + (y + 1) * cell - 3, fill=col, r=4)
        text(c, str(v), ox + (x + 0.5) * cell, oy - 74 + (y + 0.5) * cell - 2, 28, "Bold", INK, "mm")
    if 0 <= k_now < 36:
        y, x = divmod(k_now, 6)
        rect(c, ix + x * cell - 4, iy + y * cell - 4, ix + (x + 3) * cell + 1, iy + (y + 3) * cell + 1, outline=ORANGE, width=7, r=6)
        rect(c, ox + x * cell - 3, oy - 74 + y * cell - 3, ox + (x + 1) * cell, oy - 74 + (y + 1) * cell, outline=ORANGE, width=6, r=6)
        arrow(c, ix + (x + 3) * cell + 10, iy + (y + 1.5) * cell, ox + x * cell - 14, oy - 74 + (y + 0.5) * cell, ORANGE, 4, 18, 1, 0.55)
        text(c, "곱하고 모두 더하기", 960, 900, 34, "Bold", ORANGE, "mm")
    if t > 20.2:
        # 결과: 테두리 열이 크게 나옴
        a2 = fade(t, 20.3)
        text(c, "밝기가 바뀌는 곳(테두리)에서 큰 숫자", ox + 3 * cell, oy + 6 * cell - 40, 30, "Bold", ORANGE, "mm", a2)


def s_layers(c, t):
    chapter(c, 2, "AI 안에는 층이 있어요", t)
    s, a = pop(t, 0.2)
    put(c, pic("shapes/main.png", 200, border=5), 170, 560, "mm", s, a)
    xs = [430 + i * 215 for i in range(6)]
    for i, x in enumerate(xs):
        lit = t > 1.4 + i * 1.3
        s1, a1 = pop(t, 0.4 + 0.12 * i)
        lay = Image.new("RGBA", (150, 190), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        d.rounded_rectangle((3, 3, 147, 187), 10, fill=BLUE if lit else LBLUE, outline=BLUE, width=4)
        tt = T(f"{i + 1}번\n층", 34, "Bold", WHITE if lit else BLUE, "center", 1.2)
        lay.paste(tt, ((150 - tt.width) // 2, (190 - tt.height) // 2), tt)
        put(c, lay, x, 560, "mm", s1, a1)
        if i < 5:
            arrow(c, x + 80, 560, x + 133, 560, GREY, 5, 16, 1, a1)
        # 특징 맵이 위로 튀어나옴
        s2, a2 = pop(t, 1.5 + i * 1.3)
        put(c, pic(f"fm/fm_fp_{i + 1}_0.png", 150, smooth_=False, border=3, bcol=INK), x, 340, "mm", s2, a2)
    arrow(c, 280, 560, 345, 560, GREY, 5, 16, 1, a)
    # 이동하는 점
    p = prog(t, 1.0, 9.0)
    if 0 < p < 1:
        x = lerp(270, xs[-1] + 90, p)
        circle(c, x, 690, 14, fill=ORANGE)
    text(c, "층마다 만든 새 그림", xs[0] + (xs[-1] - xs[0]) / 2, 225, 32, "SemiBold", SUB, "mm", fade(t, 2.0))
    # 앞 · 뒤 설명
    a3 = fade(t, 10.2)
    if a3 > 0:
        rect(c, xs[0] - 90, 760, xs[1] + 90, 830, fill=WHITE, outline=LINE, width=2, r=8, a=a3)
        text(c, "테두리 · 색 같은 단순한 모양", (xs[0] + xs[1]) / 2, 795, 30, "SemiBold", INK, "mm", a3)
        a4 = fade(t, 12.0)
        rect(c, xs[4] - 90, 760, xs[5] + 90, 830, fill=WHITE, outline=LINE, width=2, r=8, a=a4)
        text(c, "작고 간단하게 정리된 정보", (xs[4] + xs[5]) / 2, 795, 30, "SemiBold", INK, "mm", a4)
    # 정답 카드
    s5, a5 = pop(t, 16.4, 0.45)
    lay = Image.new("RGBA", (300, 250), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    d.rounded_rectangle((3, 3, 297, 247), 12, fill=WHITE, outline=GREEN, width=5)
    for q, l in enumerate(["원기둥", "빨강", "크기 6"]):
        tt = T(l, 46, "Bold", INK)
        lay.paste(tt, ((300 - tt.width) // 2, 28 + q * 70), tt)
    put(c, lay, 1770, 560, "mm", s5, a5)
    if a5 > 0:
        arrow(c, xs[-1] + 80, 560, 1612, 560, GREEN, 6, 20, out_cubic(prog(t, 16.0, 16.5)))


LONG = [["0.38127491", "-1.20473365", "0.07719032"],
        ["0.91540127", "-0.33081946", "1.42208863"],
        ["-0.65392718", "0.21876054", "-0.04417290"]]


def s_long(c, t):
    chapter(c, 3, "숫자가 너무 길어요", t)
    out = smooth(prog(t, 8.0, 8.6))
    a = fade(t, 0.2) * (1 - out)
    if a > 0:
        cw, ch = 300, 90
        x0, y0 = 960 - 1.5 * cw, 260
        text(c, "필터 속 진짜 숫자", 960, 210, 34, "SemiBold", SUB, "mm", a)
        for y in range(3):
            for x in range(3):
                k = y * 3 + x
                tk = 0.4 + k * 0.28
                s_ = LONG[y][x]
                n = int(clamp((t - tk) / 0.5) * len(s_))
                rect(c, x0 + x * cw + 6, y0 + y * ch + 6, x0 + (x + 1) * cw - 6, y0 + (y + 1) * ch - 6, fill=WHITE, outline=ORANGE, width=3, r=6, a=a)
                if n > 0:
                    text(c, s_[:n], x0 + (x + 0.5) * cw, y0 + (y + 0.5) * ch, 38, "SemiBold", INK, "mm", a)
        # 32비트 칸
        if t > 4.0:
            bits = "00111110110000110011100100110101"
            text(c, "0.38127491 을 저장하는 데 필요한 칸", 960, 640, 34, "SemiBold", SUB, "mm", fade(t, 4.0) * (1 - out))
            for i, b in enumerate(bits):
                tk = 4.3 + i * 0.06
                if t >= tk:
                    x = 960 - 16 * 46 / 2 * 2 + i * 46 + 23
                    s1, a1 = pop(t, tk, 0.25)
                    lay = Image.new("RGBA", (42, 56), (0, 0, 0, 0))
                    ImageDraw.Draw(lay).rounded_rectangle((1, 1, 41, 55), 5, fill=BLUE if b == "1" else WHITE, outline=BLUE, width=2)
                    tt = T(b, 28, "Bold", WHITE if b == "1" else BLUE)
                    lay.paste(tt, ((42 - tt.width) // 2, (56 - tt.height) // 2), tt)
                    put(c, lay, x, 730, "mm", s1, a1 * (1 - out))
            s2, a2 = pop(t, 6.4)
            text(c, "32칸 = 32비트", 960, 830, 52, "ExtraBold", BLUE, "mm", a2 * (1 - out), s2)
    if t > 8.4:
        s1, a1 = pop(t, 8.6)
        text(c, "ResNet-18 모델 속 숫자", 960, 260, 40, "SemiBold", SUB, "mm", a1)
        n = count(t, 8.8, 11.5, 0, 11689512)
        text(c, f"{int(n):,}개", 960, 360, 110, "ExtraBold", INK, "mm", a1)
        s2, a2 = pop(t, 12.4)
        put(c, ira("heavy", 400), 640, 700, "mm", s2, a2)
        s3, a3 = pop(t, 13.2)
        put(c, ira("phone_sad", 330), 1280, 700, "mm", s3, a3)
        text(c, "무겁고 느림", 1280, 900, 40, "Bold", RED, "mm", fade(t, 14.0))


# 눈금 장면용 값 (0~1 사이)
VALS = [0.08, 0.215, 0.381, 0.47, 0.623, 0.749, 0.905]
VCOL = [(66, 133, 244), (52, 168, 83), (234, 67, 53), (156, 39, 176), (0, 150, 136), (255, 152, 0), (121, 85, 72)]


def ruler(c, y, n, label, t, t_show, t_drop, show_err=False):
    a = fade(t, t_show)
    if a <= 0:
        return
    x0, x1 = 300, 1620
    text(c, label, 150, y, 36, "Bold", INK, "mm", a, align="center")
    line(c, [(x0, y), (x1, y)], INK, 4, a)
    for i in range(n):
        x = lerp(x0, x1, i / (n - 1))
        big = n <= 16 or i % 32 == 0
        line(c, [(x, y), (x, y - (26 if big else 12))], INK, 3 if n <= 16 else 1, a)
    for k, v in enumerate(VALS):
        sx = lerp(x0, x1, v)
        q = round(v * (n - 1)) / (n - 1)
        qx = lerp(x0, x1, q)
        d = smooth(prog(t, t_drop + k * 0.12, t_drop + k * 0.12 + 0.6))
        px = lerp(sx, qx, d)
        py = lerp(y - 120, y - 40, d)
        if show_err and d > 0.95 and abs(qx - sx) > 3:
            ea = fade(t, t_drop + 1.6)
            line(c, [(sx, y - 120), (sx, y - 40)], GREY, 2, ea * 0.6)
            arrow(c, sx, y - 70, qx, y - 70, ORANGE, 4, 14, 1, ea) if abs(qx - sx) > 18 else line(c, [(sx, y - 70), (qx, y - 70)], ORANGE, 5, ea)
            circle(c, sx, y - 120, 13, outline=VCOL[k], width=3, a=ea * 0.6)
        circle(c, px, py, 15, fill=VCOL[k], a=a)


def s_quant(c, t):
    chapter(c, 4, "숫자를 짧게: 양자화", t)
    # A 0~7: 0.38127491 → 0.4
    if t < 7.4:
        a = 1 - smooth(prog(t, 6.8, 7.4))
        full = "0.38127491"
        d = smooth(prog(t, 2.0, 3.4))
        f = font("ExtraBold", 150)
        x = 960 - f.getlength(full) / 2
        for i, ch in enumerate(full):
            w = f.getlength(ch)
            if i < 3:
                col = INK
                text(c, ch if not (i == 2 and t > 3.6) else "4", x + w / 2 + (f.getlength(full) - f.getlength("0.4")) / 2 * d, 440, 150, "ExtraBold", BLUE if (i == 2 and t > 3.6) else INK, "mm", a)
            else:
                text(c, ch, x + w / 2, 440 + 300 * d * (0.6 + 0.1 * i), 150, "ExtraBold", GREY, "mm", a * (1 - d))
            x += w
        s, a2 = pop(t, 4.2)
        text(c, "양자화 (quantization)", 960, 640, 64, "Bold", ORANGE, "mm", a2 * a, s)
        text(c, "숫자를 정해진 칸 중 가까운 값으로 바꾸기", 960, 730, 36, "Medium", SUB, "mm", fade(t, 4.8) * a)
        return
    # B 7.4~19: 두 자
    if t < 19.6:
        a = 1 - smooth(prog(t, 19.0, 19.6))
        if a < 1:
            pass
        c2 = c if a >= 0.999 else c.copy()
        ruler(c2, 420, 256, "8비트\n눈금 256개", t, 7.6, 8.8)
        ruler(c2, 760, 16, "4비트\n눈금 16개", t, 10.2, 11.4, show_err=True)
        if t > 13.6:
            s, a3 = pop(t, 13.8)
            text(c2, "이 차이 = 오차", 1180, 840, 44, "Bold", ORANGE, "mm", a3, s)
        put(c2, ira("ruler", 150), 1760, 250, "mm", *pop(t, 7.8)) if t > 7.8 else None
        if a < 0.999:
            c.paste(Image.blend(BGIM, c2, a))
        return
    # C 19.6~28: 크기 막대
    rows = [("32비트", 1.0, "원래 크기", GREY), ("8비트", 0.25, "1/4", BLUE), ("4비트", 0.125, "1/8", ORANGE)]
    for i, (lab, v, note, col) in enumerate(rows):
        y = 330 + i * 150
        a = fade(t, 19.8 + 0.4 * i)
        text(c, lab, 260, y, 44, "Bold", INK, "rm", a)
        p = out_cubic(prog(t, 20.0 + 0.4 * i, 21.0 + 0.4 * i))
        rect(c, 300, y - 40, 300 + 1100 * v * p, y + 40, fill=col, r=6, a=a)
        text(c, note, 300 + 1100 * v * p + 24, y, 40, "Bold", col if col != GREY else SUB, "lm", fade(t, 21.0 + 0.4 * i))
    text(c, "모델을 저장하는 데 필요한 크기", 300, 230, 32, "SemiBold", SUB, "lm", fade(t, 19.8))
    s, a = pop(t, 24.0)
    put(c, ira("worry_boy", 300), 1560, 800, "mm", s, a)


def answer_col(c, x, y, title, img, label, ok, t, t0, acc=None, t_acc=None, acc_col=INK, title_col=INK):
    a = fade(t, t0)
    text(c, title, x, y - 250, 42, "Bold", title_col, "mm", a)
    s, a1 = pop(t, t0 + 0.2)
    put(c, pic(f"fm/{img}", 360, border=6), x, y, "mm", s, a1)
    text(c, label, x, y + 225, 46, "Bold", GREEN if ok else RED, "mm", fade(t, t0 + 0.6))
    s2, a2 = pop(t, t0 + 0.9, 0.35)
    put(c, stamp(ok), x + 160, y - 160, "mm", s2, a2)
    if acc is not None:
        v = count(t, t_acc, t_acc + 1.4, 100 if acc < 100 else 0, acc)
        text(c, f"{v:.0f}%", x, y + 330, 80, "ExtraBold", acc_col, "mm", fade(t, t_acc))


_st = {}


def stamp(ok):
    if ok in _st:
        return _st[ok]
    lay = Image.new("RGBA", (130, 130), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    if ok:
        d.ellipse((10, 10, 120, 120), outline=GREEN, width=16)
    else:
        d.line((18, 18, 112, 112), fill=RED, width=18)
        d.line((112, 18, 18, 112), fill=RED, width=18)
    _st[ok] = lay
    return lay


def s_result(c, t):
    chapter(c, 5, "줄인 모델로 시험", t)
    answer_col(c, 400, 450, "원래 모델 (32비트)", "ans_1_0_5.png", "빨간 원기둥", True, t, 0.4, 100, 13.4)
    answer_col(c, 960, 450, "8비트", "ans_1_0_5.png", "빨간 원기둥", True, t, 4.2, 100, 13.6)
    answer_col(c, 1520, 450, "4비트", "ans_3_2_5.png", "노란 캡슐", False, t, 8.2, ACC_W4, 13.8, RED)
    text(c, "그림: AI가 고른 답", 960, 140, 30, "Medium", SUB, "mm", fade(t, 1.0))


def layer_row(c, t, y, states, t0=0.0, labels=True, w=200, h=170, gap=250, x0=None):
    xs = [(x0 or 960 - 2.5 * gap) + i * gap for i in range(6)]
    for i, x in enumerate(xs):
        s, a = pop(t, t0 + 0.1 * i)
        st = states[i]
        fill, out_, tcol = {"q": ((228, 231, 236), GREY, SUB), "fp": (BLUE, BLUE, WHITE), "?": (LBLUE, BLUE, BLUE)}[st]
        lay = Image.new("RGBA", (w + 8, h + 8), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        d.rounded_rectangle((4, 4, w + 3, h + 3), 12, fill=fill, outline=out_, width=4)
        t1 = T(f"{i + 1}번 층", 32, "Bold", tcol)
        lay.paste(t1, ((w + 8 - t1.width) // 2, 30), t1)
        sub = {"q": "4비트", "fp": "원래 숫자", "?": "?"}[st]
        t2 = T(sub, 30 if st != "?" else 52, "SemiBold" if st != "?" else "ExtraBold", tcol)
        lay.paste(t2, ((w + 8 - t2.width) // 2, 88 if st != "?" else 72), t2)
        put(c, lay, x, y, "mm", s, a)
    return xs


def s_who(c, t):
    chapter(c, 7, "어느 층 때문일까?", t)
    xs = layer_row(c, t, 420, ["?"] * 6, 0.3)
    s, a = pop(t, 1.2)
    # 탐정이 돋보기를 들고 층들을 따라 이동
    p = smooth(prog(t, 1.4, 5.0))
    put(c, ira("detective", 300), lerp(xs[0], xs[-1], p), 640, "mm", s, a)
    for i, (n, l) in enumerate([("방법 1", "층마다 오차 재기"), ("방법 2", "한 층씩 되돌리기")]):
        s2, a2 = pop(t, 5.8 + 0.8 * i)
        lay = Image.new("RGBA", (620, 120), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        d.rounded_rectangle((3, 3, 617, 117), 12, fill=WHITE, outline=BLUE, width=4)
        t1 = T(n, 40, "ExtraBold", BLUE)
        lay.paste(t1, (34, (120 - t1.height) // 2), t1)
        t2 = T(l, 40, "Bold", INK)
        lay.paste(t2, (200, (120 - t2.height) // 2), t2)
        put(c, lay, 620 + 680 * i, 850, "mm", s2, a2)


def bars(c, t, vals, x0, gap, base, scale, t0, col_fn, fmt, w=110, labels=True, lab_y=None, dt=0.35):
    for i, v in enumerate(vals):
        x = x0 + i * gap
        p = out_cubic(prog(t, t0 + dt * i, t0 + dt * i + 0.8))
        hgt = v * scale * p
        col = col_fn(i)
        if hgt >= 0:
            rect(c, x - w / 2, base - hgt, x + w / 2, base, fill=col, r=6)
        else:
            rect(c, x - w / 2, base, x + w / 2, base - hgt, fill=col, r=6)
        if p > 0.05:
            ty = base - hgt - 30 if v >= 0 else base - 30
            text(c, fmt(v), x, ty, 36, "Bold", INK, "mm", fade(t, t0 + dt * i + 0.5))
        if labels:
            text(c, f"{i + 1}번 층", x, lab_y or base + 40, 32, "SemiBold", INK, "mm", fade(t, t0 - 0.3))


def s_method1(c, t):
    chapter(c, 6, "방법 1: 층마다 오차 재기", t)
    x0, gap = 360, 240
    for i in range(6):
        x = x0 + i * gap
        s, a = pop(t, 0.6 + 0.3 * i)
        put(c, pic(f"fm/fm_fp_{i + 1}_0.png", 100, smooth_=False, border=2, bcol=INK), x - 55, 250, "mm", s, a)
        put(c, pic(f"fm/fm_q4_{i + 1}_0.png", 100, smooth_=False, border=2, bcol=INK), x + 55, 250, "mm", s, a)
    a = fade(t, 0.8)
    text(c, "원래", x0 - 55, 175, 26, "SemiBold", SUB, "mm", a)
    text(c, "4비트", x0 + 55, 175, 26, "SemiBold", SUB, "mm", a)
    base = 860
    line(c, [(x0 - 130, base), (x0 + 5 * gap + 130, base)], INK, 3, fade(t, 5.6))
    text(c, "오차", x0 - 200, base - 200, 34, "Bold", SUB, "mm", fade(t, 5.6))
    hi = t > 11.2
    bars(c, t, NMSE_ALL, x0, gap, base, 360, 6.0,
         lambda i: ORANGE if (hi and i == 5) else (190, 196, 206), lambda v: f"{v:.2f}", lab_y=base + 36)
    if t > 15.6:
        s, a = pop(t, 15.8)
        put(c, ira("detective_point", 300), x0 + 3.3 * gap, 520, "mm", s, a)
        s2, a2 = pop(t, 16.4)
        text(c, "6번 층이 범인?", x0 + 1.6 * gap, 470, 60, "ExtraBold", RED, "mm", a2, s2)


def s_method2(c, t):
    chapter(c, 7, "방법 2: 한 층만 되돌리기", t)
    up = smooth(prog(t, 22.0, 22.8))
    if up < 1:
        st = ["q"] * 6
        if 6.4 <= t < 14.0:
            st[5] = "fp"
        if t >= 14.4:
            st[0] = "fp"
        a0 = 1 - up
        c2 = c.copy() if a0 < 1 else c
        xs = layer_row(c2, t, 280, st, 0.3)
        # 스위치 그림
        for i, x in enumerate(xs):
            pass
        # 결과 판
        wrong = t < 15.2
        img = "ans_3_2_5.png" if wrong else "ans_1_0_5.png"
        a1 = fade(t, 1.0)
        put(c2, pic(f"fm/{img}", 300, border=6), 760, 640, "mm", 1, a1)
        arrow(c2, 960, 385, 960, 450, GREY, 5, 18, 1, a1)
        text(c2, "노란 캡슐" if wrong else "빨간 원기둥", 760, 830, 44, "Bold", RED if wrong else GREEN, "mm", a1)
        ok_t = 15.2
        if wrong:
            put(c2, stamp(False), 900, 510, "mm", 1, a1)
        else:
            s, a = pop(t, ok_t, 0.35)
            put(c2, stamp(True), 900, 510, "mm", s, a)
        # 정확도
        if t < 7.6:
            acc, dlt, dcol = ACC_W4, None, INK
        elif t < 14.0:
            acc = count(t, 7.6, 9.0, ACC_W4, ACC_W4 + GAIN[5]); dlt, dcol = f"{GAIN[5]:+.1f}%p", SUB
        elif t < 15.2:
            acc, dlt, dcol = ACC_W4, None, INK
        else:
            acc = count(t, 15.4, 17.0, ACC_W4, ACC_W4 + GAIN[0]); dlt, dcol = f"+{GAIN[0]:.1f}%p", GREEN
        text(c2, "정확도", 1240, 600, 36, "SemiBold", SUB, "mm", a1)
        text(c2, f"{acc:.1f}%", 1240, 690, 96, "ExtraBold", INK, "mm", a1)
        if dlt:
            text(c2, dlt, 1240, 790, 52, "Bold", dcol, "mm", fade(t, 8.6 if t < 14 else 16.4))
        # 스위치 손가락 효과
        for tk, i in [(6.4, 5), (14.4, 0)]:
            if tk <= t < tk + 0.6:
                circle(c2, xs[i], 280, 20 + 80 * prog(t, tk, tk + 0.6), outline=ORANGE, width=6, a=1 - prog(t, tk, tk + 0.6))
        if a0 < 1:
            c.paste(Image.blend(BGIM, c2, a0))
    if t > 22.4:
        base = 640
        x0, gap = 360, 240
        line(c, [(x0 - 130, base), (x0 + 5 * gap + 130, base)], INK, 3, fade(t, 22.6))
        text(c, "되돌렸을 때\n정확도 변화 (%p)", x0 - 130, 230, 30, "Bold", SUB, "lm", fade(t, 22.6))
        hi = t > 26.4
        bars(c, t, GAIN, x0, gap, base, 17, 23.0,
             lambda i: GREEN if (hi and i == 0) else ((190, 196, 206) if GAIN[i] >= 0 else (230, 160, 160)),
             lambda v: f"{v:+.1f}", lab_y=base + 70)
        if t > 26.4:
            s, a = pop(t, 26.6)
            put(c, ira("surprised", 260), 1660, 300, "mm", s, a)


def s_why(c, t):
    chapter(c, 8, "왜 다를까?", t)
    if t < 8.4:
        a = 1 - smooth(prog(t, 7.8, 8.4))
        c2 = c.copy() if a < 1 else c
        for k, (title, vals, best, col, fmt, sc) in enumerate([
                ("오차", NMSE_ALL, 5, ORANGE, lambda v: f"{v:.2f}", 240),
                ("되돌렸을 때 효과", GAIN, 0, GREEN, lambda v: f"{v:+.0f}", 18)]):
            x0 = 230 + k * 860
            ta = fade(t, 0.3 + 1.6 * k)
            text(c2, title, x0 + 300, 220, 44, "Bold", col, "mm", ta)
            base = 700
            line(c2, [(x0 - 60, base), (x0 + 660, base)], INK, 3, ta)
            for i, v in enumerate(vals):
                x = x0 + i * 120
                p = out_cubic(prog(t, 0.6 + 1.6 * k + 0.12 * i, 1.4 + 1.6 * k + 0.12 * i))
                hgt = v * sc * p
                fill = col if (i == best and t > 3.6) else (190, 196, 206)
                if hgt >= 0:
                    rect(c2, x - 40, base - hgt, x + 40, base, fill=fill, r=5)
                else:
                    rect(c2, x - 40, base, x + 40, base - hgt, fill=fill, r=5)
                text(c2, f"{i + 1}", x, base + 34, 30, "SemiBold", INK, "mm", ta)
            s, a2 = pop(t, 3.8 + 0.6 * k)
            bx = x0 + best * 120
            text(c2, f"{best + 1}번 층", bx, base - max(vals) * sc - 45, 44, "ExtraBold", col, "mm", a2, s)
        s, a3 = pop(t, 5.0)
        text(c2, "≠", 960, 480, 120, "ExtraBold", RED, "mm", a3, s)
        if a < 1:
            c.paste(Image.blend(BGIM, c2, a))
        return
    if t < 19.6:
        # 쌓이는 오차 탑: 회색 = 앞에서 넘어온 오차, 주황 = 그 층이 새로 만든 오차
        a = 1 - smooth(prog(t, 19.0, 19.6))
        c2 = c.copy() if a < 1 else c
        base, x0, gap, sc = 820, 380, 230, 400
        line(c2, [(x0 - 120, base), (x0 + 5 * gap + 120, base)], INK, 3)
        for i in range(6):
            x = x0 + i * gap
            t_i = 8.8 + i * 1.5
            own = NMSE_ONLY[i] * sc
            inh = max(0.0, NMSE_ALL[i] - NMSE_ONLY[i]) * sc
            text(c2, f"{i + 1}번 층", x, base + 36, 30, "SemiBold", INK, "mm", fade(t, 8.6))
            if t < t_i:
                continue
            # 회색: 이전 탑에서 날아옴
            p = smooth(prog(t, t_i, t_i + 0.8))
            if inh > 0:
                px = lerp(x - gap, x, p)
                rect(c2, px - 55, base - inh * p - 0, px + 55, base, fill=(176, 182, 192), r=4)
            q = out_back(prog(t, t_i + 0.7, t_i + 1.1))
            if prog(t, t_i + 0.7, t_i + 1.1) > 0:
                top = base - inh * p
                rect(c2, x - 55, top - own * q, x + 55, top, fill=ORANGE, r=4)
            if t > t_i + 1.1:
                text(c2, f"{NMSE_ALL[i]:.2f}", x, base - inh - own - 30, 32, "Bold", INK, "mm")
        la = fade(t, 9.0)
        rect(c2, 1340, 170, 1372, 202, fill=ORANGE, r=4, a=la)
        text(c2, "그 층이 새로 만든 오차", 1390, 186, 30, "SemiBold", INK, "lm", la)
        rect(c2, 1340, 222, 1372, 254, fill=(176, 182, 192), r=4, a=fade(t, 13.0))
        text(c2, "앞에서 넘어온 오차", 1390, 238, 30, "SemiBold", INK, "lm", fade(t, 13.0))
        if a < 1:
            c.paste(Image.blend(BGIM, c2, a))
        return
    # 도미노
    s, a = pop(t, 19.8, 0.5)
    put(c, ira("domino", 360), 960, 470, "mm", s, a)
    for i in range(6):
        pass
    a2 = fade(t, 21.0)
    text(c, "1번 층", 640, 700, 40, "Bold", BLUE, "mm", a2)
    text(c, "6번 층", 1180, 700, 40, "Bold", SUB, "mm", a2)
    arrow(c, 760, 700, 1060, 700, GREY, 5, 18, out_cubic(prog(t, 21.2, 22.0)), a2)
    s3, a3 = pop(t, 24.4)
    text(c, "앞쪽을 고치면 뒤쪽도 함께 개선", 960, 830, 50, "ExtraBold", GREEN, "mm", a3, s3)


def s_use(c, t):
    chapter(c, 9, "활용", t)
    xs = layer_row(c, t, 290, ["fp"] + ["q"] * 5, 0.3)
    # 1번 층 상자 위에 8비트 표시
    s, a = pop(t, 1.2)
    text(c, "8비트", xs[0], 180, 40, "ExtraBold", BLUE, "mm", a, s)
    text(c, "4비트", (xs[1] + xs[5]) / 2, 180, 40, "ExtraBold", SUB, "mm", a, s)
    # 크기
    a2 = fade(t, 4.4)
    text(c, "모델 크기", 300, 520, 38, "Bold", INK, "rm", a2)
    p = out_cubic(prog(t, 4.6, 5.6))
    rect(c, 340, 485, 340 + 1000 * p, 555, fill=(190, 196, 206), r=6, a=a2)
    q = out_cubic(prog(t, 5.8, 6.4))
    rect(c, 1340, 485, 1340 + 6 * q + 2, 555, fill=ORANGE, r=2, a=a2 * (q > 0))
    text(c, "+0.6%", 1380, 520, 44, "ExtraBold", ORANGE, "lm", fade(t, 6.2))
    # 정확도
    a3 = fade(t, 7.6)
    text(c, "정확도", 300, 700, 38, "Bold", INK, "rm", a3)
    v = count(t, 8.0, 10.0, ACC_W4, ACC_L1_8BIT)
    rect(c, 340, 665, 340 + 1000 * v / 100, 735, fill=GREEN, r=6, a=a3)
    text(c, f"{v:.0f}%", 340 + 1000 * v / 100 + 24, 700, 52, "ExtraBold", GREEN, "lm", a3)
    text(c, f"(전부 4비트일 때 {ACC_W4:.0f}%)", 340, 790, 32, "Medium", SUB, "lm", fade(t, 10.4))
    s4, a4 = pop(t, 10.8)
    put(c, ira("phone_happy", 280), 1660, 680, "mm", s4, a4)


def hbar_block(c, t, t0, title, rows, y0, best_col):
    text(c, title, 200, y0, 48, "ExtraBold", INK, "lm", fade(t, t0))
    for i, (lab, v) in enumerate(rows):
        y = y0 + 110 + i * 110
        a = fade(t, t0 + 0.6 + 1.6 * i)
        text(c, lab, 760, y, 36, "Bold", INK, "rm", a)
        p = out_cubic(prog(t, t0 + 0.8 + 1.6 * i, t0 + 2.0 + 1.6 * i))
        col = (190, 196, 206) if i == 0 else (ORANGE if i == 1 else GREEN)
        rect(c, 800, y - 36, 800 + 14 * v * p, y + 36, fill=col, r=6, a=a)
        text(c, f"{v * p:.0f}%" if p < 1 else f"{v:.0f}%", 800 + 14 * v * p + 20, y, 46, "ExtraBold", INK, "lm", a)


def s_big(c, t):
    chapter(c, 10, "실제 사진 모델에서 확인", t)
    if t < 9.6:
        a = 1 - smooth(prog(t, 9.0, 9.6))
        c2 = c.copy() if a < 1 else c
        for k, (nm, x, y, h) in enumerate([("dog", 460, 380, 280), ("cat", 1460, 360, 240), ("car", 520, 760, 200), ("bird", 1440, 760, 240)]):
            s, a1 = pop(t, 0.4 + 0.35 * k)
            put(c2, ira(nm, h), x, y, "mm", s, a1)
        s, a2 = pop(t, 1.0)
        text(c2, "사진 1,000가지 종류", 960, 470, 72, "ExtraBold", INK, "mm", a2, s)
        for k, nm in enumerate(["ResNet-18 (CNN)", "DeiT-Tiny (트랜스포머)"]):
            s, a3 = pop(t, 5.2 + 0.7 * k)
            lay = Image.new("RGBA", (520, 100), (0, 0, 0, 0))
            ImageDraw.Draw(lay).rounded_rectangle((3, 3, 517, 97), 12, fill=WHITE, outline=BLUE, width=4)
            tt = T(nm, 40, "Bold", BLUE)
            lay.paste(tt, ((520 - tt.width) // 2, (100 - tt.height) // 2), tt)
            put(c2, lay, 960, 590 + 120 * k, "mm", s, a3)
        if a < 1:
            c.paste(Image.blend(BGIM, c2, a))
        return
    if t < 21.4:
        a = 1 - smooth(prog(t, 20.8, 21.4))
        c2 = c.copy() if a < 1 else c
        hbar_block(c2, t, 9.8, "ResNet-18", RES, 260, GREEN)
        if a < 1:
            c.paste(Image.blend(BGIM, c2, a))
        return
    hbar_block(c, t, 21.6, "DeiT-Tiny", DEIT, 260, GREEN)
    text(c, "사진 1만 장으로 측정", 200, 820, 30, "Medium", SUB, "lm", fade(t, 22.0))


def s_sum(c, t):
    chapter(c, 11, "정리", t)
    items = [("숫자를 짧게 줄이면(양자화)\nAI 모델이 가벼워집니다.", "phone_happy"),
             ("너무 짧게 줄이면\n정답을 틀리기 시작합니다.", "batsu"),
             ("고쳐야 할 층은 오차가 큰 층이 아니라,\n되돌렸을 때 효과가 큰 층입니다.", "detective")]
    for i, (s_, icon) in enumerate(items):
        t0 = 1.0 + 4.6 * i
        y = 300 + i * 230
        s, a = pop(t, t0)
        lay = Image.new("RGBA", (90, 90), (0, 0, 0, 0))
        ImageDraw.Draw(lay).rounded_rectangle((2, 2, 88, 88), 10, fill=BLUE)
        n = T(str(i + 1), 54, "ExtraBold", WHITE)
        lay.paste(n, ((90 - n.width) // 2, (90 - n.height) // 2 - 4), n)
        put(c, lay, 230, y, "mm", s, a)
        p = out_cubic(prog(t, t0 + 0.1, t0 + 0.6))
        text(c, s_, 320 - 40 * (1 - p), y, 54, "Bold", INK if i < 2 else BLUE, "lm", p)
        s2, a2 = pop(t, t0 + 0.5)
        put(c, ira(icon, 210), 1640, y, "mm", s2, a2)


def s_end(c, t):
    a = fade(t, 0.2)
    text(c, "학습에는 정밀한 숫자가 필요합니다.", 960, 330, 64, "Bold", INK, "mm", a)
    text(c, "학습이 끝난 뒤의 추론에도 같은 정밀도가 필요할까요?", 960, 430, 64, "ExtraBold", BLUE, "mm", fade(t, 1.4))
    credit = ("컴퓨터비전 기말 프로젝트  김진서 · 이하형 · 정채호준\n"
              "데이터: 3D Shapes (DeepMind), ImageNet    모델: ResNet-18 (torchvision), DeiT-Tiny (timm)\n"
              "그림: いらすとや (irasutoya)    효과음: soundeffect-lab.info\n"
              "배경음: \"Wallpaper\" Kevin MacLeod (incompetech.com), CC BY 4.0")
    text(c, credit, 960, 780, 28, "Medium", SUB, "mm", fade(t, 3.0), align="center")


# ───────────────────────── 장면 목록 ─────────────────────────
# (그리기, 길이, 자막 [(시작, 끝, 문장)], 효과음 [(시각, 파일, dB)])
POP, TICK, OK, NG, WHOOSH, REVEAL, CARD, BIG, SWITCH = ("pa1", "s_틱1", "kira1", "despair1", "hyun1", "jajean1", "s_카드1", "s_큰숫자", "cursor8")
# 원래 장면 시간(초) 기준으로 적고, 속도 배율로 줄인다 (v2: 합니다체, 약 3분 이내).
# (그리기, 원래 시작, 원래 끝, 속도, 자막 [(원래 시작, 원래 끝, 문장)], 효과음 [(원래 시각, 파일, dB)])
PLAN = [
    (s_title, 0, 5.0, 1.3, [], [(0.3, POP, -8), (1.2, WHOOSH, -12)]),
    (s_quiz, 0, 13.6, 1.7, [(0.3, 9.2, "이 그림 속 물체의 모양, 색, 크기를 맞히는 문제입니다."),
                            (9.4, 13.6, "문제 그림은 모두 48만 장입니다.")],
     [(0.3, POP, -8), (4.6, CARD, -6), (5.4, CARD, -6), (6.2, CARD, -6), (9.4, WHOOSH, -10)]),
    (s_learn, 0, 13.0, 1.8, [(0.3, 6.2, "그림 6만 장과 정답으로 AI 모델을 학습시켰습니다."),
                             (6.3, 13.0, "처음 보는 그림 1만 장에서도 정확도 100%입니다.")],
     [(0.2, POP, -8), (10.0, BIG, -4), (10.0, OK, -8), (10.5, POP, -10)]),
    (s_conv, 5.4, 22.0, 1.7, [(5.4, 11.2, "AI 모델은 여러 층(layer)으로 이루어져 있고, 층마다 숫자 판(필터)이 있습니다."),
                              (11.2, 16.4, "필터는 그림 위를 한 칸씩 이동하며 숫자를 곱하고 더해 새 그림을 만듭니다."),
                              (16.4, 22.0, "이 필터는 밝기가 바뀌는 곳, 즉 물체의 테두리를 찾아냅니다.")],
     [(6.0, CARD, -6)]),
    (s_layers, 0, 21.0, 1.7, [(0.3, 10.0, "그림은 6개 층을 차례로 지나며, 층마다 새 그림이 만들어집니다."),
                              (10.0, 16.2, "앞쪽 층은 테두리 같은 단순한 모양을, 뒤쪽 층은 간단하게 정리된 정보를 남깁니다."),
                              (16.2, 21.0, "마지막 층 다음에 정답을 고릅니다. 빨간 원기둥, 크기 6.")],
     [(1.5 + 1.3 * i, TICK, -8) for i in range(6)] + [(16.4, OK, -6)]),
    (s_long, 0, 17.0, 1.8, [(0.3, 4.0, "필터 속 숫자는 실제로 이렇게 깁니다."),
                            (4.0, 8.2, "숫자 하나를 저장하는 데 32칸(32비트)이 필요합니다."),
                            (8.4, 17.0, "큰 모델에는 이런 숫자가 1,000만 개 이상이라, 스마트폰에서는 무겁고 느립니다.")],
     [(6.4, CARD, -6), (8.6, WHOOSH, -10), (12.4, POP, -8), (13.2, NG, -12)]),
    (s_quant, 0, 28.0, 1.8, [(0.3, 7.0, "그래서 숫자를 짧게 줄입니다. 이것이 양자화(quantization)입니다."),
                             (7.4, 12.6, "값은 정해진 눈금 중 가장 가까운 눈금으로 옮겨집니다."),
                             (12.6, 19.4, "눈금이 적을수록 원래 값과의 차이, 즉 오차가 커집니다."),
                             (19.6, 23.8, "4비트로 줄이면 모델 크기는 8분의 1이 됩니다."),
                             (23.8, 28.0, "대신 오차가 생깁니다. 정답은 그대로일까요?")],
     [(2.0, WHOOSH, -10), (3.6, TICK, -6), (4.2, REVEAL, -8), (8.8, TICK, -10), (11.4, TICK, -10), (13.8, CARD, -6), (24.0, POP, -8)]),
    (s_result, 0, 20.0, 1.7, [(0.3, 8.2, "같은 그림을 넣으면, 8비트 모델은 빨간 원기둥으로 맞힙니다."),
                              (8.2, 13.2, "4비트 모델은 노란 캡슐로 틀립니다."),
                              (13.2, 20.0, "1만 장 전체의 정확도는 100%에서 58%로 떨어집니다.")],
     [(1.3, OK, -10), (5.1, OK, -10), (9.1, NG, -6), (13.8, BIG, -6)]),
    (s_method1, 0, 21.0, 1.7, [(0.3, 11.2, "어느 층 때문일까요? 방법 1: 층마다 원래 모델과 4비트 모델의 차이(오차)를 잽니다."),
                               (11.2, 15.6, "오차는 6번 층이 가장 큽니다."),
                               (15.6, 21.0, "그럼 6번 층이 원인일까요?")],
     [(0.6, POP, -12), (11.2, CARD, -6), (15.8, REVEAL, -8)]),
    (s_method2, 0, 33.0, 1.8, [(0.3, 6.2, "방법 2: 나머지는 4비트로 두고, 한 층만 원래 숫자로 되돌립니다."),
                               (6.2, 14.0, "6번 층을 되돌려도 여전히 오답이고, 정확도도 그대로입니다."),
                               (14.2, 22.0, "1번 층을 되돌리면 정답이 돌아오고, 정확도가 19%p 오릅니다."),
                               (22.2, 33.0, "6개 층 모두 보면, 1번 층의 효과가 가장 크고 6번 층은 효과가 거의 없습니다.")],
     [(6.4, SWITCH, -4), (8.6, NG, -12), (14.4, SWITCH, -4), (15.2, OK, -6), (16.4, BIG, -6), (22.4, WHOOSH, -10), (26.6, POP, -8)]),
    (s_why, 0, 29.0, 1.8, [(0.3, 8.2, "오차가 가장 큰 층(6번)과 효과가 가장 큰 층(1번)이 서로 다릅니다."),
                           (8.6, 19.4, "층마다 새로 생기는 오차(주황)는 작지만, 앞에서 넘어온 오차(회색)가 뒤로 갈수록 커집니다."),
                           (19.8, 29.0, "6번 층의 큰 오차는 대부분 앞에서 넘어온 것으로 볼 수 있어, 앞쪽 층을 고칠 때 효과가 큽니다.")],
     [(3.8, CARD, -6), (5.0, REVEAL, -8)] + [(9.6 + 1.5 * i, TICK, -8) for i in range(6)] + [(19.8, POP, -8), (24.4, OK, -8)]),
    (s_use, 0, 13.0, 1.7, [(0.3, 4.4, "1번 층만 8비트로 두면,"),
                           (4.4, 13.0, "모델 크기는 0.6%만 늘고, 정확도는 58%에서 74%로 오릅니다.")],
     [(1.2, POP, -8), (6.2, TICK, -6), (8.0, BIG, -6), (10.8, OK, -8)]),
    (s_big, 0, 31.0, 1.8, [(0.3, 9.6, "사진 1,000종류를 분류하는 큰 모델(ResNet-18, DeiT-Tiny)에서도 확인했습니다."),
                           (9.8, 21.2, "ResNet-18 4비트: 오차가 큰 층 3개를 8비트로 올리면 19%, 효과가 큰 층 3개면 55%입니다."),
                           (21.6, 31.0, "DeiT-Tiny 5비트에서도 25% 대 33%로, 효과가 큰 층을 고른 쪽이 높습니다.")],
     [(0.4, POP, -10), (0.75, POP, -10), (1.1, POP, -10), (1.45, POP, -10), (5.2, CARD, -6), (5.9, CARD, -6),
      (12.0, TICK, -8), (13.6, TICK, -8), (15.2, BIG, -6), (23.8, TICK, -8), (25.4, TICK, -8), (27.0, BIG, -6)]),
    (s_sum, 0, 16.0, 1.5, [], [(1.0, CARD, -6), (5.6, CARD, -6), (10.2, CARD, -6), (10.7, OK, -8)]),
    (s_end, 0, 8.0, 1.3, [], [(0.2, WHOOSH, -12)]),
]


def _wrap(fn, off, spd):
    return lambda c, t: fn(c, off + t * spd)


SCENES = [(_wrap(fn, a, spd), (b - a) / spd,
           [((x - a) / spd, (y - a) / spd, s) for (x, y, s) in caps],
           [((x - a) / spd, f, db) for (x, f, db) in sfx if a <= x < b])
          for (fn, a, b, spd, caps, sfx) in PLAN]


def timeline():
    starts, t = [], 0.0
    for sc in SCENES:
        starts.append(t)
        t += sc[1]
    return starts, t


STARTS, TOTAL = timeline()


def caption(c, s, a):
    if not s or a <= 0:
        return
    f = font("SemiBold", 44)
    # 너무 길면 두 줄로 (공백 기준 가운데 근처에서)
    if f.getlength(s) > 1500:
        mid = len(s) // 2
        cut = min((i for i, ch in enumerate(s) if ch == " "), key=lambda i: abs(i - mid))
        s = s[:cut] + "\n" + s[cut + 1:]
    im = T(s, 44, "SemiBold", WHITE, "center", 1.3)
    pw, ph = im.width + 64, im.height + 26
    y1 = H - 46
    rect(c, W / 2 - pw / 2, y1 - ph, W / 2 + pw / 2, y1, fill=(20, 22, 26), r=8, a=0.86 * a)
    put(c, im, W / 2, y1 - ph / 2, "mm", 1, a)


def frame(tg):
    c = BGIM.copy()
    for i, (fn, dur, caps, _) in enumerate(SCENES):
        st = STARTS[i]
        if st <= tg < st + dur:
            t = tg - st
            fn(c, t)
            # 장면 전환: 앞 0.25초 페이드 인
            if t < 0.25 and i > 0:
                c = Image.blend(BGIM, c, smooth(t / 0.25))
            for (a0, a1, s) in caps:
                if a0 <= t < a1:
                    caption(c, s, fade(t, a0, a1, 0.15))
            break
    return c


# ───────────────────────── 소리 ─────────────────────────
SR = 44100


def decode(path):
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2)


def build_audio(path):
    n = int((TOTAL + 1) * SR)
    mixb = np.zeros((n, 2), np.float32)
    bgm = decode(BGM)
    reps = int(np.ceil(n / len(bgm)))
    bg = np.tile(bgm, (reps, 1))[:n]
    env = np.ones(n, np.float32)
    fi, fo = int(1.5 * SR), int(3.0 * SR)
    env[:fi] = np.linspace(0, 1, fi)
    end = int(TOTAL * SR)
    env[end - fo:end] = np.linspace(1, 0, fo)
    env[end:] = 0
    rms = np.sqrt((bgm ** 2).mean()) + 1e-9
    mixb += bg * env[:, None] * (0.045 / rms)   # 배경음은 작게 (자막을 읽는 데 방해되지 않게)
    cache = {}
    for i, sc in enumerate(SCENES):
        for (t, name, db) in sc[3]:
            f = os.path.join(SFX_DIR, name + ".mp3")
            if name not in cache:
                cache[name] = decode(f)
            s = cache[name] * (10 ** (db / 20)) * 0.6
            p = int((STARTS[i] + t) * SR)
            e = min(n, p + len(s))
            mixb[p:e] += s[:e - p]
    peak = np.abs(mixb).max()
    if peak > 0.95:
        mixb *= 0.95 / peak
    mixb.astype(np.float32).tofile(path)


# ───────────────────────── 실행 ─────────────────────────
def save_script():
    with open(os.path.join(HERE, "대본.txt"), "w") as f:
        for i, sc in enumerate(SCENES):
            for (a0, a1, s) in sc[2]:
                if s:
                    t = STARTS[i] + a0
                    f.write(f"{int(t // 60)}:{t % 60:04.1f}  {s}\n")


def main():
    args = sys.argv[1:]
    save_script()
    if args and args[0] == "--frames":
        os.makedirs(os.path.join(HERE, "확인"), exist_ok=True)
        for s in args[1].split(","):
            frame(float(s)).save(os.path.join(HERE, "확인", f"f_{float(s):06.1f}.png"))
        return
    if args and args[0] == "--sheet":
        os.makedirs(os.path.join(HERE, "확인"), exist_ok=True)
        for i, sc in enumerate(SCENES):
            ts = [STARTS[i] + sc[1] * q for q in (0.3, 0.55, 0.8, 0.98)]
            ims = [frame(t).resize((640, 360)) for t in ts]
            sheet = Image.new("RGB", (1280, 720))
            for k, im in enumerate(ims):
                sheet.paste(im, ((k % 2) * 640, (k // 2) * 360))
            sheet.save(os.path.join(HERE, "확인", f"sheet_{i:02d}.jpg"), quality=88)
        return
    wav = os.path.join(HERE, "확인_audio.f32")
    build_audio(wav)
    nf = int(TOTAL * FPS)
    cmd = [FFMPEG, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", wav,
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-shortest", "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(nf):
        p.stdin.write(frame(k / FPS).tobytes())
        if k % 300 == 0:
            print(f"{k / FPS:6.1f}s / {TOTAL:.1f}s", flush=True)
    p.stdin.close()
    p.wait()
    os.remove(wav)
    print("완성:", OUT, f"{TOTAL:.1f}초")


if __name__ == "__main__":
    main()
