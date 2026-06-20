"""
=============================================================================
테마 — 코이핀(Koyfin)형 모던데스크 (기본 다크) + 라이트 토글
=============================================================================
디자인 무드:
  - 다크(기본): 딥 슬레이트-네이비 배경(#0e1320) + 카드(#161d2b),
    차분한 블루 강조(#3b9eff), 보조지표는 muted·핵심 숫자는 순백으로 대비 최대.
  - 라이트: 밝은 회색 배경 + 흰 카드 (토스풍, 보조 옵션).
  - 한국 시장 관례: 상승=빨강(#f6465d), 하락=파랑(#4d8dfc) 고정.
  - 강조색은 사이드바에서 큐레이션 프리셋으로 변경 (기본: 블루 #3b9eff).

런타임 테마 전환 구조:
  COLORS / BASE_LAYOUT / 컬러스케일을 모듈 전역 객체로 두고,
  set_theme() 이 내용물을 '제자리에서(in-place)' 갈아끼운다.
  → 차트 함수들이 임포트해 둔 참조가 그대로 유효 (재임포트 불필요).

사용 예 (dashboard.py 사이드바):
    from app.ui.theme import set_theme, inject_terminal_css
    set_theme(dark=st.toggle("🌙 다크 모드"), accent=st.color_picker("메인 색상"))
    inject_terminal_css()     # set_theme() '이후' 호출 — 매 rerun 마다

⚠️ .streamlit/config.toml 의 네이티브 테마는 서버 시작 시에만 읽히므로
   위젯 내부 색까지 완전히 바꾸려면 서버 재시작이 필요할 수 있다.
=============================================================================
"""

from __future__ import annotations

import streamlit as st

DEFAULT_ACCENT = "#3b9eff"   # 기본 메인 색 — 코이핀형 차분한 블루

# ─────────────────────────────────────────────────────────────────────────────
# 팔레트 — 라이트/다크. up/down 은 한국 시장 관례(상승=빨강, 하락=파랑) 고정.
# ─────────────────────────────────────────────────────────────────────────────

# 강조색은 '상승=빨강·하락=파랑'과 충돌하지 않는 색만 큐레이션 (자유 RGB 피커 대신).
# 기본은 앰버 — 블룸버그/기관 터미널의 전통 강조색.
ACCENT_PRESETS: dict[str, str] = {
    "블루":    "#3b9eff",   # 기본 (코이핀형)
    "바이올렛": "#8b7cf6",
    "민트":    "#00c896",
    "앰버":    "#f5a623",
    "시안":    "#22d3ee",
}

_LIGHT: dict[str, str] = {
    "bg":     "#f2f4f6",   # 앱 배경 (토스 gray100)
    "panel":  "#ffffff",   # 카드/패널
    "up":     "#f04452",   # 상승/매수 — 토스 레드
    "down":   "#3182f6",   # 하락/매도 — 토스 블루
    "accent": DEFAULT_ACCENT,
    "warn":   "#fe9800",   # 중립/주의 — 오렌지
    "blue":   "#3182f6",
    "purple": "#7048e8",
    "text":   "#191f28",   # 본문 (토스 gray900)
    "value":  "#0b1220",   # 핵심 숫자(메트릭) — 더 진하게
    "muted":  "#8b95a1",   # 보조 (토스 gray500)
    "grid":   "#e5e8eb",   # 그리드 (토스 gray200)
    "border": "#e5e8eb",
    "elev":   "#eef1f5",   # 칩 등 살짝 띄운 표면
    "shadow": "rgba(25, 31, 40, 0.06)",
}

# 코이핀형 — 딥 슬레이트-네이비. 차분한 블루 강조, 보조는 muted, 숫자는 순백.
_DARK: dict[str, str] = {
    "bg":     "#0e1320",   # 딥 슬레이트-네이비 배경
    "panel":  "#161d2b",   # 카드/패널 (배경보다 한 단계 밝게)
    "up":     "#f6465d",   # 상승 — 선명한 레드(한국 관례)
    "down":   "#4d8dfc",   # 하락 — 블루(한국 관례)
    "accent": DEFAULT_ACCENT,
    "warn":   "#f0a93b",
    "blue":   "#4d8dfc",
    "purple": "#9d8cff",
    "text":   "#c3ccdb",   # 본문 (슬레이트 라이트)
    "value":  "#f5f8ff",   # 핵심 숫자 — 거의 순백으로 대비 최대
    "muted":  "#6b7689",   # 보조 (슬레이트 muted)
    "grid":   "#1b2436",   # 그리드 (은은하게)
    "border": "#232c3d",   # 보더 (얇고 깔끔하게)
    "elev":   "#1e2740",   # 칩 등 살짝 띄운 표면
    "shadow": "rgba(0, 0, 0, 0.35)",
}

# 모듈 전역 — set_theme() 이 in-place 로 갱신 (참조 유지가 핵심)
COLORS: dict[str, str] = dict(_LIGHT)

PALETTE: list[str] = [
    COLORS["accent"], "#f04452", "#fe9800", "#7048e8", "#3182f6", "#f06595",
]

DIVERGING_SCALE: list = [[0.0, COLORS["down"]], [0.5, "#ffffff"], [1.0, COLORS["up"]]]
HEAT_SCALE: list = [[0.0, "#74c0fc"], [0.45, COLORS["warn"]], [1.0, COLORS["up"]]]
SCORE_SCALE: list = [[0.0, COLORS["down"]], [0.5, "#e5e8eb"], [1.0, COLORS["up"]]]

BASE_LAYOUT: dict = {}   # set_theme() 이 채움

_FONT = "'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"


def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def set_theme(dark: bool = False, accent: str | None = None) -> None:
    """
    테마 적용 — 전역 COLORS/BASE_LAYOUT/컬러스케일을 in-place 갱신.
    차트를 그리기 '전에' (사이드바 최상단에서) 호출해야 한다.
    """
    palette = dict(_DARK if dark else _LIGHT)
    if accent:
        palette["accent"] = accent
    COLORS.clear()
    COLORS.update(palette)

    PALETTE[0] = COLORS["accent"]    # 첫 번째 시리즈 색 = 메인 색

    mid = COLORS["panel"] if dark else "#ffffff"
    DIVERGING_SCALE[:] = [[0.0, COLORS["down"]], [0.5, mid], [1.0, COLORS["up"]]]
    HEAT_SCALE[:] = [[0.0, "#74c0fc"], [0.45, COLORS["warn"]], [1.0, COLORS["up"]]]
    SCORE_SCALE[:] = [[0.0, COLORS["down"]], [0.5, COLORS["grid"]], [1.0, COLORS["up"]]]

    BASE_LAYOUT.clear()
    BASE_LAYOUT.update(
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"], family="Pretendard, sans-serif"),
        hoverlabel=dict(
            bgcolor=COLORS["panel"],
            bordercolor=COLORS["border"],
            font=dict(color=COLORS["text"], family="Pretendard, sans-serif", size=12),
        ),
    )


# 모듈 로드 시 라이트 테마로 초기화 (set_theme 미호출 환경 대비)
set_theme(dark=False)


def terminal_title(text: str, size: int = 16) -> dict:
    """차트 제목 공통 스타일 (진한 텍스트, 좌측 정렬)."""
    return dict(text=text, font=dict(size=size, color=COLORS["text"]),
                x=0.0, xanchor="left")


def scene_axis(title: str, **extra) -> dict:
    """3D scene 축 공통 스타일 — 그리드 정리 + 테마 톤 통일."""
    return dict(
        title=title,
        backgroundcolor=COLORS["panel"],
        gridcolor=COLORS["grid"],
        zerolinecolor=COLORS["grid"],
        showspikes=False,
        **extra,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit CSS — 현재 COLORS 기준으로 매번 생성 (테마 전환 대응)
# ─────────────────────────────────────────────────────────────────────────────

def _build_css() -> str:
    c = COLORS
    accent = c["accent"]
    accent_soft = hex_to_rgba(accent, 0.10)
    return f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');

/* ── 전역 — 폰트·배경·숫자 정렬(금융 UI 핵심: tabular-nums) ─────────────── */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {c["bg"]} !important;
    color: {c["text"]};
    font-family: {_FONT};
    -webkit-font-smoothing: antialiased;
    font-feature-settings: "tnum" 1, "cv01" 1;   /* 등폭 숫자 */
}}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
/* 데이터 고밀도 — 루트 폰트 살짝 축소로 전체 UI 압축 (코이핀 느낌) */
html {{ font-size: 15px; }}
.block-container {{
    padding-top: 0.8rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}}
[data-testid="stVerticalBlock"] {{ gap: 0.55rem; }}
h1, h2, h3, h4, p, span, label, li, div {{ font-family: {_FONT}; }}
h1, h2, h3, h4, [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{ color: {c["text"]}; }}
h1 {{ font-weight: 800; letter-spacing: -0.03em; font-size: 1.55rem !important; }}
h2 {{ font-weight: 700; letter-spacing: -0.02em; font-size: 1.2rem !important; }}
/* 섹션 헤더 — 좌측 액센트 바 + 절제된 크기 (기관 리서치 느낌) */
h3 {{
    font-weight: 700; letter-spacing: -0.01em; font-size: 1.0rem !important;
    border-left: 3px solid {accent}; padding-left: 0.55rem; margin-top: 0.4rem;
}}
a {{ color: {accent}; text-decoration: none; font-weight: 600; }}
a:hover {{ text-decoration: underline; }}
hr {{ margin: 0.8rem 0; border-color: {c["border"]}; opacity: 0.6; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
    color: {c["muted"]} !important; font-size: 0.8rem;
}}

/* ── 상단 앱 바 (코이핀형) ───────────────────────────────────────────────── */
.app-header {{
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 0.5rem;
    padding: 0.55rem 0.9rem; margin: -0.4rem 0 1rem 0;
    background: {c["panel"]};
    border: 1px solid {c["border"]};
    border-radius: 8px;
    border-left: 3px solid {accent};
}}
.app-header .brand {{
    font-size: 1.15rem; font-weight: 800; letter-spacing: 0.02em; color: {c["text"]};
}}
.app-header .brand .accent {{ color: {accent}; }}
.app-header .brand .sub {{
    font-size: 0.72rem; font-weight: 600; color: {c["muted"]};
    margin-left: 0.55rem; letter-spacing: 0.02em;
}}
.app-header .meta {{
    font-size: 0.78rem; color: {c["muted"]}; font-feature-settings: "tnum" 1;
    display: flex; align-items: center; gap: 0.5rem;
}}
.app-header .pill {{
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.18rem 0.55rem; border-radius: 999px;
    background: {c["elev"]}; border: 1px solid {c["border"]};
    font-weight: 700; color: {c["text"]}; font-size: 0.74rem;
}}
.app-header .dot {{
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
}}
.app-header .clock {{ font-feature-settings: "tnum" 1; color: {c["muted"]}; }}

/* ── 티커 타일 (코이핀형 컴팩트 시세 카드) ──────────────────────────────── */
.tile {{
    background: {c["panel"]};
    border: 1px solid {c["border"]};
    border-radius: 8px;
    padding: 0.65rem 0.85rem;
    transition: border-color 0.12s ease;
}}
.tile:hover {{ border-color: {hex_to_rgba(accent, 0.6)}; }}
.tile .sym {{
    font-size: 0.82rem; font-weight: 800; color: {c["text"]}; letter-spacing: 0.02em;
}}
.tile .nm {{
    font-size: 0.72rem; font-weight: 500; color: {c["muted"]}; margin-left: 0.35rem;
}}
.tile .px {{
    font-size: 1.5rem; font-weight: 800; color: {c["value"]};
    font-variant-numeric: tabular-nums; letter-spacing: -0.02em;
    margin: 0.1rem 0 0.05rem 0; line-height: 1.15;
}}
.tile .chg {{ font-size: 0.84rem; font-weight: 700; font-variant-numeric: tabular-nums; }}
.tile .chg.up {{ color: {c["up"]}; }}
.tile .chg.down {{ color: {c["down"]}; }}
.tile .chg.flat {{ color: {c["muted"]}; }}
.tile.err .px {{ font-size: 1rem; color: {c["muted"]}; }}

/* ── 사이드바 ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: {c["panel"]};
    border-right: 1px solid {c["border"]};
}}
[data-testid="stSidebar"] * {{ color: {c["text"]}; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
    text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem;
    font-weight: 700; color: {c["muted"]} !important;
}}

/* ── 입력 위젯 ─────────────────────────────────────────────────────────── */
.stTextInput input, .stNumberInput input, .stTextArea textarea {{
    background-color: {c["bg"]};
    color: {c["text"]}; border-color: {c["border"]}; border-radius: 8px;
}}
[data-baseweb="select"] > div {{
    background-color: {c["bg"]}; border-color: {c["border"]};
    color: {c["text"]}; border-radius: 8px;
}}

/* ── 강조색 일관 적용 — 기본 위젯도 강조색을 따르되, 가독성 우선 ─────────── */
/* 멀티셀렉트 칩: 강조색 배경 대신 '중립 표면 + 일반 글자 + 강조 테두리' (어떤 색이든 가독) */
span[data-baseweb="tag"] {{
    background-color: {c["elev"]} !important;
    color: {c["text"]} !important;
    border: 1px solid {hex_to_rgba(accent, 0.55)} !important;
    border-radius: 7px; font-weight: 600;
}}
span[data-baseweb="tag"] span {{ color: {c["text"]} !important; }}
span[data-baseweb="tag"] svg {{ fill: {c["muted"]} !important; }}
/* 토글(스위치) 켜짐 */
[data-testid="stCheckbox"] [aria-checked="true"],
label[data-baseweb="checkbox"] [aria-checked="true"] > div:first-child {{
    background-color: {accent} !important;
}}
[data-baseweb="checkbox"] [data-checked="true"] {{ background-color: {accent} !important; }}
/* 라디오 선택 점 */
[data-baseweb="radio"] [aria-checked="true"] div:first-child {{
    background-color: {accent} !important; border-color: {accent} !important;
}}
/* 슬라이더 트랙·썸·값 */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
    background-color: {accent} !important;
}}
[data-testid="stSlider"] [data-testid="stThumbValue"] {{ color: {accent} !important; }}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div:nth-child(2) {{
    background: {accent} !important;
}}
/* 진행바·스피너 */
[data-testid="stProgress"] [role="progressbar"] > div {{ background-color: {accent} !important; }}

/* ── 메트릭 카드 — 절제된 12px 라운드 + 상단 액센트 + 등폭 숫자 ────────── */
[data-testid="stMetric"] {{
    background-color: {c["panel"]};
    border: 1px solid {c["border"]};
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
}}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{
    color: {c["muted"]} !important; font-weight: 600; font-size: 0.78rem;
    letter-spacing: 0.01em;
}}
[data-testid="stMetricValue"] {{
    color: {c["value"]}; font-weight: 800; font-size: 1.5rem;
    letter-spacing: -0.02em; font-feature-settings: "tnum" 1;
    font-variant-numeric: tabular-nums;
}}
[data-testid="stMetricDelta"] {{ font-weight: 700; font-variant-numeric: tabular-nums; }}
/* 한국 시장 관례: 상승 = 빨강, 하락 = 파랑 (토스증권과 동일) */
[data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-Up"]) {{ color: {c["up"]} !important; }}
[data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Up"] {{ fill: {c["up"]} !important; color: {c["up"]} !important; }}
[data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-Down"]) {{ color: {c["down"]} !important; }}
[data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Down"] {{ fill: {c["down"]} !important; color: {c["down"]} !important; }}

/* ── 탭 ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {c["border"]}; }}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent; color: {c["muted"]}; font-weight: 600;
    border-radius: 8px 8px 0 0; padding: 0.4rem 0.95rem; font-size: 0.88rem;
}}
.stTabs [aria-selected="true"] {{
    color: {accent} !important; background-color: {accent_soft} !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {accent}; height: 2px; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* ── 버튼 ───────────────────────────────────────────────────────────────── */
.stButton > button {{
    font-family: {_FONT}; font-weight: 700; border-radius: 8px;
    border: 1px solid {c["border"]}; background-color: {c["panel"]};
    color: {c["text"]}; transition: all 0.12s ease;
}}
.stButton > button:hover {{ border-color: {accent}; color: {accent}; }}
.stButton > button[kind="primary"] {{
    background-color: {accent}; color: #ffffff; border: none;
    box-shadow: 0 2px 8px {hex_to_rgba(accent, 0.35)};
}}
.stButton > button[kind="primary"]:hover {{
    filter: brightness(1.06); color: #ffffff; transform: translateY(-1px);
}}

/* ── 데이터프레임·익스팬더·컨테이너 ───────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {c["border"]}; border-radius: 8px;
    background-color: {c["panel"]};
    font-variant-numeric: tabular-nums;
}}
[data-testid="stExpander"] {{
    border: 1px solid {c["border"]}; border-radius: 8px;
    background-color: {c["panel"]}; box-shadow: none;
}}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {{ color: {c["text"]}; }}
[data-testid="stVerticalBlockBorderWrapper"] > div {{
    border-color: {c["border"]} !important; border-radius: 8px !important;
    background-color: {c["panel"]};
}}
[data-testid="stPlotlyChart"] {{
    background-color: {c["panel"]};
    border: 1px solid {c["border"]}; border-radius: 8px;
    box-shadow: none; overflow: hidden;
}}

/* ── 알림 박스 ─────────────────────────────────────────────────────────── */
.stAlert {{ border-radius: 8px; font-size: 0.88rem; }}

/* ── 스크롤바 (디테일 — 기본 굵은 회색 대신 절제된 톤) ─────────────────── */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {c["border"]}; border-radius: 6px; }}
::-webkit-scrollbar-thumb:hover {{ background: {c["muted"]}; }}

/* ── 모바일 반응형 — 좁은 화면(≤640px)에서 컬럼 세로 스택 ─────────────── */
@media (max-width: 640px) {{
    [data-testid="stHorizontalBlock"] {{ flex-direction: column !important; gap: 0.4rem !important; }}
    [data-testid="stColumn"], [data-testid="column"] {{
        width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important;
    }}
    .block-container {{ padding-left: 0.7rem !important; padding-right: 0.7rem !important; }}
    .app-header {{ flex-direction: column; align-items: flex-start; }}
    [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {{ overflow-x: auto !important; }}
}}
</style>
"""


def render_header(brand_main: str, brand_accent: str, subtitle: str = "",
                  clocks: list[str] | None = None,
                  status_color: str | None = None, status_text: str = "") -> None:
    """
    상단 앱 바 렌더 — 코이핀형 (브랜드 워드마크 + 상태 pill + 시계).

    brand_main + brand_accent: 워드마크(앞 일반색 + 강조색).
    subtitle: 워드마크 옆 작은 설명.
    clocks: 우측 시계 문자열들(KST/ET 등).
    status_color/status_text: 상태 pill(예: 시장 개장=초록 점 + 'US OPEN').
    """
    sub = f'<span class="sub">{subtitle}</span>' if subtitle else ""
    pill = ""
    if status_color:
        pill = (f'<span class="pill"><span class="dot" style="background:{status_color}">'
                f'</span>{status_text}</span>')
    clock_html = ""
    if clocks:
        clock_html = '<span class="clock">' + "&nbsp;·&nbsp;".join(clocks) + "</span>"
    st.markdown(
        f'<div class="app-header">'
        f'<div class="brand">{brand_main}<span class="accent">{brand_accent}</span>{sub}</div>'
        f'<div class="meta">{pill}{clock_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def ticker_tile(name: str, sym: str, price: float | None,
                change: float | None, pct: float | None) -> str:
    """코이핀형 컴팩트 시세 타일 HTML 반환 (col.markdown 으로 렌더)."""
    if price is None or price != price:        # None 또는 NaN
        return (f'<div class="tile err"><div class="sym">{sym}'
                f'<span class="nm">{name}</span></div>'
                f'<div class="px">데이터 없음</div></div>')
    cls = "up" if (change or 0) > 0 else "down" if (change or 0) < 0 else "flat"
    arrow = "▲" if cls == "up" else "▼" if cls == "down" else "·"
    chg = (f'{arrow} {change:+,.2f} ({pct:+.2f}%)'
           if change is not None and pct is not None else "—")
    return (f'<div class="tile"><div class="sym">{sym}'
            f'<span class="nm">{name}</span></div>'
            f'<div class="px">${price:,.2f}</div>'
            f'<div class="chg {cls}">{chg}</div></div>')


def inject_terminal_css() -> None:
    """현재 테마 기준 CSS 주입. set_theme() 호출 '이후', 매 rerun 마다 호출."""
    st.markdown(_build_css(), unsafe_allow_html=True)
