# Investment Analysis System Blueprint

> 본 프로젝트는 3개의 투자 분석 시스템을 포함하는 통합 블루프린트입니다.
> 각 시스템은 LLM 에이전트가 순차적으로 구현할 수 있도록 단계별로 설계되어 있습니다.

---

## 프로젝트 개요

| 프로젝트 | 디렉토리 | 설명 | 현재 상태 |
|----------|----------|------|-----------|
| **배당 최적화** | `implementations/DividendOptimizer` | 미국 배당주 포트폴리오 최적화 | API/프론트/백테스트/실시간 업데이트 구현, 운영 자동화/성과추적 보강 필요 |
| **미국 주식 분석** | `implementations/USStockAnalysis` | 스마트머니 기반 미국 주식 스크리닝 | 파이프라인/Flask/대시보드/검증 리포트 구현, 스케줄링/배포 설정 보강 필요 |
| **국내 주식 분석** | `implementations/StockAI` | 파동 분석 기반 국내 주식 스크리닝 | 파이프라인/Streamlit+Flask/성과추적/알림 구현, 데이터 산출물 생성 및 스케줄링 보강 필요 |

---

## 현재 구현 상황 상세

### 1. DividendOptimizer (배당 최적화 시스템)

**목적**: 미국 배당주/ETF를 활용한 월배당 포트폴리오 최적화

**구현 완료 (코드 기준)**:
| 파일 | 상태 | 설명 |
|------|------|------|
| `flask_app.py` | 완료 | Flask 웹 서버 (배당, 리스크, 백테스트, 실시간 가격, 차트 API) |
| `engine.py` | 완료 | 포트폴리오 생성 엔진 |
| `loader.py` | 완료 | 배당 데이터 로더 (yfinance) |
| `portfolio_optimizer.py` | 완료 | 최적화 알고리즘 (Greedy/Risk Parity) |
| `risk_analytics.py` | 완료 | 리스크 지표 (Volatility, Sharpe, Drawdown) |
| `dividend_analyzer.py` | 완료 | 배당 지속성 분석 |
| `backtest.py` | 완료 | 백테스트 엔진 |
| `config/*.json` | 완료 | 10개 테마, 28개 태그 설정 |
| `data/universe_seed.json` | 완료 | 214개 티커 (88 ETF + 126 개별주) |
| `templates/*.html` | 완료 | 랜딩/대시보드/배당 UI + JavaScript 로직 |
| `static/css/main.css` | 완료 | 스타일시트 |

**추가 보강 필요**:
- 데이터 수집/캐시 자동화 (가격/배당/FX 주기적 갱신)
- 포트폴리오 실행 이력 및 성과 추적 저장소 (SQLite 등)
- 실시간 가격 업데이트 최적화 (캐시/레이트리밋, WebSocket은 선택)

---

### 2. USStockAnalysis (미국 주식 분석 시스템)

**목적**: S&P 500 종목의 스마트머니/수급 기반 종목 선별

**구현 완료 (코드 기준)**:
| 파일 | 상태 | 설명 |
|------|------|------|
| `create_us_daily_prices.py` | 완료 | S&P 500 가격 데이터 수집 |
| `analyze_volume.py` | 완료 | OBV, A/D Line, MFI 분석 |
| `analyze_13f.py` | 완료 | 기관 보유/인사이더 매매 분석 |
| `analyze_etf_flows.py` | 완료 | ETF 자금 흐름 분석 |
| `smart_money_screener_v2.py` | 완료 | 6팩터 종합 스크리닝 |
| `sector_heatmap.py` | 완료 | 섹터별 히트맵 |
| `options_flow.py` | 완료 | 옵션 플로우 분석 |
| `insider_tracker.py` | 완료 | 인사이더 매매 추적 |
| `portfolio_risk.py` | 완료 | 포트폴리오 리스크 분석 |
| `macro_analyzer.py` | 완료 | 매크로 경제 AI 분석 |
| `ai_summary_generator.py` | 완료 | 종목별 AI 요약 생성 |
| `final_report_generator.py` | 완료 | 최종 Top 10 리포트 |
| `economic_calendar.py` | 완료 | 경제 캘린더 |
| `update_all.py` | 완료 | 통합 파이프라인 |
| `flask_app.py` | 완료 | Flask 웹 서버 + 실시간/히스토리/기술지표 API |
| `templates/index.html` | 완료 | 프론트엔드 UI + JavaScript 로직 |
| `us_config.py` | 완료 | 데이터 경로/캐시/운영 설정 |
| `data_validation.py` | 완료 | 산출물 스키마 검증 및 리포트 |

**보강 완료**:
- `DATA_DIR` 기반 산출물 경로 표준화 및 검증 리포트 생성
- 실시간 가격/차트 캐시, 레이트리밋, 재시도 로직
- 히스토리 성과 요약(평균/중앙/승률) 제공
- 파이프라인 요약 리포트 및 검증 리포트 자동 생성

**남은 작업**:
- 운영 스케줄러(cron/launchd) 구성 및 실패 알림 연동
- 배포 환경별 설정/비밀키 관리 정리
  - 산출물 스키마/필수 컬럼 검증과 누락 시 폴백 처리
  - 데이터 생성 주기/만료 정책 정의
  - 파일 단위 실행 로그/요약 리포트

---

### 3. StockAI (국내 주식 분석 시스템)

**목적**: 한국 주식의 파동 분석 및 AI 기반 투자 의견 생성

**구현 완료 (코드 기준)**:
| 파일 | 상태 | 설명 |
|------|------|------|
| `create_complete_daily_prices.py` | 완료 | 네이버 금융 시세 수집 |
| `all_institutional_trend_data.py` | 완료 | 기관/외국인 수급 분석 |
| `analysis2.py` | 완료 | 4단계 파동 분석 엔진 |
| `investigate_top_stocks.py` | 완료 | Gemini AI 뉴스 분석 |
| `run_analysis.py` | 완료 | 파이프라인 오케스트레이터 |
| `dashboard/app.py` | 완료 | Streamlit 대시보드 |
| `dashboard/utils.py` | 완료 | 유틸리티 함수 |
| `flask_app.py` | 완료 | Flask 웹 서버 + API (추천/성과/검색 등) |
| `templates/index.html` | 완료 | Flask 프론트엔드 UI + JavaScript 로직 |
| `track_performance.py` | 완료 | 추천 성과 추적/리포트 |
| `notifier.py` | 완료 | Telegram/Discord/Slack 알림 |

**추가 보강 필요**:
- 파이프라인 산출물 생성 및 보관 정책 정리 (CSV/리포트/히스토리)
- 스케줄링 자동화 (일일 분석 + 성과 업데이트 + 알림)
- Streamlit/Flask UI 운영 전략 정리 (단일화 또는 역할 분리)

---

## 구현 페이즈 분류 및 상태

### DividendOptimizer
| 페이즈 | 블루프린트/범위 | 상태 | 근거 |
|--------|------------------|------|------|
| Data Step 1/2 | ETF/개별주 유니버스 | 완료 | `data/universe_seed.json` |
| Frontend Step 1 | 랜딩 페이지 | 완료 | `templates/index.html` |
| Frontend Step 2 | 대시보드 레이아웃 | 완료 | `templates/dashboard.html` |
| Frontend Step 3 | 배당 UI 마크업 | 완료 | `templates/dividend.html` |
| Frontend Step 4 | 배당 UI JS 로직 | 완료 | `templates/dividend.html` |
| Backend/Analytics | 최적화/리스크/백테스트 API | 완료 | `engine.py`, `risk_analytics.py`, `backtest.py`, `flask_app.py` |
| 운영화 | 데이터 갱신/성과 저장 | 보강 필요 | 스케줄링/DB 없음 |

### USStockAnalysis
| 페이즈 | 블루프린트/범위 | 상태 | 근거 |
|--------|------------------|------|------|
| Part 1 | 데이터 수집 | 완료 | `create_us_daily_prices.py` |
| Part 2 | 분석/스크리닝 | 완료 | `analyze_volume.py`, `smart_money_screener_v2.py` |
| Part 3 | AI 분석 | 완료 | `macro_analyzer.py`, `ai_summary_generator.py` |
| Part 4 | Web Server | 완료 | `flask_app.py` |
| Part 5 | Frontend UI | 완료 | `templates/index.html` |
| Part 6 | Frontend Logic | 완료 | `templates/index.html` |
| 운영화 | 산출물/스케줄링 | 부분 완료 | 데이터 표준화/검증/요약 완료, 스케줄러 연동 필요 |

### StockAI
| 페이즈 | 범위 | 상태 | 근거 |
|--------|------|------|------|
| Data Collection | 일별 시세 수집 | 완료 | `create_complete_daily_prices.py` |
| Core Analysis | 파동/수급 분석 | 완료 | `analysis2.py`, `all_institutional_trend_data.py` |
| AI Reports | 뉴스/LLM 분석 | 완료 | `investigate_top_stocks.py` |
| Dashboard | Streamlit UI | 완료 | `dashboard/app.py` |
| Web Server | Flask API + UI | 완료 | `flask_app.py`, `templates/index.html` |
| Performance | 성과 추적 | 완료 | `track_performance.py`, `history/` |
| Notifications | 알림 시스템 | 완료 | `notifier.py`, `run_analysis.py` |
| 운영화 | 스케줄링/보관 정책 | 보강 필요 | 자동 실행 및 보관 정책 필요 |

---

## 상세 구현 계획 (보강)

### Project 1: DividendOptimizer
1) **데이터 갱신/캐시 파이프라인**
   - 배당/가격/환율 데이터를 주기적으로 갱신하고 캐시 레이어 추가
   - 실패 시 폴백/에러 리포팅 추가
2) **성과 추적 저장소**
   - 포트폴리오 생성 결과를 SQLite에 저장
   - 조회/비교용 API 및 UI 히스토리 뷰 추가
3) **실시간 가격 최적화**
   - 서버측 캐시/배치 업데이트로 yfinance 호출 수 제한
   - 필요 시 WebSocket 기반 푸시 모드 추가

### Project 2: USStockAnalysis
1) **산출물 경로 표준화** (완료)
   - `DATA_DIR` 기반으로 CSV/JSON 생성 경로 통일
   - 권장 산출물: `smart_money_picks_v2.csv`, `us_volume_analysis.csv`,
     `us_13f_holdings.csv`, `us_etf_flows.csv`, `etf_flow_analysis.json`,
     `macro_analysis.json`, `ai_summaries.json`, `us_sector_heatmap.csv`
   - `flask_app.py` 로딩 경로 정리 및 누락 시 안내 메시지 강화
2) **파이프라인 자동화** (부분 완료)
   - `update_all.py` 실행 결과 검증(파일 존재/스키마 체크)
   - 실행 단계별 결과 요약(JSON/Markdown) 생성
   - 스케줄러(cron/launchd) 연동은 운영 환경에서 추가
3) **API 성능/안정성** (완료)
   - 실시간 가격/차트 데이터 캐시 및 배치 갱신
   - yfinance 호출 재시도/백오프 및 레이트리밋 보호
   - 요청 실패 시 기본값/폴백 처리 정리
4) **히스토리/성과 추적 강화** (완료)
   - `history/picks_YYYY-MM-DD.json` 스키마 유지
   - 히스토리 API에서 평균 수익률/승률 집계 제공
5) **운영 설정 정리** (완료)
   - `.env` 기반 설정값(캐시 TTL, 히스토리 보관일 등) 문서화
   - 실행 가이드에 데이터 경로/리포트 파일 반영

### Project 3: StockAI
1) **산출물 보관 정책**
   - `daily_prices.csv`, `wave_transition_analysis_results.csv` 등 보관 기준 정의
   - 오래된 리포트/히스토리 정리 루틴 추가
2) **자동 실행/알림 운영화**
   - 일일 분석 + 성과 리포트 + 알림 발송 스케줄 구성
   - 알림 템플릿/채널별 On/Off 설정 정리
3) **UI 운영 전략**
   - Streamlit vs Flask 중 주 UI 결정
   - 운영 UI 기준에 맞춰 링크/문서/실행 가이드 정리

---

# 통합 실행 가이드

## 전체 시스템 실행

```bash
# 1. 환경 설정
cd implementations
pip install -r requirements.txt

# 2. 환경변수 설정
cp .env.example .env
# .env 파일에 API 키 입력

# 3. 각 시스템 실행

# [DividendOptimizer]
cd DividendOptimizer
python loader.py          # 배당 데이터 수집
python flask_app.py       # 서버 실행 (port 5001)

# [USStockAnalysis]
cd USStockAnalysis
export DATA_DIR=./data
python update_all.py      # 전체 분석 파이프라인
python flask_app.py       # 서버 실행 (port 5002)

# [StockAI]
cd StockAI
python run_analysis.py    # 전체 분석 파이프라인
streamlit run dashboard/app.py  # Streamlit 대시보드 실행
python flask_app.py       # Flask 대시보드 실행 (port 5003)
```

---

## 테스트 실행

```bash
# [DividendOptimizer]
cd implementations/DividendOptimizer
pytest

# [USStockAnalysis]
cd implementations/USStockAnalysis
export DATA_DIR=./data
pytest

# [StockAI]
cd implementations/StockAI
pytest
```

---

## 에이전트 실행 체크리스트

각 Phase 완료 시 다음을 확인:

- [ ] 코드가 에러 없이 실행되는가?
- [ ] API 응답이 올바른 형식인가?
- [ ] UI가 정상적으로 렌더링되는가?
- [ ] 데이터가 올바르게 표시되는가?

---

## 문서 참조

| 시스템 | 상세 문서 |
|--------|----------|
| DividendOptimizer | `배당/` 폴더의 STEP1~4 문서 |
| USStockAnalysis | `미국 주식/` 폴더의 PART1~6 문서 |
| StockAI | `국내 주식/README.md` |

---

*Last Updated: 2026-01-11*
